import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch

from experiment_protocol import (
    DATASET_PROTOCOLS,
    PROTOCOL,
    DatasetPoolProtocol,
    DatasetProtocol,
    assign_case_roles,
    select_case_ids,
)
from dataset_io import (
    DATA_DIR,
    DATASET_GROUND_TRUTH_RULES,
    DatasetCase,
    discover_dataset_cases,
    download,
    load_dataset_case,
    prepare_dataset,
)
from native_reliability import analyze_predictions
from shared_inference import (
    UPSTREAM_DIR,
    run_model,
    save_predictions,
    sha256_file,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ASSET_DIR = DATA_DIR / "checkpoints"
DATASET_ARCHIVE = DATA_DIR / "downloads" / PROTOCOL.dataset_archive_name
CACHE_SCHEMA_VERSION = 1


def _hash_case_inputs(
    cases_by_id: dict[str, DatasetCase], selected_case_ids: list[str]
) -> str:
    inputs = []
    for case_id in selected_case_ids:
        case = cases_by_id[case_id]
        inputs.append(
            {
                "case_id": case_id,
                "files": [
                    {
                        "name": path.name,
                        "sha256": sha256_file(path),
                    }
                    for path in (case.image_path, *case.mask_paths)
                ],
            }
        )
    serialized = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _upstream_source_hash() -> str:
    digest = hashlib.sha256()
    source_files = sorted(UPSTREAM_DIR.rglob("*.py"))
    for path in source_files:
        digest.update(path.relative_to(UPSTREAM_DIR).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _build_cache_identity(
    cases_by_id: dict[str, DatasetCase],
    selected_case_ids: list[str],
    role_case_ids: dict[str, list[str]],
    case_roles: dict[str, dict[str, str]],
    dataset_protocol: DatasetProtocol,
    source_pools: tuple[DatasetPoolProtocol, ...],
    case_count: int,
    sampling_seed: int,
    split_seed: int,
    base_checkpoint_sha256: str,
    adapter_checkpoint_sha256: str,
) -> dict[str, object]:
    upstream_revision = subprocess.run(
        ["git", "-C", str(UPSTREAM_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime_versions = {
        "python": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in (
                "torch",
                "torchvision",
                "numpy",
                "scipy",
                "Pillow",
                "timm",
            )
        },
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    source_hashes = {
        name: sha256_file(PROJECT_DIR / name)
        for name in (
            "run_experiment.py",
            "shared_inference.py",
            "experiment_protocol.py",
            "dataset_io.py",
        )
    }
    source_hashes["upstream_python"] = _upstream_source_hash()
    identity = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_id": dataset_protocol.dataset_id,
        "dataset_split": " + ".join(pool.official_name for pool in source_pools),
        "dataset_protocol": dataset_protocol.as_dict(),
        "protocol": {**PROTOCOL.as_dict(), "dataset_id": dataset_protocol.dataset_id},
        "run": {
            "case_count": case_count,
            "sampling_seed": sampling_seed,
            "split_seed": split_seed,
        },
        "selected_case_ids": selected_case_ids,
        "case_role_ids": role_case_ids,
        "case_roles": case_roles,
        "case_input_sha256": _hash_case_inputs(cases_by_id, selected_case_ids),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "adapter_checkpoint_sha256": adapter_checkpoint_sha256,
        "source_hashes": source_hashes,
        "upstream_revision": upstream_revision,
        "runtime_versions": runtime_versions,
    }
    return json.loads(json.dumps(identity, sort_keys=True))


def _cache_fingerprint(identity: dict[str, object]) -> str:
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _case_cache_fingerprints(
    identity: dict[str, object], cases_by_id: dict[str, DatasetCase]
) -> dict[str, str]:
    run_fields = {
        "run",
        "selected_case_ids",
        "case_role_ids",
        "case_roles",
        "case_input_sha256",
    }
    shared_identity = {
        key: value for key, value in identity.items() if key not in run_fields
    }
    return {
        case_id: _cache_fingerprint(
            {
                **shared_identity,
                "case_id": case_id,
                "input_sha256": _hash_case_inputs(
                    cases_by_id, [case_id]
                ),
                "case_role": identity["case_roles"][case_id],
            }
        )
        for case_id in identity["selected_case_ids"]
    }


def _load_cached_case_ids(
    inference_dir: Path,
    case_fingerprints: dict[str, str],
    identity: dict[str, object],
) -> tuple[set[str], dict[str, dict[str, str]]]:
    metadata_path = inference_dir / "metadata.json"
    if not metadata_path.is_file():
        return set(), {}

    try:
        with metadata_path.open() as file:
            metadata = json.load(file)
    except json.JSONDecodeError as error:
        print(f"Ignoring invalid inference cache metadata: {error}")
        return set(), {}
    if not isinstance(metadata, dict):
        return set(), {}
    entries = metadata.get("prediction_cache", {})
    if not isinstance(entries, dict):
        return set(), {}
    if not entries:
        old_identity = metadata.get("cache_identity")
        summary_path = inference_dir / "summary.json"
        if isinstance(old_identity, dict) and summary_path.is_file():
            old_base = {
                key: value
                for key, value in old_identity.items()
                if key != "source_hashes"
            }
            current_base = {
                key: value for key, value in identity.items() if key != "source_hashes"
            }
            old_sources = old_identity.get("source_hashes")
            current_sources = identity.get("source_hashes")
            if isinstance(old_sources, dict) and isinstance(current_sources, dict):
                old_sources = dict(old_sources)
                current_sources = dict(current_sources)
                old_sources.pop("run_experiment.py", None)
                current_sources.pop("run_experiment.py", None)
                if (
                    old_base == current_base
                    and all(
                        current_sources.get(name) == digest
                        for name, digest in old_sources.items()
                        if name != "run_experiment.py"
                    )
                    and metadata.get("cache_fingerprint")
                    == _cache_fingerprint(old_identity)
                    and metadata.get("summary_sha256") == sha256_file(summary_path)
                ):
                    old_hashes = metadata.get("prediction_sha256", {})
                    if isinstance(old_hashes, dict):
                        entries = {
                            case_id: {
                                "fingerprint": case_fingerprints[case_id],
                                "sha256": old_hashes[case_id],
                            }
                            for case_id in case_fingerprints
                            if isinstance(old_hashes.get(case_id), str)
                        }

    reusable = {}
    prediction_cache = {
        case_id: entry
        for case_id, entry in entries.items()
        if isinstance(case_id, str)
        and isinstance(entry, dict)
        and isinstance(entry.get("fingerprint"), str)
        and isinstance(entry.get("sha256"), str)
    }
    prediction_dir = inference_dir / "predictions"
    for case_id, fingerprint in case_fingerprints.items():
        entry = prediction_cache.get(case_id)
        prediction_path = prediction_dir / f"{case_id}.npz"
        if (
            isinstance(entry, dict)
            and entry.get("fingerprint") == fingerprint
            and prediction_path.is_file()
            and entry.get("sha256") == sha256_file(prediction_path)
        ):
            reusable[case_id] = entry
    return set(reusable), prediction_cache


def write_shared_predictions(
    data_dir: Path,
    asset_dir: Path,
    output_dir: Path,
    case_count: int | None,
    sampling_seed: int,
    split_seed: int,
    dataset_id: str = PROTOCOL.dataset_id,
    prompt_rule: str = "distance",
    prompt_seed: int = 1,
) -> dict[str, object]:
    base_path = asset_dir / PROTOCOL.base_checkpoint_name
    adapter_path = asset_dir / PROTOCOL.adapter_checkpoint_name
    if not base_path.is_file() or not adapter_path.is_file():
        raise FileNotFoundError(
            f"SAM checkpoints are missing from {asset_dir}. Run run_experiment.py "
            "to download them or set --asset-dir."
        )
    base_checkpoint_sha256 = sha256_file(base_path)
    adapter_checkpoint_sha256 = sha256_file(adapter_path)
    if base_checkpoint_sha256 != PROTOCOL.base_checkpoint_sha256:
        raise ValueError(
            "SAM checkpoint checksum does not match the shared protocol: "
            f"{base_path}"
        )
    if adapter_checkpoint_sha256 != PROTOCOL.adapter_checkpoint_sha256:
        raise ValueError(
            "Med-SA checkpoint checksum does not match the shared protocol: "
            f"{adapter_path}"
        )

    if dataset_id not in DATASET_PROTOCOLS:
        raise ValueError(f"Unknown dataset protocol: {dataset_id}")
    if case_count is not None and case_count < 2:
        raise ValueError("case_count must select at least two cases per pool.")
    dataset_protocol = DATASET_PROTOCOLS[dataset_id]
    cases_by_pool = discover_dataset_cases(data_dir, dataset_id)
    if set(cases_by_pool) != {pool.pool_id for pool in dataset_protocol.pools}:
        raise ValueError(f"{dataset_id} cases do not match its declared source pools.")

    selected_by_pool = {}
    case_by_id = {}
    for pool in dataset_protocol.pools:
        cases = cases_by_pool[pool.pool_id]
        if len(cases) != pool.expected_cases:
            raise ValueError(
                f"Expected {pool.expected_cases} {pool.official_name} cases, "
                f"found {len(cases)}."
            )
        count = min(case_count if case_count is not None else len(cases), len(cases))
        selected_by_pool[pool.pool_id] = select_case_ids(
            [case.case_id for case in cases], count, sampling_seed
        )
        case_by_id.update({case.case_id: case for case in cases})
    selected_ids = [
        case_id
        for pool in dataset_protocol.pools
        for case_id in selected_by_pool[pool.pool_id]
    ]
    assignments = assign_case_roles(
        dataset_id,
        selected_by_pool,
        split_seed,
    )
    assignments_by_id = {assignment.case_id: assignment for assignment in assignments}
    role_case_ids = {
        role: [
            assignment.case_id
            for assignment in assignments
            if assignment.analysis_role == role
        ]
        for pool in dataset_protocol.pools
        for role, _ in pool.role_fractions
    }
    case_roles = {
        assignment.case_id: {
            "source_pool": assignment.source_pool,
            "analysis_role": assignment.analysis_role,
        }
        for assignment in assignments
    }
    cache_identity = _build_cache_identity(
        case_by_id,
        selected_ids,
        role_case_ids,
        case_roles,
        dataset_protocol,
        dataset_protocol.pools,
        case_count if case_count is not None else len(selected_ids),
        sampling_seed,
        split_seed,
        base_checkpoint_sha256,
        adapter_checkpoint_sha256,
    )
    if prompt_rule not in ("distance", "random_foreground"):
        raise ValueError(f"Unknown prompt rule: {prompt_rule}")
    prompt_settings = {
        "rule": prompt_rule, "seed": prompt_seed,
        "description": PROTOCOL.prompt_rule if prompt_rule == "distance" else (
            "Uniform positive foreground sampling without replacement; nested "
            "clicks, stable per-case seed, identical points for both models."
        ),
    }
    # Top-level field is retained by _case_cache_fingerprints.
    cache_identity["prompt_settings"] = prompt_settings
    case_fingerprints = _case_cache_fingerprints(cache_identity, case_by_id)
    cached_case_ids, prediction_cache = _load_cached_case_ids(
        output_dir, case_fingerprints, cache_identity
    )
    missing_ids = [
        case_id for case_id in selected_ids if case_id not in cached_case_ids
    ]
    print(
        f"Reusing {len(cached_case_ids)} cached case(s); "
        f"running inference for {len(missing_ids)} new or changed case(s)."
    )
    sampled_cases = [
        load_dataset_case(case_by_id[case_id], prompt_rule, prompt_seed)
        for case_id in missing_ids
    ]
    evaluation_role = (
        "final_test"
        if "final_test" in role_case_ids
        else "internal_evaluation_overlap_risk"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements = {}
    if sampled_cases:
        device = torch.device("cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for the shared SAM and Med-SA inference."
            )

        predictions = {}
        for model_name in PROTOCOL.models:
            predictions[model_name], measurements[model_name] = run_model(
                model_name, sampled_cases, device, base_path, adapter_path
            )

        save_predictions(
            output_dir,
            sampled_cases,
            predictions,
            {
                case_id: assignments_by_id[case_id]
                for case_id in missing_ids
            },
        )
        for case_id in missing_ids:
            prediction_path = output_dir / "predictions" / f"{case_id}.npz"
            prediction_cache[case_id] = {
                "fingerprint": case_fingerprints[case_id],
                "sha256": sha256_file(prediction_path),
            }
    else:
        measurements = {
            model_name: {"images": 0, "inference_seconds": 0.0}
            for model_name in PROTOCOL.models
        }
    summary = {
        "schema_version": 2,
        "dataset_id": dataset_id,
        "dataset_split": " + ".join(
            pool.official_name for pool in dataset_protocol.pools
        ),
        "sample_count": len(selected_ids),
        "inference_new_count": len(missing_ids),
        "inference_reused_count": len(cached_case_ids),
        "case_role_ids": role_case_ids,
        "case_roles": case_roles,
        "role_counts": {
            role: len(case_ids) for role, case_ids in role_case_ids.items()
        },
        "calibration_count": len(role_case_ids.get("calibration_fit", [])),
        "validation_count": len(role_case_ids.get("validation", [])),
        "evaluation_role": evaluation_role,
        "evaluation_count": len(role_case_ids.get(evaluation_role, [])),
        "sampling_seed": sampling_seed,
        "partition_seed": split_seed,
        "model_seed": PROTOCOL.model_seed,
        "ground_truth": DATASET_GROUND_TRUTH_RULES[dataset_id],
        "gpu": (
            torch.cuda.get_device_name(device)
            if sampled_cases
            else "not used (predictions cached)"
        ),
        "measurements": measurements,
        "prompt_rule": prompt_settings["description"],
        "prompt_settings": prompt_settings,
        "mask_selection": PROTOCOL.mask_selection,
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    metadata = {
        "protocol": {**PROTOCOL.as_dict(), "dataset_id": dataset_id,
                     "prompt_rule": prompt_settings["description"]},
        "prompt_settings": prompt_settings,
        "dataset_protocol": dataset_protocol.as_dict(),
        "adapter_exposure_status": dataset_protocol.adapter_exposure_status,
        "run": {
            "case_count": (
                case_count if case_count is not None else len(selected_ids)
            ),
            "sampling_seed": sampling_seed,
            "split_seed": split_seed,
            "role_counts": summary["role_counts"],
        },
        "selected_case_ids": selected_ids,
        "dataset_archive_sha256": (
            sha256_file(DATASET_ARCHIVE)
            if dataset_id == "refuge2" and DATASET_ARCHIVE.is_file()
            else None
        ),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "adapter_checkpoint_sha256": adapter_checkpoint_sha256,
        "base_checkpoint_path": str(base_path.resolve()),
        "adapter_checkpoint_path": str(adapter_path.resolve()),
        "inference_script_path": str((PROJECT_DIR / "shared_inference.py").resolve()),
        "inference_script_sha256": sha256_file(PROJECT_DIR / "shared_inference.py"),
        "upstream_repo_path": str(UPSTREAM_DIR.resolve()),
        "cache_identity": cache_identity,
        "cache_fingerprint": _cache_fingerprint(cache_identity),
        "summary_sha256": sha256_file(output_dir / "summary.json"),
        "prediction_sha256": {
            case_id: sha256_file(output_dir / "predictions" / f"{case_id}.npz")
            for case_id in selected_ids
        },
        "prediction_cache": prediction_cache,
        "adapter_training_overlap": dataset_protocol.adapter_exposure_evidence,
    }
    metadata_tmp_path = output_dir / "metadata.json.tmp"
    with metadata_tmp_path.open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    metadata_tmp_path.replace(output_dir / "metadata.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM and Med-SA on the same cases and measure how well their "
            "predicted-IoU scores match observed IoU."
        ),
        epilog=(
            "Drishti-GS1 uses case_manifest.csv with case_id, source_pool, "
            "image_path, and optic_disc_mask columns. Its masks must be binary. "
            "A case-count cap applies separately to each source pool."
        ),
    )
    parser.add_argument(
        "--dataset-id", choices=tuple(DATASET_PROTOCOLS), default=PROTOCOL.dataset_id
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--case-count",
        type=int,
        help="Maximum number of cases selected from each source pool.",
    )
    parser.add_argument("--sampling-seed", type=int, default=PROTOCOL.sampling_seed)
    parser.add_argument("--split-seed", type=int, default=PROTOCOL.split_seed)
    parser.add_argument("--prompt-rule", choices=("distance", "random_foreground"),
                        default="distance")
    parser.add_argument("--prompt-seed", type=int, default=1)
    args = parser.parse_args()

    data_dir = args.data_dir or DATA_DIR / args.dataset_id
    prepare_dataset(args.dataset_id, data_dir)
    download(
        PROTOCOL.base_checkpoint_url, args.asset_dir / PROTOCOL.base_checkpoint_name
    )
    download(
        PROTOCOL.adapter_checkpoint_url,
        args.asset_dir / PROTOCOL.adapter_checkpoint_name,
    )
    default_run = PROJECT_DIR / "runs" / args.dataset_id
    if args.prompt_rule != "distance":
        default_run = PROJECT_DIR / "runs" / (
            f"{args.dataset_id}_{args.prompt_rule}_seed{args.prompt_seed}"
        )
    run_dir = args.run_dir or default_run
    run_dir = run_dir.resolve()
    inference_dir = run_dir / "inference"
    write_shared_predictions(
        data_dir.resolve(),
        args.asset_dir.resolve(),
        inference_dir,
        args.case_count,
        args.sampling_seed,
        args.split_seed,
        args.dataset_id,
        args.prompt_rule,
        args.prompt_seed,
    )

    result = analyze_predictions(
        inference_dir,
        run_dir,
        replicates=PROTOCOL.bootstrap_replicates,
        seed=PROTOCOL.bootstrap_seed,
    )
    for row in result["metrics"]:
        print(
            f"{row['model']} {row['click_count']} click(s): "
            f"MAE={row['mae']:.3f} "
            f"(95% CI {row['mae_ci_low']:.3f}-{row['mae_ci_high']:.3f}), "
            f"bias={row['bias']:+.3f} "
            f"(95% CI {row['bias_ci_low']:+.3f}-{row['bias_ci_high']:+.3f})"
        )
    print(f"Saved shared predictions and native-score summary to {run_dir}.")


if __name__ == "__main__":
    main()
