import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import shared_inference
from experiment_protocol import DATASET_PROTOCOLS, PROTOCOL, prompt_clicks


DATA_DIR = Path(__file__).resolve().parent / "data"
# Drishti has no public link (it needs a login), so its files go in data/ by hand.
DATASET_DOWNLOADS = {"refuge2": PROTOCOL.dataset_url}


def download(url: str, path: Path) -> Path:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        torch.hub.download_url_to_file(url, str(path))
    return path


def prepare_dataset(dataset_id: str, data_dir: Path) -> None:
    url = DATASET_DOWNLOADS.get(dataset_id)
    if url is None or data_dir.is_dir():
        return
    archive = download(url, DATA_DIR / "downloads" / PROTOCOL.dataset_archive_name)
    # Unpack beside the target first so an interrupted run never looks finished.
    partial = data_dir.with_name(data_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    shutil.unpack_archive(archive, partial)
    partial.rename(data_dir)


@dataclass(frozen=True)
class DatasetCase:
    case_id: str
    image_path: Path
    mask_paths: tuple[Path, ...]
    required_votes: int


DATASET_GROUND_TRUTH_RULES = {
    "refuge2": PROTOCOL.ground_truth_rule,
    "drishti_gs1": "Foreground is any nonzero pixel in the binary optic-disc mask.",
}

_DRISHTI_MANIFEST_COLUMNS = (
    "case_id",
    "source_pool",
    "image_path",
    "optic_disc_mask",
)


def _validate_case_counts(
    dataset_id: str, cases_by_pool: dict[str, list[DatasetCase]]
) -> None:
    for pool in DATASET_PROTOCOLS[dataset_id].pools:
        count = len(cases_by_pool[pool.pool_id])
        if count != pool.expected_cases:
            raise ValueError(
                f"{pool.pool_id} must contain {pool.expected_cases} cases; "
                f"found {count}."
            )


def _manifest_path(root: Path, value: str, column: str) -> Path:
    relative_path = Path(value)
    if not value or relative_path.is_absolute():
        raise ValueError(
            f"{column} must be a non-empty path relative to the data root."
        )

    root = root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{column} must name a file under the data root: {value}")
    return path


def _discover_drishti_cases(data_dir: Path) -> dict[str, list[DatasetCase]]:
    manifest_path = data_dir / "case_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Drishti-GS1 needs a registered manual download. Place the files "
            f"and case_manifest.csv in {data_dir}."
        )
    cases_by_pool = {
        pool.pool_id: [] for pool in DATASET_PROTOCOLS["drishti_gs1"].pools
    }
    seen_ids = set()

    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(_DRISHTI_MANIFEST_COLUMNS):
            raise ValueError(
                "case_manifest.csv must have columns "
                + ", ".join(_DRISHTI_MANIFEST_COLUMNS)
                + " in that order."
            )

        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Each manifest row must contain exactly four values.")
            case_id = row["case_id"].strip()
            pool_id = row["source_pool"].strip()
            if not case_id:
                raise ValueError("Manifest case IDs must not be empty.")
            if pool_id not in cases_by_pool:
                raise ValueError(f"Unknown Drishti source pool: {pool_id}")
            key = (pool_id, case_id)
            if key in seen_ids:
                raise ValueError(f"Duplicate case ID in {pool_id}: {case_id}")
            seen_ids.add(key)

            image_path = _manifest_path(
                data_dir, row["image_path"].strip(), "image_path"
            )
            mask_value = row["optic_disc_mask"].strip()
            if Path(mask_value).suffix.casefold() != ".png":
                raise ValueError("Drishti optic-disc masks must be PNG files.")
            mask_path = _manifest_path(data_dir, mask_value, "optic_disc_mask")

            with Image.open(mask_path) as mask_file:
                mask = np.asarray(mask_file)
            if mask.ndim != 2 or not np.isin(mask, (0, 1, 255)).all():
                raise ValueError(
                    f"Drishti mask for {case_id} must be a binary PNG with "
                    "values 0, 1, or 255."
                )
            with Image.open(image_path) as image_file:
                image_shape = (image_file.height, image_file.width)
            if image_shape != mask.shape:
                raise ValueError(f"Image and mask dimensions disagree for {case_id}.")

            cases_by_pool[pool_id].append(
                DatasetCase(
                    f"{pool_id}-{case_id}",
                    image_path,
                    (mask_path,),
                    required_votes=1,
                )
            )

    missing_pools = set(cases_by_pool) - {
        pool_id for pool_id, cases in cases_by_pool.items() if cases
    }
    if missing_pools:
        raise ValueError(f"Missing Drishti source pools: {sorted(missing_pools)}")
    _validate_case_counts("drishti_gs1", cases_by_pool)
    return cases_by_pool


def _discover_refuge_cases(data_dir: Path) -> dict[str, list[DatasetCase]]:
    pool_id = DATASET_PROTOCOLS["refuge2"].pools[0].pool_id
    refuge_cases = shared_inference.discover_test_cases(data_dir)
    cases_by_pool = {
        pool_id: [
            DatasetCase(case.case_id, case.image_path, case.rater_paths, 4)
            for case in refuge_cases
        ]
    }
    _validate_case_counts("refuge2", cases_by_pool)
    return cases_by_pool


_DATASET_DISCOVERERS = {
    "refuge2": _discover_refuge_cases,
    "drishti_gs1": _discover_drishti_cases,
}


def discover_dataset_cases(
    data_dir: Path, dataset_id: str
) -> dict[str, list[DatasetCase]]:
    if dataset_id not in DATASET_PROTOCOLS:
        raise ValueError(f"Unknown dataset protocol: {dataset_id}")
    try:
        discover = _DATASET_DISCOVERERS[dataset_id]
    except KeyError as error:
        raise ValueError(f"No dataset loader is available for: {dataset_id}") from error
    return discover(Path(data_dir))


def load_dataset_case(
    case: DatasetCase, prompt_rule: str = "distance", prompt_seed: int = 1,
) -> shared_inference.CaseData:
    if not case.mask_paths or not 1 <= case.required_votes <= len(case.mask_paths):
        raise ValueError(f"Invalid mask vote rule for {case.case_id}.")

    with Image.open(case.image_path) as image_file:
        image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
    masks = []
    for mask_path in case.mask_paths:
        with Image.open(mask_path) as mask_file:
            masks.append(np.asarray(mask_file.convert("L"), dtype=np.uint8) > 0)
    if any(mask.shape != masks[0].shape for mask in masks):
        raise ValueError(f"Mask dimensions disagree for {case.case_id}.")

    ground_truth = np.stack(masks).sum(axis=0) >= case.required_votes
    if image.shape[:2] != ground_truth.shape:
        raise ValueError(f"Image and mask dimensions disagree for {case.case_id}.")
    clicks = prompt_clicks(
        ground_truth, count=max(PROTOCOL.click_counts), rule=prompt_rule,
        seed=prompt_seed, case_id=case.case_id,
    )
    return shared_inference.CaseData(case.case_id, image, ground_truth, clicks)
