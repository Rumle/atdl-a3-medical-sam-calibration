import hashlib
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class DatasetPoolProtocol:
    pool_id: str
    official_name: str
    expected_cases: int
    annotation_access: str
    role_fractions: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class DatasetProtocol:
    dataset_id: str
    display_name: str
    target: str
    source: str
    runtime_status: str
    adapter_exposure_status: str
    adapter_exposure_evidence: str
    pools: tuple[DatasetPoolProtocol, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CaseAssignment:
    dataset_id: str
    source_pool: str
    analysis_role: str
    case_id: str


DATASET_PROTOCOLS = {
    "refuge2": DatasetProtocol(
        dataset_id="refuge2",
        display_name="REFUGE2",
        target="optic_disc",
        source="Medical-SAM-Adapter benchmark and official REFUGE split",
        runtime_status="available",
        adapter_exposure_status="possible_overlap",
        adapter_exposure_evidence=(
            "Med-SA includes REFUGE2 in its reported benchmarks. The upstream "
            "REFUGE code can use Test-400 to choose a checkpoint. The released "
            "optic-disc adapter has no case-level training or selection manifest."
        ),
        pools=(
            DatasetPoolProtocol(
                pool_id="test_400",
                official_name="REFUGE Test-400",
                expected_cases=400,
                annotation_access="available",
                role_fractions=(
                    ("calibration_fit", 0.5),
                    ("validation", 0.25),
                    ("internal_evaluation_overlap_risk", 0.25),
                ),
            ),
        ),
    ),
    "drishti_gs1": DatasetProtocol(
        dataset_id="drishti_gs1",
        display_name="Drishti-GS1",
        target="optic_disc",
        source="Official Drishti-GS1 dataset",
        runtime_status="not_available_in_repository",
        adapter_exposure_status="unknown",
        adapter_exposure_evidence=(
            "Drishti-GS1 was not listed among the Med-SA benchmark datasets we "
            "checked. We do not have the exact checkpoint case history, so we "
            "cannot confirm that these cases are independent."
        ),
        pools=(
            DatasetPoolProtocol(
                pool_id="training_50",
                official_name="Training-50",
                expected_cases=50,
                annotation_access="available",
                role_fractions=(("calibration_fit", 0.6), ("validation", 0.4)),
            ),
            DatasetPoolProtocol(
                pool_id="test_51",
                official_name="Test-51",
                expected_cases=51,
                annotation_access="registration_required",
                role_fractions=(("final_test", 1.0),),
            ),
        ),
    ),
}


@dataclass(frozen=True)
class ExperimentProtocol:
    dataset_id: str = "refuge2"
    sampling_seed: int = 1
    split_seed: int = 1
    model_seed: int = 1
    bootstrap_seed: int = 1
    bootstrap_replicates: int = 2000
    models: tuple[str, ...] = ("base_sam", "med_sa_adapter")
    click_counts: tuple[int, ...] = (1, 2, 3)
    image_size: int = 1024
    dataset_url: str = (
        "https://huggingface.co/datasets/realslimman/REFUGE-MultiRater/resolve/main/"
        "REFUGE-Multirater.zip"
    )
    dataset_archive_name: str = "REFUGE-Multirater.zip"
    base_checkpoint_name: str = "sam_vit_b_01ec64.pth"
    base_checkpoint_url: str = (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    )
    base_checkpoint_sha256: str = (
        "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
    )
    adapter_checkpoint_url: str = (
        "https://huggingface.co/KidsWithTokens/Medical-Adapter-Zoo/resolve/main/"
        "OpticDisc_Fundus_SAM_1024.pth"
    )
    adapter_checkpoint_name: str = "OpticDisc_Fundus_SAM_1024.pth"
    adapter_checkpoint_sha256: str = (
        "a1e39b25aa17e71fe18a6776bd771fc862b7d28c7562e574646473effdff1af1"
    )
    ground_truth_rule: str = (
        "Pixelwise majority vote from seven raters. Mark a pixel as foreground "
        "when at least four raters do."
    )
    prompt_rule: str = (
        "Choose positive clicks from the consensus mask. The first click is the "
        "foreground pixel farthest from the boundary. Each later click is the "
        "remaining foreground pixel farthest from the earlier clicks. Smaller "
        "prompts use the first points of the three-click prompt."
    )
    prompt_source: str = (
        "Simulated from the ground truth. These are not clinician interactions."
    )
    mask_selection: str = (
        "Use the output from mask token 0 and its matching predicted-IoU score."
    )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


PROTOCOL = ExperimentProtocol()


def assign_case_roles(
    dataset_id: str,
    cases_by_pool: dict[str, list[str]],
    seed: int,
) -> list[CaseAssignment]:
    """Assign each case to a role using the source pools defined for its dataset."""
    if dataset_id not in DATASET_PROTOCOLS:
        raise ValueError(f"Unknown dataset protocol: {dataset_id}")
    dataset = DATASET_PROTOCOLS[dataset_id]
    expected_pool_ids = {pool.pool_id for pool in dataset.pools}
    if set(cases_by_pool) != expected_pool_ids:
        raise ValueError(
            f"{dataset_id} requires source pools {sorted(expected_pool_ids)}."
        )

    all_case_ids = [
        case_id for pool in dataset.pools for case_id in cases_by_pool[pool.pool_id]
    ]
    if any(not case_id for case_id in all_case_ids):
        raise ValueError("Case IDs must not be empty.")
    if len(all_case_ids) != len(set(all_case_ids)):
        raise ValueError("Case IDs must be unique across source pools.")

    assignments = []
    for pool in dataset.pools:
        case_ids = cases_by_pool[pool.pool_id]
        if not case_ids or len(case_ids) > pool.expected_cases:
            raise ValueError(
                f"{pool.official_name} must contain between 1 and "
                f"{pool.expected_cases} cases."
            )
        fractions = np.asarray([fraction for _, fraction in pool.role_fractions])
        if (
            not len(fractions)
            or np.any(fractions <= 0)
            or not np.isclose(fractions.sum(), 1.0)
        ):
            raise ValueError(f"Invalid role fractions for {pool.official_name}.")
        if len(case_ids) < len(pool.role_fractions):
            raise ValueError(
                f"{pool.official_name} needs at least one case per declared role."
            )

        exact_sizes = fractions * len(case_ids)
        sizes = np.floor(exact_sizes).astype(int)
        remainder = len(case_ids) - int(sizes.sum())
        remainder_order = np.argsort(-(exact_sizes - sizes), kind="stable")
        sizes[remainder_order[:remainder]] += 1
        if np.any(sizes == 0):
            raise ValueError(
                f"{pool.official_name} does not have enough cases for every role."
            )

        ordered_ids = sorted(case_ids)
        if len(pool.role_fractions) > 1:
            order = np.random.default_rng(seed).permutation(len(ordered_ids))
            ordered_ids = [ordered_ids[index] for index in order]
        start = 0
        for (role, _), size in zip(pool.role_fractions, sizes):
            for case_id in ordered_ids[start : start + size]:
                assignments.append(
                    CaseAssignment(dataset_id, pool.pool_id, role, case_id)
                )
            start += size

    return assignments


def select_case_ids(case_ids: list[str], count: int, seed: int) -> list[str]:
    """Pick a reproducible group of cases before splitting it for calibration and evaluation."""
    case_ids = sorted(case_ids)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Case IDs must be unique.")
    if not 2 <= count <= len(case_ids):
        raise ValueError("count must leave at least two cases and cannot exceed the cohort.")
    if count == len(case_ids):
        return case_ids
    indices = np.random.default_rng(seed).choice(len(case_ids), count, replace=False)
    return [case_ids[index] for index in sorted(indices)]


def split_case_ids(
    case_ids: list[str], fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split cases into calibration and evaluation groups in a reproducible way."""
    if len(case_ids) < 2 or len(set(case_ids)) != len(case_ids):
        raise ValueError("At least two unique case IDs are required.")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1.")
    calibration_size = int(round(len(case_ids) * fraction))
    if not 0 < calibration_size < len(case_ids):
        raise ValueError("The split must leave cases in both subsets.")
    order = np.random.default_rng(seed).permutation(len(case_ids))
    calibration = [case_ids[index] for index in order[:calibration_size]]
    evaluation = [case_ids[index] for index in order[calibration_size:]]
    return calibration, evaluation


def prompt_clicks(
    mask: np.ndarray, count: int, rule: str = "distance", seed: int = 1,
    case_id: str = "",
) -> np.ndarray:
    """Choose nested positive points from the ground-truth mask using the fixed distance rule."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("The click mask must be a non-empty 2D foreground mask.")
    if count < 1 or count > int(mask.sum()):
        raise ValueError("count must be between one and the number of foreground pixels.")

    if rule not in ("distance", "random_foreground"):
        raise ValueError(f"Unknown prompt rule: {rule}")
    if rule == "random_foreground":
        if not case_id:
            raise ValueError("Random prompts require a stable case ID.")
        # Stable across processes and independent of case ordering and role.
        digest = hashlib.sha256(f"{seed}:{case_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:16], "little"))
        foreground = np.flatnonzero(mask)
        # Repeated draws preserve prefixes even if count changes.
        selected = []
        for _ in range(count):
            index = int(rng.integers(len(foreground)))
            selected.append(foreground[index])
            foreground = np.delete(foreground, index)
        ys, xs = np.unravel_index(selected, mask.shape)
        return np.column_stack((xs, ys)).astype(np.int64)

    boundary_distance = distance_transform_edt(mask)
    first_flat = int(np.argmax(boundary_distance))
    first_y, first_x = np.unravel_index(first_flat, mask.shape)
    clicks = [(first_x, first_y)]
    rows, columns = np.indices(mask.shape)
    available = mask.copy()
    available[first_y, first_x] = False

    while len(clicks) < count:
        nearest_click = np.full(mask.shape, np.inf, dtype=np.float64)
        for x, y in clicks:
            nearest_click = np.minimum(
                nearest_click, (columns - x) ** 2 + (rows - y) ** 2
            )
        scores = np.where(available, nearest_click, -1)
        next_y, next_x = np.unravel_index(int(np.argmax(scores)), mask.shape)
        clicks.append((next_x, next_y))
        available[next_y, next_x] = False

    return np.asarray(clicks, dtype=np.int64)
