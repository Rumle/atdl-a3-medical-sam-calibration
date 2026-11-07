import zipfile
from dataclasses import dataclass
from pathlib import Path

import gdown
import numpy as np
import torch
from PIL import Image
from skimage.draw import polygon

import shared_inference
from experiment_protocol import DATASET_PROTOCOLS, PROTOCOL, prompt_clicks


DATA_DIR = Path(__file__).resolve().parent / "data"


def download(url: str, path: Path) -> Path:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        partial = path.with_name(path.name + ".partial")
        if url.startswith("https://drive.google.com/"):
            result = gdown.download(url, str(partial), quiet=False, resume=True)
            if result is None:
                raise RuntimeError(f"Download failed: {url}")
        else:
            torch.hub.download_url_to_file(url, str(partial))
        partial.replace(path)
    return path


def prepare_dataset(dataset_id: str, data_dir: Path) -> None:
    dataset = DATASET_PROTOCOLS[dataset_id]
    if data_dir.is_dir():
        discover_dataset_cases(data_dir, dataset_id)
        return
    # Unpack beside the target first so an interrupted run never looks finished.
    partial = data_dir.with_name(data_dir.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    for archive_name, url in dataset.downloads:
        archive = download(url, DATA_DIR / "downloads" / archive_name)
        with zipfile.ZipFile(archive) as contents:
            # PAPILA clinical records are not needed for this segmentation study.
            members = [
                name
                for name in contents.namelist()
                if "__MACOSX" not in name and (
                    dataset_id != "papila"
                    or "/FundusImages/" in name
                    or "/ExpertsSegmentations/Contours/" in name
                )
            ]
            for name in members:
                if not (partial / name).resolve().is_relative_to(partial.resolve()):
                    raise ValueError(f"Archive member escapes the data folder: {name}")
            contents.extractall(partial, members=members)
    if dataset_id == "papila":
        _prepare_papila_masks(partial)
    discover_dataset_cases(partial, dataset_id)
    partial.rename(data_dir)


def _prepare_papila_masks(data_dir: Path) -> None:
    masks = data_dir / "masks"
    masks.mkdir(exist_ok=True)
    for image in data_dir.rglob("FundusImages/*.jpg"):
        contours = image.parent.parent / "ExpertsSegmentations" / "Contours"
        points = np.loadtxt(contours / f"{image.stem}_disc_exp1.txt")
        with Image.open(image) as source:
            shape = (source.height, source.width)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
        ):
            raise ValueError(f"Invalid optic-disc contour for {image.stem}")
        if np.any(points < 0) or np.any(points >= (shape[1], shape[0])):
            raise ValueError(f"Optic-disc contour falls outside image: {image.stem}")
        rows, columns = polygon(points[:, 1], points[:, 0], shape=shape)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[rows, columns] = 255
        Image.fromarray(mask).save(masks / f"{image.stem}.png")


@dataclass(frozen=True)
class DatasetCase:
    case_id: str
    image_path: Path
    mask_paths: tuple[Path, ...]
    required_votes: int


def _validate_case_counts(
    dataset_id: str, cases_by_pool: dict[str, list[DatasetCase]]
) -> None:
    case_ids = [
        case.case_id for cases in cases_by_pool.values() for case in cases
    ]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Duplicate case IDs in {dataset_id}.")
    for pool in DATASET_PROTOCOLS[dataset_id].pools:
        count = len(cases_by_pool[pool.pool_id])
        if count != pool.expected_cases:
            raise ValueError(
                f"{pool.pool_id} must contain {pool.expected_cases} cases; "
                f"found {count}."
            )


def _disc_case(image: Path, mask: Path) -> DatasetCase:
    if not mask.is_file():
        raise FileNotFoundError(f"Missing optic-disc mask: {mask}")
    with Image.open(mask) as source:
        pixels = np.asarray(source)
    if pixels.ndim != 2 or not np.isin(pixels, (0, 1, 255)).all():
        raise ValueError(f"Optic-disc mask must be a binary grayscale PNG: {mask}")
    if not pixels.any():
        raise ValueError(f"Optic-disc mask is empty: {mask}")
    with Image.open(image) as source:
        if pixels.shape != (source.height, source.width):
            raise ValueError(f"Image and mask dimensions disagree for {image.stem}.")
    return DatasetCase(image.stem, image, (mask,), required_votes=1)


def _discover_papila_cases(data_dir: Path) -> dict[str, list[DatasetCase]]:
    cases = {
        "all_488": [
            _disc_case(image, data_dir / "masks" / f"{image.stem}.png")
            for image in sorted(data_dir.rglob("FundusImages/*.jpg"))
        ]
    }
    _validate_case_counts("papila", cases)
    return cases


def _discover_rim_cases(data_dir: Path) -> dict[str, list[DatasetCase]]:
    masks_by_name = {}
    for mask in data_dir.rglob("*-1-Disc-T.png"):
        if mask.name in masks_by_name:
            raise ValueError(f"Duplicate optic-disc mask: {mask.name}")
        masks_by_name[mask.name] = mask
    cases = {}
    for pool, split in (("training_339", "training_set"), ("test_146", "test_set")):
        cases[pool] = []
        for image in sorted(data_dir.rglob(f"partitioned_randomly/{split}/*/*.png")):
            mask_name = f"{image.stem}-1-Disc-T.png"
            if mask_name not in masks_by_name:
                raise FileNotFoundError(f"Missing optic-disc mask for {image.stem}")
            cases[pool].append(_disc_case(image, masks_by_name[mask_name]))
    _validate_case_counts("rim_one_dl", cases)
    return cases


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
    "papila": _discover_papila_cases,
    "rim_one_dl": _discover_rim_cases,
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
    case: DatasetCase,
    prompt_rule: str = "distance",
    prompt_seed: int = 1,
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
        ground_truth,
        count=max(PROTOCOL.click_counts),
        rule=prompt_rule,
        seed=prompt_seed,
        case_id=case.case_id,
    )
    return shared_inference.CaseData(case.case_id, image, ground_truth, clicks)
