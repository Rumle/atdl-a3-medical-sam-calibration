import numpy as np

from experiment_protocol import DATASET_PROTOCOLS


def bootstrap_unit(dataset_id):
    dataset = DATASET_PROTOCOLS.get(dataset_id)
    return "patient" if dataset and dataset.split_by_patient else "case"


def dataset_id_from_rows(rows):
    dataset_ids = {row.get("dataset_id") for row in rows}
    if len(dataset_ids) != 1:
        raise ValueError("Bootstrap input must contain exactly one dataset.")
    return dataset_ids.pop()


def bootstrap_indices(case_ids, dataset_id=None, replicates=2000, seed=1):
    """Resample patients for paired-eye datasets and individual cases otherwise."""
    if replicates < 1 or not case_ids:
        raise ValueError("Bootstrap needs cases and at least one repetition.")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Bootstrap case IDs must be unique.")
    groups = {}
    for index, case_id in enumerate(case_ids):
        if bootstrap_unit(dataset_id) == "patient":
            if not case_id.endswith(("OD", "OS")):
                raise ValueError(f"Patient-grouped eye ID must end in OD or OS: {case_id}")
            group_id = case_id[:-2]
        else:
            group_id = case_id
        groups.setdefault(group_id, []).append(index)

    groups = [np.asarray(indices) for indices in groups.values()]
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        chosen = rng.integers(0, len(groups), size=len(groups))
        yield np.concatenate([groups[index] for index in chosen])
