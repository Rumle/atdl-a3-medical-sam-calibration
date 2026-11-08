# SAM score calibration across optic-disc datasets

This study checks whether score corrections fitted to SAM and Medical SAM Adapter (Med-SA) predicted IoU at one click remain accurate at two or three clicks. Both segmentation models remain fixed throughout. Calibration adjusts only the predicted scores. The masks themselves stay unchanged.

## Study

SAM ViT-B is tested against the released Med-SA optic-disc adapter on REFUGE Test-400, PAPILA and RIM-ONE DL. Nested positive clicks are simulated from each reference mask. Constant, linear and isotonic corrections are fitted to the scores. Transfer from one-click fitting is compared with corrections fitted at each click count. Pooled, separate and shared-slope linear corrections are also tested.

| Dataset | Calibration fit | Evaluation | Reference mask |
| --- | ---: | ---: | --- |
| REFUGE Test-400 | 300 | 100 | Pixel marked by at least four of seven raters |
| PAPILA | 366 | 122 | Filled expert 1 optic-disc outline, split by patient |
| RIM-ONE DL | 339 | 146 | Supplied optic-disc mask from the official test partition |

Native scores overestimate observed IoU in every model, dataset and click setting. One-click corrections increase three-click error on REFUGE and for SAM on RIM-ONE DL. Transfer gaps are smaller or depend on the correction method on PAPILA and for Med-SA on RIM-ONE DL. A pooled linear correction has higher mean-click MAE than separate corrections in some settings. A shared-slope correction with click-specific intercepts stays within 0.002 of separate mean-click MAE on each dataset.

The REFUGE evaluation set may overlap with Med-SA checkpoint development. We could not verify checkpoint exposure for PAPILA or RIM-ONE DL, so independence is unknown for those datasets.

## Reproduce the experiments

Use Python 3.12. The runner uses CUDA when it is available and otherwise uses the CPU. CPU inference follows the same commands but takes longer.

```bash
uv venv .venv --python 3.12
```

Install the CPU version of PyTorch on a CPU-only machine:

```bash
uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

On a machine with a supported NVIDIA driver, install the CUDA 12.8 version:

```bash
uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining packages and initialize the Medical-SAM-Adapter submodule:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
git submodule update --init --recursive
```

Run each dataset from the repository root:

```bash
.venv/bin/python run_experiment.py --dataset-id refuge2
.venv/bin/python run_experiment.py --dataset-id papila
.venv/bin/python run_experiment.py --dataset-id rim_one_dl
```

The runner downloads missing datasets and checkpoints into `data/`. It reuses matching predictions and runs inference for missing or changed cases. It then runs the native-score, transfer-calibration and pooled-versus-separate analyses. Results are saved under `runs/<dataset_id>/`. RIM-ONE DL data and predictions are not redistributed.

## Repository layout

- `run_experiment.py` runs inference and all analyses for one dataset.
- `experiment_protocol.py`, `dataset_io.py` and `shared_inference.py` define the protocol, load cases and run the fixed models.
- `analysis/` contains the supporting analysis and figure modules.
- `runs/` stores predictions, metrics, tables and figures for each dataset.
