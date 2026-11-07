# Med-SA score calibration study

We compare the native quality scores from original SAM and a released Medical SAM Adapter (Med-SA) checkpoint with the actual IoU of their optic-disc masks on REFUGE Test-400, PAPILA and RIM-ONE DL.

The pipeline includes native-score analysis, linear score corrections, click-count transfer and pooled-versus-separate corrections.

## Research question

Does a score correction fitted at one click count (one, two, or three clicks) also work at the other click counts? Does this differ between SAM and Med-SA?

We expect a correction to work best at the click count it was fitted on, and to work worse the further the target click count is from it. For example, a one-click correction should transfer better to two clicks than to three. We keep the segmentation checkpoints fixed. We do not train a segmentation model or reproduce Med-SA's training or full evaluation.

The REFUGE results remain exploratory. Med-SA reports REFUGE2 and the upstream REFUGE code can use Test-400 to select a checkpoint. No case-level record of the data used to train or select the released adapter is available so exposure to these cases cannot be confirmed. Evaluation cases are kept separate from calibration fitting and validation yet independence from checkpoint development cannot be verified. They are labelled `internal_evaluation_overlap_risk`.

## Experiment setup

Two models are compared:

- `base_sam`: the official SAM ViT-B checkpoint.
- `med_sa_adapter`: the same base checkpoint together with the released Med-SA optic-disc adapter.

Every REFUGE case supplies an image plus seven optic-disc rater masks. A pixel is treated as foreground in the consensus mask when at least four raters mark it as foreground. Each image is resized so that its longer side equals 1024 pixels. Mask token 0 is used together with its matching predicted-IoU score.

Prompts contain one, two or three positive clicks. Each larger prompt includes the smaller ones so the clicks are added sequentially. The clicks are simulated from the ground-truth mask. By default the first click lands on the foreground pixel farthest from the boundary. Each subsequent click lands on a remaining foreground pixel farthest from the earlier clicks. The alternative rule samples foreground pixels randomly with a fixed per-case seed. These are simulated prompts rather than clicks recorded from clinicians.

REFUGE cases are assigned to roles by case ID with fixed random seeds:

| Role | Cases | Purpose |
| --- | ---: | --- |
| `calibration_fit` | 200 | Fit score corrections |
| `validation` | 100 | Choose methods or settings |
| `internal_evaluation_overlap_risk` | 100 | Current REFUGE evaluation |

The native-score analysis uses only the evaluation cases. It neither fits nor selects a method.

PAPILA uses expert 1's optic-disc outline and a patient-level split of 244 calibration-fit, 122 validation and 122 final-test eyes. RIM-ONE DL uses its supplied optic-disc masks and official random partition with 254 calibration-fit, 85 validation and 146 final-test cases. Independence from checkpoint development is unverified for both datasets.

Shared settings live in [`experiment_protocol.py`](experiment_protocol.py). Saved predictions are reused for a case only when the settings, inputs, checkpoints, and code all match so a change of setting forces inference to be rerun for the affected cases.

| Setting | Value |
| --- | --- |
| Sampling, split, model, and bootstrap seeds | `1` |
| Bootstrap replicates | `2000` |
| Image size | 1024 pixels on the longer side |
| Click counts | 1, 2, and 3 positive clicks |
| Ground truth | Dataset-specific rule stored in the dataset protocol |
| Mask and score | Token 0 and its matching predicted-IoU score |

## Current native-score results

For every case the clipped model score \(q_i\) is compared against the observed mask IoU \(y_i\):

- **MAE** equals the average absolute difference between the score and the observed IoU.
- **RMSE** equals the square root of the average squared difference.
- **Signed bias** equals the average of \(q_i-y_i\). A positive value means the score exceeds the observed IoU.
- **Coarse ECE** equals the weighted average absolute gap between the score and observed IoU means across ten equal-width bins between 0 and 1.

MAE and signed bias have percentile 95% confidence intervals from 2,000 bootstrap repetitions. PAPILA resamples its 61 evaluation patients and includes both eyes each time a patient is drawn. Other datasets resample individual cases. Paired method comparisons use the same draws. Confidence intervals for paired differences between the models are not reported. The table below lists native-score results for the 100 REFUGE evaluation cases stored in [`runs/refuge2/native_summary.json`](runs/refuge2/native_summary.json). These scores have not been calibrated.

| Model | Clicks | MAE (95% CI) | Signed bias (95% CI) | Coarse ECE |
| --- | ---: | ---: | ---: | ---: |
| Original SAM | 1 | 0.379 (0.335–0.423) | +0.353 (+0.300–+0.404) | 0.357 |
| Original SAM | 2 | 0.345 (0.299–0.395) | +0.298 (+0.237–+0.359) | 0.300 |
| Original SAM | 3 | 0.313 (0.255–0.377) | +0.252 (+0.183–+0.325) | 0.256 |
| Med-SA adapter | 1 | 0.246 (0.205–0.291) | +0.200 (+0.149–+0.253) | 0.208 |
| Med-SA adapter | 2 | 0.142 (0.112–0.175) | +0.105 (+0.070–+0.143) | 0.113 |
| Med-SA adapter | 3 | 0.089 (0.068–0.114) | +0.028 (−0.001–+0.059) | 0.056 |

On REFUGE both models showed declines in MAE, bias and ECE as more clicks were added and the decline was larger for Med-SA. These native-score figures alone do not establish calibration transfer or performance on independent data. Full results are saved in [`runs/refuge2/`](runs/refuge2/), [`runs/papila/`](runs/papila/) and [`runs/rim_one_dl/`](runs/rim_one_dl/).

Inference was executed on an RTX 5080 GPU. The saved summary originates from a later rerun that reused all 400 cached predictions and therefore captures neither GPU use nor inference time. The inference metadata stores the checkpoint hashes, the dataset archive hash, and the inference-script hash.

## Running the experiment

Inference requires CUDA. The setup that was tested used Python 3.12 together with PyTorch 2.11.0 and CUDA 12.8 wheels. [`requirements.txt`](requirements.txt) does not pin every dependency. Pillow and certain transitive dependencies remain unpinned so the installed package versions should be recorded when the experiment is run. The NVIDIA driver must support the installed PyTorch CUDA runtime.

Create the environment and verify that PyTorch can access the GPU:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

Clone the repository with the upstream Medical-SAM-Adapter submodule or initialize the submodule in an existing clone:

```bash
git submodule update --init --recursive
```

We used submodule revision `5888e722876511f177e8e762498a7986753b7f7d`.

The dataset and the checkpoints are not included in this repository. On startup the runner downloads any missing public files into `data/` which Git ignores:

| File | Saved to | Source | SHA-256 |
| --- | --- | --- | --- |
| `sam_vit_b_01ec64.pth` | `data/checkpoints/` | [SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |
| `OpticDisc_Fundus_SAM_1024.pth` | `data/checkpoints/` | [Med-SA adapter](https://huggingface.co/KidsWithTokens/Medical-Adapter-Zoo/resolve/main/OpticDisc_Fundus_SAM_1024.pth) | `a1e39b25aa17e71fe18a6776bd771fc862b7d28c7562e574646473effdff1af1` |
| `REFUGE-Multirater.zip` | `data/downloads/`, unpacked to `data/refuge2/` | [REFUGE data](https://huggingface.co/datasets/realslimman/REFUGE-MultiRater/resolve/main/REFUGE-Multirater.zip) | Recorded in the run metadata |

PAPILA and RIM-ONE DL also download automatically into `data/downloads/` and unpack into `data/papila/` and `data/rim_one_dl/`. Their sources and mask rules are stored in [`experiment_protocol.py`](experiment_protocol.py). RIM-ONE forbids redistribution of its data, so its cached prediction files containing reference masks are also Git-ignored.

The REFUGE data should contain one `Test-400` folder with 400 cases. Each case needs a `.jpg` image and seven masks named `<case_id>_seg_disc_<rater>.png`. The runner checks the number of cases and the checkpoint hashes against the protocol.

Run one dataset at a time. Each dataset writes to `runs/<dataset_id>/`:

```bash
.venv/bin/python run_experiment.py --dataset-id refuge2
.venv/bin/python run_experiment.py --dataset-id papila
.venv/bin/python run_experiment.py --dataset-id rim_one_dl
```

Running the same command again reuses every saved prediction that still matches and runs inference only for missing or changed cases. It rebuilds the native-score analysis and both group calibration analyses. Corrections are fitted within each dataset, not transferred between datasets.

## Code and output files

Each inference run contains:

- `summary.json` with the cases, roles, seeds, GPU, and run times.
- `metadata.json` with the protocol, checkpoint hashes, and exposure note.
- `predictions/<case_id>.npz` with the consensus mask, clicks, model masks and scores, and case details.

The native analysis produces `shared_case_results.csv`, `native_case_results.csv`, `native_metrics.csv`, `native_reliability.png` and `native_summary.json`. Group outputs go in `member4_analysis/` and `pooled_vs_separate/` inside the same run folder. They include fitted corrections, metrics, paired comparisons and analysis settings.

The metadata records the inference-script hash yet does not record the full Git working tree or a complete package lockfile. When results are shared the Git commit, any local changes, and the installed package versions should also be recorded. This project does not train a model so no training time needs to be reported.

## Calibration analyses

The linear method fits a straight-line correction by least squares and clips the result to the range from 0 to 1:

$$
\hat{y} = \operatorname{clip}(a q + b, 0, 1)
$$

Here \(q\) is the clipped native predicted-IoU score, \(a\) and \(b\) are fitted coefficients and \(\hat{y}\) is the corrected score. The fitting target \(y\) is observed mask IoU.

[`member4_analysis.py`](member4_analysis.py) compares per-click linear corrections with native scores and calibration means. It measures one-to-three-click transfer and supports paired click-rule comparisons. [`pooled_vs_separate.py`](pooled_vs_separate.py) compares pooled and per-click linear corrections with budget-matched and one/three-click controls.

Both analyses use calibration-fit and evaluation cases, leaving validation unused. Bootstrap intervals hold corrections fixed. The isotonic transfer script is not present locally.

## Sources

- Junde Wu et al., “Medical SAM adapter: Adapting segment anything model for medical image segmentation.” *Medical Image Analysis*, 102, 103547 (2025). [Publisher record](https://doi.org/10.1016/j.media.2025.103547).
- Kirillov et al., “Segment Anything.” [arXiv:2304.02643](https://arxiv.org/abs/2304.02643).
- [Project GitHub repository](https://github.com/Rumle/atdl-a3-medical-sam-calibration).
