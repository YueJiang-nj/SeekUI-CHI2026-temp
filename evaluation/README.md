# Evaluation

This directory contains the evaluation scripts for scanpath prediction.

## 🔧 Preprocessing (Optional)

Before running evaluation, you need to generate `clusters.npy` for the ScanMatch metric:

```bash
python preprocess_SScluster.py
```

This script reads `test_predictions_SeekUI.json` and `train_scanpath.json`, computes MeanShift clusters on the combined fixation data, and saves the result to `clusters.npy`.

> **Note:** We have already provided a pre-computed `clusters.npy`, so you can skip this step and directly run the evaluation scripts below.

## 🚀 Usage

### `evaluation.py` — Overall Evaluation

Evaluate scanpath predictions on the full dataset.

```bash
python evaluation.py --prediction_file <path_to_prediction_json>
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--prediction_file` | str | `test_predictions_SeekUI.json` | Path to the prediction JSON file |

**Example:**

```bash
# Use default prediction file
python evaluation.py

# Specify a custom prediction file
python evaluation.py --prediction_file test_predictions_custom.json
```

**Expected output** (using `test_predictions_SeekUI.json`):

```
sm_score_wo_d  : 0.3479
mm_score_Vec   : 0.9474
mm_score_Dir   : 0.6771
mm_score_Len   : 0.9340
mm_score_Pos   : 0.8552
sed_score      : 5.7760
stde_score     : 0.8606
ss_score       : 0.3206
CC             : 0.4104
AUC            : 0.7474
NSS            : 1.3947
sAUC           : 0.6595
```

---

### `evaluation_types.py` — Per-Type Evaluation

Evaluate scanpath predictions separately for **web**, **desktop**, and **mobile** UI types.

```bash
python evaluation_types.py --prediction_file <path_to_prediction_json> --img2type_file <path_to_img2type_json>
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--prediction_file` | str | `test_predictions_SeekUI.json` | Path to the prediction JSON file |
| `--img2type_file` | str | `/l/dataset/VIS_GUI/dataset/img_to_type.json` | Path to the image-to-type mapping JSON file |

**Example:**

```bash
# Use default files
python evaluation_types.py

# Specify custom files
python evaluation_types.py --prediction_file test_predictions_custom.json --img2type_file /path/to/img_to_type.json
```

## 📂 Baseline Model Predictions

Prediction JSON files from other baseline models are available on Google Drive:

🔗 [**Download Baseline Predictions**](https://drive.google.com/drive/folders/1OyK6dwvAY9n33uMIkAM-LEbBd-IoUFfe?usp=sharing)

Available models:
- **Chen**
- **Eyeformer**
- **Gazeformer**

Download the corresponding JSON file and pass it via `--prediction_file` to run evaluation.
