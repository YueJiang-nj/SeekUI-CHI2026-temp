# Inference

This directory contains two inference scripts for scanpath prediction of our SeekUI model.

## Scripts

### `prediction.py` — Inference without Explanation

Runs scanpath prediction **without** an explanation step. The model directly outputs the predicted scanpath coordinates.

### `prediction_think.py` — Inference with Explanation

Runs scanpath prediction **with** an explanation step. The model is prompted to output its explanation in `<think>...</think>` tags and the final answer in `<answer>...</answer>` tags. Both the explanation and the predicted scanpath are saved in the output.

## Usage

### Inference without Explanation

```bash
python prediction.py \
    --model_path /path/to/SeekUI_no_explain \
    --output test_predictions_SeekUI_sft.json
```

### Inference with Explanation

```bash
python prediction_think.py \
    --model_path /path/to/SeekUI \
    --output test_predictions_SeekUI.json
```

The predicted scanpath could be found in `evaluation/test_predictions_SeekUI_sft.json` and `evaluation/test_predictions_SeekUI.json`.


## Arguments

Both scripts share the same set of command-line arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_path` | `str` | *(see below)* | Path to the pretrained model checkpoint |
| `--cache_dir` | `str` | *(optional)* | Cache directory for the model |
| `--scanpath_train` | `str` | `/path/to/scanpath_train.json` | Path to the training scanpath JSON file |
| `--scanpath_test` | `str` | `/path/to/scanpath_test.json` | Path to the cached test scanpath JSON file |
| `--target2text` | `str` | `/path/to/target2text.json` | Path to the target-to-text mapping JSON |
| `--test_data_path` | `str` | `/path/to/length20_test_text.csv` | Path to the test CSV data file |
| `--image_root` | `str` | `/path/to/dataset` | Root directory for images |
| `--output` | `str` | *(see below)* | Path to the output prediction JSON file |
| `--max_new_tokens` | `int` | *(see below)* | Maximum number of new tokens to generate |

### Default Values That Differ Between Scripts

| Argument | `prediction.py` | `prediction_think.py` |
|---|---|---|
| `--model_path` | `SeekUI_no_explain` | `SeekUI` |
| `--output` | `test_predictions_SeekUI_sft.json` | `test_predictions_SeekUI.json` |
| `--max_new_tokens` | `128` | `512` |

## Input Data

- **Test CSV**: A CSV file with columns including `img_usr_tgt`, `image`, `width`, `height`, `username`, `target_id`, `target_x`, `target_y`, `target_width`, `target_height`, `x`, `y`, `t`.
- **target2text.json**: A JSON mapping from target IDs to their text descriptions.
- **Images**: GUI screenshot image folder `vsgui10k-images` stored under `--image_root` directory.

## Output Format

Both scripts output a JSON file. Each entry contains the original test sample fields plus a `prediction` field with the predicted scanpath as a list of `[x, y]` coordinate pairs.

The explanation variant (`prediction_think.py`) additionally includes a `think` field with the model's explanation.

Example output entry:

```json
{
    "img_usr_tgt": "example_id",
    "image": "vsgui10k-images/abc123.png",
    "width": 1280,
    "height": 800,
    "prediction": [[640, 400], [320, 200], [100, 50]],
    "think": "The scanpath begins with ..."
}
```

