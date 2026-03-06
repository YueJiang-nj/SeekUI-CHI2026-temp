# Inference

This directory contains the inference script for scanpath prediction of our SeekUI model.

## Scripts

### `prediction_think.py` — Inference with Explanation

Runs scanpath prediction **with** an explanation step. The model is prompted to output its explanation in `<think>...</think>` tags and the final answer in `<answer>...</answer>` tags. Both the explanation and the predicted scanpath are saved in the output.

## Usage

### Inference with Explanation

```bash
python prediction_think.py \
    --model_path /path/to/SeekUI \
    --scanpath_test /path/to/data/scanpath_test.json \
    --image_root /path/to/data \
    --output /path/to/output/test_predictions_SeekUI.json
```

The predicted scanpath could be found in `evaluation/test_predictions_SeekUI_sft.json` and `evaluation/test_predictions_SeekUI.json`.


## Arguments

Both scripts share the same set of command-line arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_path` | `str` | *(see below)* | Path to the pretrained model checkpoint |
| `--cache_dir` | `str` | *(optional)* | Cache directory for the model |
| `--scanpath_test` | `str` | `/path/to/data/scanpath_test.json` | Path to the cached test scanpath JSON file |
| `--target2text` | `str` | `/path/to/data/target2text.json` | Path to the target-to-text mapping JSON |
| `--test_data_path` | `str` | `/path/to/data/length20_test_text.csv` | Path to the test CSV data file |
| `--image_root` | `str` | `/path/to/data` | Root directory for images |
| `--output` | `str` | `/path/to/output/test_predictions_SeekUI.json` | Path to the output prediction JSON file |
| `--max_new_tokens` | `int` | `512` | Maximum number of new tokens to generate |

## Input Data

- **Test CSV**: A CSV file with columns including `img_usr_tgt`, `image`, `width`, `height`, `username`, `target_id`, `target_x`, `target_y`, `target_width`, `target_height`, `x`, `y`, `t`.
- **target2text.json**: A JSON mapping from target IDs to their text descriptions.
- **Images**: GUI screenshot image folder `vsgui10k-images` stored under `--image_root` directory.

For more details, please check the [Data](../data/Data.md).

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

