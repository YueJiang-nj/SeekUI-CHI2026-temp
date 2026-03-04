# Data Configuration

## Overview

The SeekUI framework uses a two-stage training pipeline. Both stages use the **VSGUI** eye-tracking scanpath dataset, containing human visual search trajectories on GUI screenshots, but with different data formats.

## Download

1. Download `vsgui10k-images` from [Google Drive](https://drive.google.com/drive/folders/1Qbrwa6uZqRxgcwyWTF0bVZCEYkP7xTWK?usp=sharing).
2. Download `scanpath_train_explanation.json` from [Google Drive](https://drive.google.com/file/d/1ZIlf3GTTqXn-_kE8DBy-F1VV8QhAlBRh/view?usp=sharing).
3. Download `target2text.json` from [Google Drive](https://drive.google.com/file/d/1pLHVWtbS3y6jWDTmwYmmQrXzKFKwWZDl/view?usp=sharing).

Place them under the `data/` directory:

```
data/
├── scanpath_train_explanation.json
├── target2text.json
└── vsgui10k-images/
    ├── f4b47d.png
    ├── c74797.png
    ├── b908c7.png
    └── ...
```

## Explanation Data Format

Each sample in the JSON file contains the following fields:

```json
{
    "img_usr_tgt": "f4b47d_d325c0_txt_LxDYaO4WnT",  
    "image": "vsgui10k-images/f4b47d.png",         
    "width": 1185.0,                          
    "height": 668.0,                          
    "username": "d325c0",   
    "target_id": "txt_LxDYaO4WnT",           
    "target_x": 506.22,                                 // Target bounding box center x
    "target_y": 117.39,                                 // Target bounding box center y
    "target_width": 29.10,                              // Target bounding box width
    "target_height": 34.28,                             // Target bounding box height
    "x": [592.85, 565.53, ...],                         // Fixation x-coordinates
    "y": [358.41, 216.21, ...],                         // Fixation y-coordinates
    "t": [0.258, 0.226, ...],                           // Fixation durations (seconds)
    "target": "N",                                      // Target element text label
    "conversations": [...]                              // Conversation pairs
}
```

---

## Stage 1: Instruction Tuning (SFT)

### Conversation Format (with thinking)

**Human prompt:**
```
Given the image with width {width} and height {height}, what is the scanpath for the visual search task on this GUI? The text on the target element is "{target}".
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows:
<think> ... </think> <answer>The scanpath is [x1, y1] [x2, y2] ...</answer>
Please strictly follow the format.
<image>
```

**GPT response:**
```
<think>{explanation_process}</think> <answer>The scanpath is [x1, y1] [x2, y2] ...</answer>
```

### Data Preparation

**Configure data path**: Update `PATH_TO_DATA` in `stage1/qwen-vl-finetune/qwenvl/data/__init__.py`:

    ```python
    VSGUI_TEXT = {
        "annotation_path": "PATH_TO_DATA/data/scanpath_train_explanation.json",
        "data_path": "PATH_TO_DATA/data",
        "target": "PATH_TO_DATA/data/target2text.json",
    }
    ```



## Stage 2: Reinforcement Learning (GRPO)

### Source Files

| File | Location | Description |
|------|----------|-------------|
`build_visgui_dataset.py` | `stage2/dataset/` | Script to convert JSON to HuggingFace Dataset format |

### GRPO Input Format

The `build_visgui_dataset.py` script converts the raw JSON into a HuggingFace `DatasetDict`:

| Field | Type | Description |
|-------|------|-------------|
| `image` | `Image` | PIL Image object (RGBA) |
| `image_path` | `str` | Absolute path to the image file |
| `problem` | `str` | The problem prompt for the model |
| `solution` | `str` | The ground truth solution |
| `resolution` | `list[int]` | `[width, height]` of the image |

**Problem prompt template:**
```
Find the target object that is "{target}" in the image, and provide the trajectory of fixation points matching human behavior to find this target object (x-coordinate between 0 and {width}, y-coordinate between 0 and {height}, integer; the first fixation point is near the middle of the image).
If no target object is '{target}' in the image, return 'No Objects'.
Output the thinking process in <think> </think>, and final answer of fixation points in <answer> </answer> tags. The output answer format should be as follows:
<think> ... </think> <answer>[{'Position': [x, y]}, ...]</answer>
Please strictly follow the format.
```

**Solution format:**
```
<answer>[{'Position': [x1, y1]}, {'Position': [x2, y2]}, ...]</answer>
```

### Data Preparation

1. **Configure image base path** in `build_visgui_dataset.py`:

    ```python
    image_base = "/path/to/your/dataset"  # Directory containing vsgui10k-images/
    ```

2. **Build the HuggingFace Dataset**:

    ```bash
    cd stage2/dataset
    python build_visgui_dataset.py
    ```

    This reads `scanpath_train.json`, converts each sample to GRPO format, and saves to `./vis_gui_train/`.

Alternatively, download the pre-processed dataset directly from HuggingFace: [sushizixin1/vsgui_train_seekui_qwen2.5_explanation](https://huggingface.co/datasets/sushizixin1/vsgui_train_seekui_qwen2.5_explanation).


### Supported Dataset Formats

SeekUI supports one input format:
1. **HuggingFace Dataset** — generated by `build_visgui_dataset.py` or downloaded from HuggingFace (recommended)
