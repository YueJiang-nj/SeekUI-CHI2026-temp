import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

VSGUI_TEXT = {
    # "annotation_path": "/scratch/work/guoz3/data/VIS_GUI/dataset/fixations/length20_train_text.csv",
    "annotation_path": "/scratch/work/guoz3/CHI2025/Qwen2.5-VL-Eye-Tracking-NLP-thinking/qwen-vl-finetune/scanpath_train_think.json",
    "data_path": "/scratch/work/guoz3/data/VIS_GUI/dataset",
    "target": "/scratch/work/guoz3/data/VIS_GUI/dataset/fixations/target2text.json",
    "mean": "./vsgui_text_mean.json",
    "std": "./vsgui_text_std.json",
}

# VSGUI_TEXT = {
#     "annotation_path": "/l/dataset/VIS_GUI/dataset/fixations/length20_train_text.csv",
#     "data_path": "/l/dataset/VIS_GUI/dataset",
#     "target": "/l/dataset/VIS_GUI/dataset/fixations/target2text.json",
#     "mean": "./vsgui_text_mean.json",
#     "std": "./vsgui_text_std.json",
# }

CC3M = {
    "annotation_path": "/l/dataset/cc3m/cc3m_examples_captions.json",
    "data_path": "/l/dataset/cc3m/cc3m_examples",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "cc3m": CC3M,
    "vsgui_text": VSGUI_TEXT,
}

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
