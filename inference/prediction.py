from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import re
import json

import csv
import os
import math
from tqdm import tqdm


def load_csv(csv_file, target2text):
    scanpath = {}

    if "text" in os.path.basename(csv_file):
        target_type = "text"
    elif "image" in os.path.basename(csv_file):
        target_type = "image"
    else:
        raise NotImplementedError

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            img_usr_tgt = row['img_usr_tgt']
            image = f"vsgui10k-images/{row['image']}.png"
            width = float(row['width'])
            height = float(row['height'])
            username = row['username']
            target_id = row['target_id']
            target_x = float(row['target_x'])
            target_y = float(row['target_y'])
            target_width = float(row['target_width'])
            target_height = float(row['target_height'])
            x = float(row['x'])
            y = float(row['y'])
            t = float(row['t'])

            if img_usr_tgt not in scanpath:
                scanpath[img_usr_tgt] = {
                    "img_usr_tgt": img_usr_tgt,
                    "image": image,
                    "width": width,
                    "height": height,
                    "username": username,
                    "target_id": target_id,
                    "target_x": target_x,
                    "target_y": target_y,
                    "target_width": target_width,
                    "target_height": target_height,
                    "x": [],
                    "y": [],
                    "t": [],
                    "target": target2text[target_id[4:]] if target_type == "text" else None
                }

            scanpath[img_usr_tgt]["x"].append(math.floor(x))
            scanpath[img_usr_tgt]["y"].append(math.floor(y))
            scanpath[img_usr_tgt]["t"].append(t)

    scanpath = list(scanpath.values())
    json.dump(scanpath, open("qwen-vl-finetune/scanpath_test.json", 'w'))
    return scanpath


# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto", cache_dir="/l/env/blip3o/hub"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/l/CHI/pretrained_model/checkpoints_nlp_full_grpo_HiT",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir="/l/env/blip3o/hub"
)
# /l/CHI/pretrained_model/checkpoints_nlp_full
# /l/CHI/pretrained_model/checkpoints_nlp_full_grpo_final
# /l/CHI/pretrained_model/checkpoints_nlp_full_grpo_HiT
# /l/CHI/pretrained_model/checkpoints_nlp_think_full_grpo_HiT
# default processer
processor = AutoProcessor.from_pretrained("/l/CHI/pretrained_model/checkpoints_nlp_full_grpo_HiT", cache_dir="/l/env/blip3o/hub")

# chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
# processor.chat_template = chat_template

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "Given the image with width 1280 and height 800, what is the scanpath for the visual search task on this GUI? The text on the target element is \"Family Law\".\n"},
#                     # "text": "Where is the target element of \"Family Law\" in this image? Output the position of x and y.\n"},
#                 {
#                     "type": "image",
#                     "image": "/l/dataset/VIS_GUI/dataset/vsgui10k-images/b908c7.png",
#                 },
#
#             ],
#         }
#     ]


scanpath_train = json.load(open("qwen-vl-finetune/scanpath_train.json", "r"))
target2txt = json.load(open("/l/dataset/VIS_GUI/dataset/fixations/target2text.json", "r"))


test_data_path = "/l/dataset/VIS_GUI/dataset/fixations/length20_test_text.csv"
if not os.path.exists("qwen-vl-finetune/scanpath_test.json"):
    scanpath_test = load_csv(test_data_path, target2txt)
else:
    scanpath_test = json.load(open("qwen-vl-finetune/scanpath_test.json", "r"))

scanpath_example = scanpath_test


results = []
for idx in tqdm(range(len(scanpath_example)), total=len(scanpath_example)):

    example_width, example_height = int(scanpath_example[idx]["width"]), int(scanpath_example[idx]["height"])
    example_image = scanpath_example[idx]["image"]
    example_target_id = scanpath_example[idx]["target_id"][4:]
    example_target = target2txt[example_target_id]
    example_target_x = math.floor(scanpath_example[idx]["target_x"] + scanpath_example[idx]["target_width"] / 2)
    example_target_y = math.floor(scanpath_example[idx]["target_y"] + scanpath_example[idx]["target_height"] / 2)

    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given the image with width {example_width} and height {example_height}, what is the scanpath for the visual search task on this GUI? The text on the target element is \"{example_target}\".\n"},
                    {
                        "type": "image",
                        "image": f"/l/dataset/VIS_GUI/dataset/{example_image}",
                    },
                ],
            }
        ]


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_text[0])

    # Load the image using PIL
    image = Image.open(messages[0]["content"][1]["image"]).convert("RGB")

    # Extract scanpath points from the sentence
    sentence = output_text[0]
    sentence = sentence.replace("[ ", "[")
    sentence = sentence.replace(" ]", "]")
    sentence = sentence.replace(",", " ")
    points = re.findall(r'\[(\d+) \s*(\d+)\]', sentence)
    points = [[int(x), int(y)] for x, y in points]
    assert len(points) != 0

    cur_example = scanpath_example[idx]
    cur_example["prediction"] = points
    results.append(cur_example)

json.dump(results, open("test_predictions_grpo_HiT.json", "w"))