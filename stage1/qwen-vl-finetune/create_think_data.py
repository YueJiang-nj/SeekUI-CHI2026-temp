import json
import copy


scanpath_train = json.load(open("scanpath_train.json", "r"))
thinking_process = json.load(open("thinking_prediction.json", "r"))

thinking_process_dict = {}

for line in thinking_process:
    thinking_process_dict[line["img_usr_tgt"]] = line


new_scanpath_train = []
for scanpath in scanpath_train:
    raw_conversations = scanpath["conversations"]
    question = raw_conversations[0]["value"]
    answer = raw_conversations[1]["value"].strip()

    width = int(scanpath["width"])
    height = int(scanpath["height"])
    target = scanpath["target"]

    assert scanpath["img_usr_tgt"] in thinking_process_dict
    thinking_answer = thinking_process_dict[scanpath["img_usr_tgt"]]["thinking"]

    new_conversations = [
        {
            "from": "human",
            "value": f"Given the image with width {width} and height {height}, what is the scanpath for the visual search task on this GUI? The text on the target element is \"{target}\".\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows:\n<think> ... </think> <answer>The scanpath is [x1, y1] [x2, y2] ...</answer>\nPlease strictly follow the format.\n<image>"
        },
        {
            "from": "gpt",
            "value": f"<think>{thinking_answer}</think> <answer>{answer}</answer>"
        }
    ]

    new_scanpath = copy.deepcopy(scanpath)
    new_scanpath["conversations"] = new_conversations

    new_scanpath_train.append(new_scanpath)


json.dump(new_scanpath_train, open("scanpath_train_think.json", "w"), indent=4)