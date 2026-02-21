import time
from datasets import DatasetDict, Dataset
from PIL import Image
import json
import os
import math
from tqdm import tqdm


"""
turn your json to DatasetDict
"""
def json_to_dataset(json_file_path):
    # read json file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    image_paths, problems, solutions = [], [], []
    images = []
    resolutions = []

    for item in tqdm(data, total=len(data)):

        image_path = item['image']
        width = int(item['width'])
        height = int(item['height'])
        target = item['target']
        resolution = [width, height]

        problem = f'''Find the target object that is "{target}" in the image, and provide the trajectory of fixation points matching human behavior to find this target object (x-coordinate between 0 and {width}, y-coordinate between 0 and {height}, integer; the first fixation point is near the middle of the image).
        If no target object is '{target}' in the image, return 'No Objects'.
        Output the thinking process in <think> </think>, and final answer of fixation points in <answer> </answer> tags.The output answer format should be as follows:
        <think> ... </think> <answer>[{{'Position': [x, y]}}, ...]</answer>
        Please strictly follow the format.'''


        solution_list = []
        for x_value, y_value in zip(item['x'], item['y']):
            x_int = math.floor(x_value)
            y_int = math.floor(y_value)
            line = {'Position': [x_int, y_int]}
            solution_list.append(line)


        solution = f"<answer>{solution_list}</answer>"

        image_paths.append(os.path.join(image_base, image_path))
        problems.append(problem)
        solutions.append(solution)
        images.append(Image.open(os.path.join(image_base, image_path)).convert('RGBA'))
        resolutions.append(resolution)

    # image_paths = [item['image_path'] for item in data]
    # problems = [item['problem'] for item in data]
    # solutions = [item['solution'] for item in data]
    #
    # images = [Image.open(image_path).convert('RGBA') for image_path in image_paths]

    dataset_dict = {
        'image': images,
        'image_path': image_paths,
        'problem': problems,
        'solution': solutions,
        'resolution': resolutions
    }

    dataset = Dataset.from_dict(dataset_dict)
    dataset_dict = DatasetDict({
        'train': dataset
    })
    return dataset_dict




image_base = "/l/dataset/VIS_GUI/dataset"


time1 = time.asctime()
print(time1)
### Your dataset in JSON file format consists of three parts: image, problem and solution


dataset_dict = json_to_dataset('./scanpath_train.json')
time2 = time.asctime()
print(time2)

"""
save to your local disk
"""
def save_dataset(dataset_dict, save_path):
    # save DatasetDict to your disk
    dataset_dict.save_to_disk(save_path)

save_path = './vis_gui_train'
save_dataset(dataset_dict, save_path)


"""
read from your local disk
"""
def load_dataset(save_path):
    # load DatasetDict
    return DatasetDict.load_from_disk(save_path)

test_dataset_dict = load_dataset('./vis_gui_train')
print()