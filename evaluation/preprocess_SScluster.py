import argparse
import os
from os.path import join
import json
import numpy as np
import torch
import math

from tqdm import tqdm
import warnings
from sklearn.cluster import MeanShift, estimate_bandwidth

warnings.filterwarnings("ignore")

# https://github.com/cvlab-stonybrook/Scanpath_Prediction/issues/24
def scanpath2clusters(meanshift, scanpath):
    string = []
    xs = scanpath['X']
    ys = scanpath['Y']
    for i in range(len(xs)):
        symbol = meanshift.predict([[xs[i], ys[i]]])[0]
        string.append(symbol)
    return string

def improved_rate(meanshift, scanpaths):
    Nc = len(meanshift.cluster_centers_)
    Nb, Nw = 0, 0
    for scanpath in scanpaths:
        string = scanpath2clusters(meanshift, scanpath)
        for i in range(len(string)-1):
            if string[i]==string[i+1]:
                Nw += 1
            else:
                Nb += 1
    return (Nb-Nw)/Nc

def compute_clusters(gt_scanpaths):
    xs, ys = [], []
    for scanpath in gt_scanpaths:
        xs += list(scanpath['X'])
        ys += list(scanpath['Y'])

    gt_gaze = np.concatenate((np.vstack(xs), np.vstack(ys)), axis=1)
    bandwidth = estimate_bandwidth(gt_gaze)
    rates = []
    factors = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8]  # [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    for factor in factors:
        bd = bandwidth * factor if bandwidth > 0.0 else None
        ms = MeanShift(bandwidth=bd)
        ms.fit(gt_gaze)
        rate = improved_rate(ms, gt_scanpaths)
        rates.append(rate)
    rates = np.vstack(rates)

    best_bd = factors[np.argmax(rates)] * bandwidth if bandwidth > 0.0 else None
    best_ms = MeanShift(bandwidth=best_bd)
    best_ms.fit(gt_gaze)

    gt_strings = []
    subjects = []
    for gt_scanpath in gt_scanpaths:
        gt_string = scanpath2clusters(best_ms, gt_scanpath)
        gt_strings.append(gt_string)
        subjects.append(gt_scanpath['username'])

    return best_ms, gt_strings, subjects



fixation_root = '.'
processed_root = '.'

test_json_data = os.path.join(fixation_root, 'test_predictions.json')
test_fixations = json.load(open(test_json_data, "r"))

train_json_data = os.path.join(fixation_root, 'train_scanpath.json')
train_fixations = json.load(open(train_json_data, 'r'))
for idx in range(len(train_fixations)):
    train_fixations[idx]['x'] = [math.floor(x) for x in train_fixations[idx]['x']]
    train_fixations[idx]['y'] = [math.floor(y) for y in train_fixations[idx]['y']]

test_img_list = [test_fixations[idx]['image'] + "_" + test_fixations[idx]['target_id'][4:] for idx in range(len(test_fixations))]
train_img_list = [train_fixations[idx]['image'] +  "_" + train_fixations[idx]['target_id'][4:] for idx in range(len(train_fixations))]

common_num_img = set(train_img_list).intersection(set(test_img_list))
assert len(common_num_img) == 0

target_height = 384
target_width = 512

fixations = test_fixations + train_fixations

data_dict = {}
for scanpath in fixations:
    raw_height = int(scanpath['height'])
    raw_width = int(scanpath['width'])

    image_name = scanpath['image'].split('/')[-1].split('.')[0]
    target_id = scanpath['target_id'][4:]

    key = f'{image_name}-{target_id}'
    # key = '{}-{}'.format(scanpath['split'], scanpath['name'][:-4])
    scanpath["X"] = (np.array(scanpath["x"]) / raw_width * target_width).tolist()
    scanpath["Y"] = (np.array(scanpath["y"]) / raw_height * target_height).tolist()
    data_dict.setdefault(key, []).append(scanpath)


clusters = {}
for key, value in tqdm(data_dict.items()):
    best_ms, gt_strings, subjects = compute_clusters(value)
    strings = {k: v for k, v in zip(subjects, gt_strings)}
    clusters[key] = {
        "strings": strings,
        "cluster": best_ms
    }

np.save(join(processed_root, 'clusters.npy'), clusters, allow_pickle=True)


