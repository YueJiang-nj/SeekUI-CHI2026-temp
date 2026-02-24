import json
import argparse
from evaluation.evaluator import Evaluator
import numpy as np


def test_prediction_form(predictions):
    for idx, line in enumerate(predictions):
        prediction = line["prediction"]
        for pred in prediction:
            try:
                assert len(pred) == 2
            except AssertionError:
                prediction.remove(pred)
                if len(prediction) == 0:
                    prediction.append([int(line["width"] // 2), int(line["height"] // 2)])


def process_scores(scores, sal_scores=None):

    sm_score_wo_d = []
    sm_score_w_d = []
    mm_score_Vec = []
    mm_score_Dir = []
    mm_score_Len = []
    mm_score_Pos = []
    mm_score_Dur = []
    sed_score = []
    stde_score = []
    ss_score = []
    ss_score_time = []
    sss_score = []
    sss_score_time = []
    for score in scores:
        sm_score_wo_d.append(score["scanmatch_score"][0])
        sm_score_w_d.append(score["scanmatch_score"][1])

        mm_score_Vec.append(score["multimatch_score"][0])
        mm_score_Dir.append(score["multimatch_score"][1])
        mm_score_Len.append(score["multimatch_score"][2])
        mm_score_Pos.append(score["multimatch_score"][3])
        mm_score_Dur.append(score["multimatch_score"][4])

        sed_score.append(score["sed_score"])
        stde_score.append(score["stde_score"])

        ss_score.append(score["SS_score"][0])
        ss_score_time.append(score["SS_score"][1])

        if "SSS_score" in score:
            sss_score.append(score["SSS_score"][0])
            sss_score_time.append(score["SSS_score"][1])

    cc_score = []
    auc_score = []
    nss_score = []
    sauc_score = []
    kld_score = []
    sim_score = []
    if sal_scores is not None:
        for sal_score in sal_scores.values():
            cc_score.append(sal_score[0])
            auc_score.append(sal_score[1])
            nss_score.append(sal_score[2])
            sauc_score.append(sal_score[3])
            kld_score.append(sal_score[4])
            sim_score.append(sal_score[5])

    metrics = {
        "sm_score_wo_d": np.mean(sm_score_wo_d),
        # "sm_score_w_d": np.mean(sm_score_w_d),
        "mm_score_Vec": np.mean(mm_score_Vec),
        "mm_score_Dir": np.mean(mm_score_Dir),
        "mm_score_Len": np.mean(mm_score_Len),
        "mm_score_Pos": np.mean(mm_score_Pos),
        # "mm_score_Dur": np.mean(mm_score_Dur),
        "sed_score": np.mean(sed_score),
        "stde_score": np.mean(stde_score),
        "ss_score": np.mean(ss_score),
        # "ss_score_time": np.mean(ss_score_time),
        # "sss_score": np.mean(sss_score) if "SSS_score" in score else 0.0,
        # "sss_score_time": np.mean(sss_score_time) if "SSS_score" in score else 0.0,
        "CC": np.mean(cc_score) if sal_scores is not None else 0.0,
        "AUC": np.mean(auc_score) if sal_scores is not None else 0.0,
        "NSS": np.mean(nss_score) if sal_scores is not None else 0.0,
        "sAUC": np.mean(sauc_score) if sal_scores is not None else 0.0,
        # "KLD": np.mean(kld_score) if sal_scores is not None else 0.0,
        # "SIM": np.mean(sim_score) if sal_scores is not None else 0.0,
    }

    return metrics


def eval_coco_on_single_gpu(model, valset):
    width = 512
    height = 384

    raw_width = 512
    raw_height = 320

    model.eval()
    loader = dataloader.DataLoader(valset, batch_size=args.batch_size, drop_last=False,
                                   num_workers=args.num_workers, collate_fn=valset.special_collate)
    print("Data loaded")

    results = []
    user_ids = []
    stop_ids = []
    image_sizes = []
    raw_image_sizes = []
    dataset_idices = []
    fixation_infos = []
    image_names = []

    with torch.no_grad():
        for data in tqdm(loader, mininterval=60, disable=True):
            src_imgs = data['src_img'].cuda()
            tgt_imgs = data['tgt_img']

            image_size = data['image_size']
            raw_image_size = data['raw_image_size']

            user_id = data['user_id']
            dataset_idx = data['dataset_idx']
            fixation_info = data['fixation_info']

            image_name = ["%s_%s" % (info["name"], info["task"]) for info in fixation_info]

            coord, pred_stops = model.generate(src_imgs, tgt_imgs, max_length=args.max_length, target_type=args.target_type)

            coord[:, :, 0] = de_normalize(coord[:, :, 0], x_mean, x_std)
            coord[:, :, 1] = de_normalize(coord[:, :, 1], y_mean, y_std)
            coord[:, :, 2] = de_normalize(coord[:, :, 2], t_mean, t_std)

            coord = coord.cpu().numpy().tolist()
            pred_stops = pred_stops.cpu().numpy().tolist()

            user_id = user_id.cpu().numpy().tolist()
            dataset_idx = dataset_idx.cpu().numpy().tolist()

            results.extend(coord)
            user_ids.extend(user_id)
            stop_ids.extend(pred_stops)

            image_sizes.append(image_size)
            raw_image_sizes.append(raw_image_size)
            dataset_idices.extend(dataset_idx)
            fixation_infos.extend(fixation_info)
            image_names.extend(image_name)


    predictions = []
    ground_truths = []

    with open(os.path.join(args.model_dir, 'predicted_result.csv'), 'w') as wfile:
        writer = csv.writer(wfile)
        writer.writerow(["image", "width", "height", "username", "x", "y", "timestamp"])

        for image, user_id, coord, stop_tag, fixation_info in zip(image_names, user_ids,
                                                                                results, stop_ids, fixation_infos):

            if 1 in stop_tag:
                first_stop_pos = stop_tag.index(1) + 1
                last_stop_pos = len(stop_tag) - stop_tag[::-1].index(1)
            else:
                first_stop_pos = len(stop_tag)
                last_stop_pos = len(stop_tag)

            scanpath = {"X": [], "Y": [], "T": []}
            target = {"X": [], "Y": [], "T": []}

            for row in coord[:first_stop_pos]:
                x = row[0] * width
                y = row[1] * height
                t = row[2]
                username = user_id
                writer.writerow([image, width, height, username,
                                 x, y, t])
                scanpath["X"].append(x)
                scanpath["Y"].append(y)
                scanpath["T"].append(t*1000.)

            fixation_length = fixation_info["length"]
            target["X"] = fixation_info["X"][:fixation_length]
            target["Y"] = fixation_info["Y"][:fixation_length]
            target["T"] = fixation_info["T"][:fixation_length]

            target["X"] = [x / raw_width * width for x in target["X"]]
            target["Y"] = [y / raw_height * height for y in target["Y"]]

            predictions.append(scanpath)
            ground_truths.append(target)

    return predictions, ground_truths, fixation_infos


def process_predictions(raw_predictions):
    predictions = []
    ground_truths = []
    fixation_infos = []

    for line in raw_predictions:
        pred = {}
        gt = {}

        x = np.array([e[0] for e in line["prediction"]]) / line["width"] * 512
        y = np.array([e[1] for e in line["prediction"]]) / line["height"] * 384

        pred["X"] = x.tolist()
        pred["Y"] = y.tolist()
        pred["T"] = [0.] * len(x)
        predictions.append(pred)

        gt_x = np.array(line["x"]) / line["width"] * 512
        gt_y = np.array(line["y"]) / line["height"] * 384
        gt["X"] = gt_x.tolist()
        gt["Y"] = gt_y.tolist()
        time = np.array(line["t"]) * 1000.
        gt["T"] = time.tolist()
        ground_truths.append(gt)

        image_name = line['image'].split('/')[-1].split('.')[0]
        target_id = line['target_id'][4:]
        fixation_info = {"image_name": image_name, "target_id": target_id, "subject": line["username"], "T": time}

        fixation_infos.append(fixation_info)

    return predictions, ground_truths, fixation_infos


parser = argparse.ArgumentParser(description="Evaluate scanpath predictions")
parser.add_argument("--prediction_file", type=str, default="test_predictions_SeekUI.json", help="Path to the prediction JSON file")
args = parser.parse_args()

test_predictions = json.load(open(args.prediction_file))
test_prediction_form(test_predictions)


predictions, ground_truths, fixation_info = process_predictions(test_predictions)
assert len(predictions) == len(ground_truths) == len(fixation_info)

evaluator = Evaluator(".", 20)
cur_dataset = "VISGUI"

scores = evaluator.measure(ground_truths, predictions, image_size=[384, 512],
                           fixation_info=fixation_info)

prediction_fixation_dict = {}
gt_fixation_dict = {}
for iter in range(len(ground_truths)):
    key = "{}-{}-{}".format(cur_dataset, fixation_info[iter]["target_id"], fixation_info[iter]["image_name"])
    prediction_fixation_dict.setdefault(key, []).append(predictions[iter])
    gt_fixation_dict.setdefault(key, []).append(ground_truths[iter])

sal_scores = evaluator.eval_saliency(prediction_fixation_dict, gt_fixation_dict)

cur_metrics = process_scores(scores, sal_scores)

for (metric_name, metric_value) in cur_metrics.items():
    print("{metric_name:15}: {metric_value:.4f}".format
          (metric_name=metric_name, metric_value=metric_value))



