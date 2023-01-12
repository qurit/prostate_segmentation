import json
import os
import pickle
import yaml

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver
import torch

import seg_3d
from seg_3d.evaluation.metrics import MetricList, get_metrics
from seg_3d.utils.logger import setup_logger, add_fh

ex = Experiment()


@ex.main
def main(metrics, thresholds, load_inference_fp, _run):
    name = _run.experiment_info['name']
    root_dir = os.path.join('seg_3d/output', name)
    add_fh(logger, output=root_dir)
    logger.info("Starting evaluation on KFold run...")

    inference_dict = {}
    results = []
    metric_list = None

    for d in os.listdir(root_dir):
        run_dir = os.path.join(root_dir, d)
        if not os.path.isdir(run_dir):
            continue

        fold_config = yaml.safe_load(open(os.path.join(run_dir, 'config.yaml'), 'r'))
        class_labels = fold_config['DATASET']['CLASS_LABELS']

        with open(os.path.join(run_dir, load_inference_fp), "rb") as f:
            curr_inference = pickle.load(f, encoding="bytes")
            # os.remove(os.path.join(run_dir, 'inference.pk'))

        if metric_list is None:
            metric_list = MetricList(metrics=get_metrics(metrics), class_labels=class_labels)

        for idx, patient in enumerate(curr_inference):
            gt = torch.from_numpy(curr_inference[patient]['gt']).float()  # .float() fixes a weird bug which may come from AMP
            preds = torch.from_numpy(curr_inference[patient]['preds']).float()

            if thresholds:
                new_preds = []
                for thres, pred in zip(thresholds, preds):
                    if thres is None:
                        new_preds.append(pred)
                        continue
                    new_preds.append(
                        torch.where(pred >= thres, torch.ones_like(pred), torch.zeros_like(pred))
                    )

                preds[:] = torch.stack(new_preds)

            metric_list(preds, gt)  # these should be the same numbers as before if no thresholding
            logger.info('results for patient {}:'.format(patient))
            patient_metrics = metric_list.get_results_idx(idx)
            for key in patient_metrics:
                logger.info("{}: {}".format(key, patient_metrics[key]))
            patient_metrics['patient'] = patient

            # convert lists to separate key, value pair
            new_entries = {}
            for key, val in patient_metrics.items():
                if len(list(val)) > 1 and type(val) is not str:
                    patient_metrics[key] = list(val)
                    for jdx, item in enumerate(val):
                        new_entries[key + '/{}'.format(class_labels[jdx])] = item

            patient_metrics = {**patient_metrics, **new_entries}
            results.append(patient_metrics)
            inference_dict = {**inference_dict, **curr_inference}

    df = pd.DataFrame(results)
    df.index = df['patient']

    # compute statistics
    stats = {}
    for k, v in df.select_dtypes(include=['float64']).items():
        stats = {**stats,
                 k + '_max'           : v.max(),
                 k + '_max_patient'   : v.idxmax(),
                 k + '_min'           : v.min(),
                 k + '_min_patient'   : v.idxmin(),
                 k + '_median'        : v.median(),
                 k + '_mean'          : v.mean(),
                 k + '_std'           : v.std()}

    logger.info('Done aggregating metrics! Scores:')
    logger.info(json.dumps(stats, indent=4))

    # save inference results
    # with open(os.path.join(root_dir, 'inference.pk'), 'wb') as f:
    #     pickle.dump(inference_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save to a .txt file
    with open(os.path.join(root_dir, 'kfold_stats.txt'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    # save df
    df.to_csv(os.path.join(root_dir, 'kfold_all_scores.csv'))

    # log to sacred experiment
    for k, v in stats.items():
        if '_mean' in k:
            k = k[:-5]  # this is just to keep metric names consistent with metrics in training
        ex.log_scalar(k, v)


@ex.config
def config():
    metrics = [
        'classwise_dice_score',
        'argmax_dice_score'
    ]
    thresholds = None
    tags = ['kfold_eval']
    load_inference_fp = ''


if __name__ == '__main__':
    logger = setup_logger(name=seg_3d.evaluation.__name__)

    # mongo observer
    ex.observers.append(
        MongoObserver(url=f'mongodb://'
                          'sample:password'
                          f'@localhost:27017/?authMechanism=SCRAM-SHA-1', db_name='db')
    )  # assumes mongo db is running
    ex.logger = logger
    ex.run_commandline()