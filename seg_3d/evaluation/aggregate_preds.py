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
            inference_dict = {**inference_dict, **pickle.load(f, encoding="bytes")}
            # os.remove(os.path.join(run_dir, 'inference.pk'))

        if metric_list is None:
            metric_list = MetricList(metrics=get_metrics(metrics), class_labels=class_labels)

        for idx, patient in enumerate(inference_dict):
            gt = inference_dict[patient]['gt']
            preds = inference_dict[patient]['preds']

            if thresholds:
                new_preds = [
                    np.where(pred >= thres, np.ones_like(pred), np.zeros_like(pred))
                    for thres, pred in zip(thresholds, preds)
                ]
                preds[:] = np.stack(new_preds)

            metric_list(torch.from_numpy(preds), torch.from_numpy(gt))
            logger.info('results for patient {}:'.format(patient))
            patient_metrics = metric_list.get_results_idx(idx)
            for key in patient_metrics:
                logger.info("{}: {}".format(key, patient_metrics[key]))
            patient_metrics['patient'] = patient

            # convert lists to separate key, value pair
            for k, v in patient_metrics:
                if len(list(v)) > 1:
                    for jdx, item in enumerate(v):
                        patient_metrics[key + '/{}'.format(class_labels[jdx])] = item

            results.append(patient_metrics)
            print(patient_metrics)

    averaged_results = metric_list.get_results(average=True)
    logger.info('Inference done! Mean metric scores:')
    logger.info(json.dumps(averaged_results, indent=4))

    for k, v in averaged_results.items():
        ex.log_scalar(k, float(v))

    # save inference results
    with open(os.path.join(root_dir, 'inference.pk'), 'wb') as f:
        pickle.dump(inference_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save metrics to a .txt file
    with open(os.path.join(root_dir, 'averaged_metrics.txt'), 'w') as f:  # TODO save stats instead like min, max, mean, median, std
        json.dump(averaged_results, f, indent=4)
    
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(os.path.join(root_dir, 'metrics.csv'), index=False)


@ex.config
def config():
    metrics = [
        'classwise_dice_score',
        'argmax_dice_score'
    ]
    thresholds = None
    tags = ['kfold_eval']
    load_inference_fp = 'eval_0/inference.pk'


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