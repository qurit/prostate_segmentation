# Experiments

## Sacred
We use [Sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) to help manage experiments and for command line interface. Below are the core features we use from Sacred:

- [Observer interface](https://sacred.readthedocs.io/en/stable/experiment.html#observe-an-experiment) to collect all experiment related information and store it in a database.
    - files added with `ex.add_artifact`
    - metrics are logged with `ex.log_scalar`
    - config added with `ex.add_config`
- [Command line interface](https://sacred.readthedocs.io/en/stable/command_line.html) to simplify experiment initialization and configuration
    - each experiment requires the following flag to specify the experiment name `--name=experiment_name` or `-n experiment_name`. An output directory containing all experiment files will be created with the experiment name.
    - To ignore oberservers use flag `-u` which is useful for some quick tests or debugging runs
    - Changes to parameters in the config can be changed on the fly via command line keyword `with` followed by the key value pair parameters, e.g. `with "DATASET.FOLD=$COUNT" "EVAL_ONLY=True"`. Note, changing the parameter `CONFIG_FILE` which loads an existing configuration file via command line, needs to have the additional parameter `"LOAD_ONLY_CFG_FILE=True"`, i.e.:
        ```bash
        python -m seg_3d.train_loop -n experiment_name with "CONFIG_FILE=./config.yaml" "LOAD_ONLY_CFG_FILE=True"
        ```

## Omniboard
We use [Omniboard](https://vivekratnavel.github.io/omniboard/#/README) as the tool for visualizing experiments logged with Sacred. From Omniboard documentation:
> Omniboard is a web dashboard for the Sacred machine learning experiment management tool. It connects to the MongoDB database used by Sacred and helps in visualizing the experiments and metrics / logs collected in each experiment.

Follow the installation steps described in the [quick start guide](https://vivekratnavel.github.io/omniboard/#/quick-start). The easiest way of setting up both MongoDB and Omniboard, use docker:
1. Install docker https://docs.docker.com/get-docker/
2. Configure **docker-compose.yml**
2. Run from the repo root directory `sudo docker compose up`
3. Open http://localhost:9000/ in the browser. Port number is specified in **docker-compose.yml**
    ```yaml
    omniboard:
        ports:
        - 127.0.0.1:9000:9000
    ```

### Saving and loading existing data into MongoDB
A docker [volume](https://docs.docker.com/storage/volumes/) is used to persist data from the MongoDB. Volumes are stored on disk at **docker/volumes**.

To load existing data, first choose the right volume in **docker/volumes** and then add the following parameter in the **docker-compose.yml**, replacing `/path_to_volume/` with the correct volume path:
```yaml
  mongo:
    volumes:  # by default volumes are mapped on disk to docker/volumes
      - /path_to_volume/:/data/db
```

## Tensorboard

We also use [Tensorboard](https://www.tensorflow.org/tensorboard) to visualize the inputs and outputs of the model during train time. To bring up the Tensorboard dashboard, run the following command, where `output-dir`` is the path to the directory storing the training runs
```bash
tensorboard --logdir ouput-dir
```

The following code snippet inside **seg_3d/train_loop.py** logs the image data to Tensorboard:
```python
# check if need to process masks and images to be visualized in tensorboard
for idx, p in enumerate(patients):
    # hardcoded, only visualize 4 patients from train set
    if p in train_dataset.patient_keys[:4]:
        for name, batch in zip(["img_orig", "img_aug", "mask_gt", "mask_pred"],[orig_imgs, sample, labels, preds]):
            tags_imgs = tensorboard_img_formatter(name=p + "/" + name, batch=batch[idx].unsqueeze(0).detach().cpu())

            # add each tag image tuple to tensorboard
            for item in tags_imgs:
                storage.put_image(*item)
```

## Experiment outputs
Each experiment has an associated output directory containing various files generated during training and evaluation.
```
output/experiment_name/
│   │
│   ├─ 0/  # fold number for KFold cross validation
│       ├─ config.yaml       # the full config for this experiment
│       ├─ model_best.pth    # weights of best model
│       ├─ model_0000199.pth # weights of model saved during a checkpoint at iteration 199
│       ├─ inference.pk      # predictions from best model (along with inputs and labels)
│       ├─ best_metrics.txt  # results from the best model
│       ├─ metrics.json      # logged metrics during each training/evaluation step
│       ├─ log.txt           # all log messages generated during experiment
│       ├─ last_checkpoint   # used for resuming training if experiment ends abruptly
│       ├─ events.out.tfevents..  # Tensorboard file
│       ├─ masks/  # figures for the prediction masks
│               ├─ cor/  # coronal plane
│               │      ├─ PSMA-01-126/  # directory for each patient
│               │      │      ├─ 56.png  # each figure shows a slice with index based on filename, e.g. 56
│               │      │      ├─ ...
│               │      ├─ ...
│               ├─ sag/  # saggital plane
│               │      ├─ ... # same as above
│               └─ tra/  # transverse/axial plane
│                      ├─ ... # same as above
│       └─ eval_0/   # directory with results from doing EVAL_ONLY mode
│               ├─ config.yaml
│               ├─ inference.pk
│               ├─ log.txt
│               └─ masks/
│
...  # additional directories for the other folds
```
