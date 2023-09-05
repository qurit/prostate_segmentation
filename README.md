# Prostate Lesion Segmentation

## Overview
This repo trains and evaluates PyTorch models on DICOM data of PET/CT scans for bladder, prostate, and tumor ROI 
detection. Create an environment for the repo using pip or conda. For the required dataset structure, 
see [dataset documentation](docs/dataset.md). For config file setup, see [experiment documentation](docs/experiments.md).
<p align="center">
<img width="512" height="512" src=docs/figures/pred_mask1.gif alt="Sample Results"/>
</p>

## Repo Structure
```
prostate-segmentation/
├── conda_env.yml                   # Conda experiment file
├── dicom_code                      # DICOM-specific code
│   ├── contour_utils.py
│   ├── custom_dicontour.py
│   ├── dataset_refactor.py
│   └── __init__.py
├── docker-compose.yml              # Sacred Docker compose file
├── docs                            # Documentation
│   ├── dataset.md
│   ├── experiments.md
│   └── figures
├── __init__.py
├── notebooks                       # Inference Notebook and required code
│   ├── run_saved_model.ipynb
│   └── unet_code
├── README.md
├── requirements.txt                # Python package requirements
├── scripts                         # Scripts for automated runs
│   ├── run_configs.sh
│   └── run_kfold.sh
└── seg_3d                          # Core pipeline code
    ├── config                      # Default configs and config related code
    ├── data                        # Dataset and data related code
    ├── evaluation                  # Metrics and evaluation/visualization related code
    ├── __init__.py
    ├── losses.py                   # Loss function definitions
    ├── modeling                    # Neural Network architecture related code
    ├── train_loop.py               # Main file for running pipeline
    └── utils                       # Early stopping, logging, scheduling, and other utils
```

## Usage
- Specify the name for each experiment via command line with `--name=sample_name` or `-n sample_name`.
An output directory containing all experiment files will be created with the experiment name.
- Comments associated with an experiment can be added in the command line with `-c` followed by
comment in single quotes.
- Changes to parameters in the config can be changed on the fly via command line keyword `with` followed
by the key value pair parameters, e.g. `with 'a=2.3' 'b="FooBar"' 'c=True'`. Note changing the parameter `CONFIG_FILE`
which loads an existing configuration file needs to be done inside the `config()` function in `train_loop.py`, i.e.
    ```python
    @ex.config
    def config():
        cfg.CONFIG_FILE = 'seg_3d/config/bladder-detection.yaml'
        cfg.merge_from_file(cfg.CONFIG_FILE)  # config file has to be loaded here!
    ```

### Train
```shell
python -m seg_3d.train_loop --name=test1
```

To resume training from a previously started run, run the following command keeping the same experiment name.
```shell
python -m seg_3d.train_loop --name=test1 with 'RESUME=True'
```

### Evaluation
In evaluation mode, use the same name as the training run and set the parameter `EVAL_ONLY` to true.
This will create a new directory prefixed **eval** inside the training run and will use by default
the file `model_best.pth` as the weight file.
```shell
python -m seg_3d.train_loop --name=test1 with 'EVAL_ONLY=True'
```

To generate plots of the mask predictions along with the samples and ground truth labels, set the following
parameter in the config to true
```yaml
TEST:
    VIS_PREDS: true
```
Another option is to run the mask visualizer in standalone using the script `visualize_preds.py`.
Here you need to specify the path to the output directory and the class labels.
```shell
python -m seg_3d.evaluation.visualize_preds
```

We also use tensorboard to visualize the inputs and outputs of the model during train time. To bring up tensorboard
dashboard run the following command, where **output-dir** is the path to the directory storing the training runs
```shell
tensorboard --logdir ouput-dir
```

## Sacred
We use Sacred to help manage experiments and for command line interface. Sacred documentation can be found
here https://sacred.readthedocs.io/en/stable/quickstart.html. Below are the core features we use from Sacred.

### Sacred Setup
1. Install docker https://docs.docker.com/get-docker/
2. Bring up omniboard and mongo database run `sudo docker compose up` (or `docker-compose up`) from the repo root directory.
3. Open http://localhost:9000/ in the browser. Port number is specified in the docker-compose.yml file
    ```yaml
        ports:
          - 127.0.0.1:9000:9000
    ```

## Useful notes
- To create a new image from a container's changes and then push to registry (note this step does not work
to do a backup of mongo db).
    ```shell
    docker commit <container-id> myname/containername:version
    docker push <image-id>
    ```
- The mongo docker image writes data into a [volume](https://docs.docker.com/storage/volumes/)
- One way to do a backup of mongo db is via [mongodump](https://www.mongodb.com/docs/database-tools/mongodump/)
and then copying the file over from the container
- Omniboard docs https://github.com/vivekratnavel/omniboard/blob/master/docs/quick-start.md 