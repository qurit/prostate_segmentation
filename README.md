# Prostate Lesion Segmentation

## Usage
We use Sacred to help manage experiments and for command line interface. Sacred documentation can be found
here https://sacred.readthedocs.io/en/stable/quickstart.html. Below are the core features we use from Sacred.
- Specify the name for each experiment via command line with `--name=sample_name` or `-n sample_name`.
An output directory containing all experiment files will be created with the experiment name.
- Comments associated with an experiment can be added in the command line with `-c` followed by
comment in single quotes.
- Changes to parameters in the config can be changed on the fly via command line keyword `with` followed
by the key value pair parameters, e.g. `with 'a=2.3' 'b="FooBar"' 'c=True'`. Note changing the parameter `CONFIG_FILE`
which loads an existing configuration file needs to be done inside the `config()` function, i.e.
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

### Evaluation
In evaluation mode, use the same name as the training run and set the parameter `EVAL_ONLY` to true.
This will create a new directory prefixed **eval** inside the training run and will use by default
the file `model_best.pth` as the weight file.
```shell
python -m seg_3d.train_loop --name=test1 with 'EVAL_ONLY=True'
```

### Inference
Similar to above.
```shell
python -m seg_3d.train_loop --name=test1 with 'PRED_ONLY=True'
```

## Sacred setup
1. Install docker https://docs.docker.com/get-docker/
2. Bring up omniboard and mongo database run `docker compose up` (or `docker-compose up`).

### Useful notes
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
