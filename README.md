# Prostate Lesion Segmentation

## Usage
Specify the name for each experiment via command line with `--name=sample_name` or `-n sample_name`.
An output directory containing all experiment files will be created with the experiment name.
Comments for a run can be added via the `-c` flag followed by text in quotations e.g. `-c 'this is a comment`.

### Train
```shell
python -m seg_3d.train_loop --name=train_test
```

### Evaluation
```shell
python -m seg_3d.train_loop --name=eval_test with 'EVAL_ONLY=True'
```

### Inference
```shell
python -m seg_3d.train_loop --name=eval_test with 'PRED_ONLY=True'
```

## Docker setup
To bring up omniboard and mongo database run `docker-compose up`.
