# Prostate Lesion Segmentation

## Usage
Specify the name for each experiment via command line with `--name=sample_name` or `-n sample_name`.
An output directory containing all experiment files will be created with the experiment name.

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
