# MLlib

Set up experiments on deep learning models

## Python Version
python 3.11.4

## Installation
```shell
pip install torch
pip install torchsampler
```

## Data preparation
Data should be processed first and put aside in `dataset` directory. The raw data and processed data are made available on request only.


## Usage
We provide MLP, AlexNet, Vgg16, Resnet18, Transformer, and Swin-transformer models for evaluation, you can conduct experiments and select 
models by `--config-file` argument shown as below. The corresponding config files are listed in `config` directory.

### Quick start
```shell
# run with the demo data in git repo 
python run.py -c config/alexnet_demo.yaml
```

### Training
```shell
# cd path/to/MLlib
python run.py --config-file config/vgg16_1d.yaml 
```

### Resume training

```shell
python run.py --config-file config/vgg16_1d.yaml TRAIN.RESUME /path/to/weight
```

### Test

```shell
python run.py --config-file config/vgg16_1d.yaml --test-only TEST.WEIGHT /path/to/weight
```

