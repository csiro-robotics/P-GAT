# P-GAT : Pose-Graph Attentional Network

This repository contains the code implementation for the paper "P-GAT : Pose-Graph Attentional Network for Lidar Place Recognition".

## Requirements
The code was tested using Python 3.9.4 with PyTorch 1.11.0. To set up the complete environment, we recommend to use following commands:
```
conda create -n `<myenv>` python=3.9.4
conda activate `<myenv>`
```
Replace the `<myenv>` by any name you want.
```
pip install -r requirements.txt
conda activate <myenv>
```

## Datasets
For the training and evaluation, P-GAT uses three datasets: 
- Oxford RobotCar dataset
- In-house dataset
- MulRan dataset

Following MinkLoc3D paper or github repository ([link](https://github.com/jac99/MinkLoc3D)) to generate the descriptors for point clouds and save them in pickles.

We use `datasets/dataset_generator_oxford.py` to generate the oxford and in-house dataset for training and testing, `datasets/dataset_generator_mulran.py` to generate DCC and riverside sub-datasets in MulRan dataset, and `datasets/dataset_generator_kaist.py` to generate KAIST sub-dataset in MulRan. 

The graph and sub-graph generation are done inside the scripts.
The usage of the scripts is:
```
usage: dataset_generator.py [-h] [--config_file FILE] ...

Generate dataset in graph structure

positional arguments:
  opts                Modify config options using the command-line

optional arguments:
  -h, --help          show this help message and exit
  --config_file FILE  path to config file
```
The arguments required by the script is maintained by the YAML configuration files. The example of the config file is `configs/data_generator_config.yml`, and an example of usage command is
```
python dataset_generator_oxford.py --config <path-to-config>/data_generator_config.yml
```

## Model training
We use `training/train.py` to train the model. 

The default configuration is in the `attentional_graph/config/defaults.py`.
The customized configuration is applied via configuration files in format of YAML. 
The example of config file in YAML can be found in `configs/training_config.yml`.

The example of training command is:
```
python training/train.py --config <path-to-config>/training_config.yml
```

## Model evaluation
We use `training/test.py` to evaluate the model.

Same as the training, the default configuration is in the `attentional_graph/config/defaults.py`,
and the customized configuration is applied via configuration files in format of YAML. 
The example of config file in YAML can be found in `configs/test_config.yml`.

The example of testing command is:
```
cp <pre-trained-model.pt> .
python testing/test.py --config <path-to-config>/test_config.yml
```
