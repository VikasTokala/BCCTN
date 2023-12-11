# Dual-input neural networks (DI-NNs)
This repository contains the code for the neural networks and training scripts
related to our NeurIPS2022 submission.

## Overview
* The `DCNN` repository contains the networks and training of our neural networks.
* The file `DCNN/utils/di_crnn.py` contains a base Torch model that is unrelated to our application of sound source localization. It is therefore the recommended entry point for those who want to use DI-NNs on their own domains.
* In turn, the file `DCNN/di_ssl_net.py` contains the network adapted for the task of sound source localization.
* The file `DCNN/trainer.py` contains a Pytorch lightning model for training.
* The `pysoundloc` directory contains the Least Squares baseline.

## Installation
The requirements of this project are listed in the file `requirements.txt`
Use the command `pip install -r requirements.txt` to install them.


## Testing the model
Under the directory `demo/`, you will find a Jupyter notebook as well as the model's pretrained weights and a small testing dataset.  

## Generating the datasets
Synthetic data was generated using a package created by the authors called SYDRA (SYnthetic Datasets for Room Acoustics).
This package is included here for convenience under the `sydra` directory. The configuration of each generated dataset is governed by [Hydra](www.hydra.cc).

Note that a Kaggle notebook and dataset will be provided after the paper review for training the models online without the need to generate the datasets.

### Synthetic datasets
To generate a synthetic dataset, one must change the configuration under `sydra/config/config.yaml` to generate the desired synthetic dataset.
Then, generate a dataset by running the command: `python main.py dataset_dir=path/to/dataset num_samples=X`.
after modifying

### Recorded datasets
To generate a dataset using the [LibriAdhoc40](https://github.com/ISmallFish/Libri-adhoc40) recorded dataset, you must first download it.
Then, change directory to `sydra/adhoc40_dataset` and run the command `python generate_dataset.py input_dir=/path/to/libri_adhoc40_dataset output_dir=/output/path mode='train|validation|test'`
to generate the training, validation or testing datasets. You can alternatively alter the configuration under `sydra/config/adhoc40_dataset.yaml`


### Generating Metadata for the Early Fusion network
The Early fusion network presented on our article uses signals generated using the Image Source method which would make training too costly and slow if generated
at training batch. They are therefore generated beforehand. They are generated using an existing dataset config, which must be provided as follows:
`python sydra/from_config.py config_file_path=/path/to/config.csv dataset_dir=/output/directory/path`.

## Training
Training, the datasets and model are also configured using Hydra. You can alter these configs at `DCNN/config`.
Once the datasets are available, you can train the models by running `python train.py`.

## Evaluating the Least Squares Sound Source Localization (LS-SSL) baseline
The code for the baseline is located under the `pysoundloc/` directory. To run the tests, run `python test_ls_ssl_baseline.py`. The choice of which dataset to evaluate the baseline on is govern by the same .yaml files above