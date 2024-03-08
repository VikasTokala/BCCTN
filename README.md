# Binaural Speech Enhancement using Complex Transformer Networks (BCCTN)
This repository contains the code for the neural network models, training and evaluation scripts for BCCTN.


# Introduction

The software was developed in the frame of WP1 for binaural speech
enhancement using deep complex-valued convolutional networks
[@Tokala2024] in acoustic scenarios with a single target speaker and
isotopic noise of various types.

Github Repository: <https://github.com/VikasTokala/BCCTN>

# Software Structure 

The code in the repository can be used to train and evaluate the network
for binaural speech enhancement. The software is written using Python
language and uses Pytorch for training the network.

-   **requirements.txt:** This file contains the required Python
    packages that need to be installed. The packages can be installed
    using `pip` or `conda` package installers.

-   **config:** This folder contains the following `yaml` files that
    control the training parameters. These can be modified to change the
    configuration of the network.

    -   **model.yaml:** This file has the parameters that alter the
        model parameters.

    -   **training.yaml:** This file contains the parameters that
        control the training of the network such as batch size, number
        of epochs, learning rate, etc.

    -   **dataset/speech_dataset.yaml:** This file contains the paths to
        the noisy and clean train, test, and validation binaural
        signals.

    -   **config.yaml :** This is the master file that has all the
        configuration parameters for the model, training, and dataset
        `yaml` files. Modifying the parameters in this file will modify
        the individual `yaml` files.

The **DCNN** folder contains the data loader, model, feature extraction,
and loss function files.

-   **base_dataset.py:** The dataloader script prepares the given
    dataset for training based on the configuration parameters.

-   **test_dataset.py:** The dataloader script prepares the test dataset
    for evaluation using the trained network.

-   **model.py:** The script defines the encoder, decoder, and
    transformer modules of the network.

-   **binaural_attention_model.py:** The script combines the encoder,
    decoder, and transformer modules defined in the model.py files and
    places the necessary skip connections.

-   **utils:** This directory contains support files and functions used
    to apply masking, read the training parameters from the config
    files, and complex-valued Pytorch functions necessary for training
    the network.

-   **feature_extractors.py:** This script contains the
    [STFT]{acronym-label="STFT" acronym-form="singular+short"} and
    [ISTFT]{acronym-label="ISTFT" acronym-form="singular+short"}
    functions which perform feature extraction for the training of the
    network.

-   **loss.py:** This script defines the loss function used in training
    the network. It also contains the functions necessary to compute the
    individual terms of the loss function.

-   **trainer.py:** This script uses Pytorch Lighting module to manage
    the training. It controls the learning rate, and optimization and
    creates logs for training.

-   **train.py:** This is the master script that integrates the
    dataloader and trainer files and runs the training operation.

-   **Checkpoints:** This directory contains the trained network
    checkpoint that can be used for binaural speech enhancement.

-   **analysedata.ipynb:** This is a Jupyter notebook used to test the
    trained network. The path to the test data and the trained model
    checkpoint are supplied to the script. The trained model is used to
    enhance the noisy speech. Spectrogram plots are generated and audio
    files are generated to listen and compare the noisy and enhanced
    speech signals.
