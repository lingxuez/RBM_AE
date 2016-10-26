---
title: "RBM, Denoising Autoencoder and pretraining"
author: Lingxue Zhu
date: 10/26/2016
---

# RBM, Autoencoder, Denoising Autoencoder and pretraining

This is for CMU 10-807 "Topics in Deep Learning", Assignment 2.

## Data

The data are stored under directory `data/`.


## RBM

To train an RBM with a given dimension of hidden layer `dim_hidden` and
number of steps for contrastive divergence `k_CD`, run 

> python trainRBM.py <dim_hidden> <k_CD>

The output will be written to `RBMresults/`, including the learned weights and bias terms. As an example, the current directory contains the output from the following command

> python trainRBM.py 100 1


## (Denoising) Autoencoder

To train a (denoising) autoencoder with a given dimension of hidden layer `dim_hidden` and probability of adding noise `noise_probability`, run

> python trainAutoencoder.py <dim_hidden> <noise_probability>

When `noise_probability` is set to 0, no noise is introduced and a regular autoencoder is learned. The output will be written to `Autoresults/`, including the learned weights and bias terms. As an example, the current directory contains the output from the following command

> python trainAutoencoder.py 100 0.25


## Pretraining Neural Network

To learn an one-layer neural network with unsupervised pretraining from RBM and (denoising) autoencoder, run

> python trainNN.py <dim_hidden> <method>

where `method` is one of 

1. `RBM`, the default is to use the results with `k_CD=1`. 
2. `Auto`, using the results from autoencoder.
3. `DenoAuto`, using the results from denoising autoencoder, the default is to use `noise_probability=0.25`.
4. `Random`, randomly initialize the weights without pretraining.

Note that the pre-trained weights must be under directory `RBMresults/` or `Autoresults/` when using the corresponding method(s). The current directory contains the pre-trained weights for 100 dimension of hidden layer, and is ready to run the following commands:

> python trainNN.py 100 RBM
> python trainNN.py 100 Auto
> python trainNN.py 100 DenoAuto
> python trainNN.py 100 Random

See the directory `NNout/` for sample output.





