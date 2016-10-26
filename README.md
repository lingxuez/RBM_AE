
# RBM, Autoencoder, Denoising Autoencoder and pretraining

This is for Assignment 2 of CMU 10-807 "Topics in Deep Learning" from Lingxue Zhu.

## Data

The MNIST dataset is stored under directory `data/`, downloaded from the [course website](http://www.cs.cmu.edu/~rsalakhu/10807_2016/assignments.html).


## RBM

To train an RBM with a given dimension of hidden layer `dim_hidden` and
number of steps for contrastive divergence `k_CD`, run 

> python trainRBM.py \<dim_hidden> \<k_CD>

The output will be written to `RBMresults/`, including the learned weights and bias terms. As an example, the current directory contains the output from the following command

> python trainRBM.py 100 1

To generate new images using a Gibbs chain starting from random configuration of the visible variables, run

> python sampleRBM.py

The default setting generates 100 new images using the Gibbs sampler with 1000 steps.


## (Denoising) Autoencoder

To train a (denoising) autoencoder with a given dimension of hidden layer `dim_hidden` and probability of adding noise `noise_probability`, run

> python trainAutoencoder.py \<dim_hidden> \<noise_probability>

When `noise_probability` is set to 0, no noise is introduced and a regular autoencoder is learned. The output will be written to `Autoresults/`, including the learned weights and bias terms. As an example, the current directory contains the output from the following command

> python trainAutoencoder.py 100 0.25


## Pretraining Neural Network

To learn an one-layer neural network to classify the 10 handwritten digits with unsupervised pretraining, run

> python trainNN.py \<dim_hidden> \<method>

where `method` defines the pretraining method, being one of 

1. `RBM`, using the results from RBM, the default is with `k_CD=1`. 
2. `Auto`, using the results from autoencoder without adding noise.
3. `DenoAuto`, using the results from denoising autoencoder, the default is to use `noise_probability=0.25`.
4. `Random`, randomly initialize the weights without pretraining.

Note that the pre-trained weights must be under directory `RBMresults/` or `Autoresults/` when using the corresponding method(s). The current directory contains the pre-trained weights for 100 dimension of hidden layer, and is ready to run the following commands:

> python trainNN.py 100 RBM

> python trainNN.py 100 Auto

> python trainNN.py 100 DenoAuto

> python trainNN.py 100 Random

See the directory `NNout/` for sample output.





