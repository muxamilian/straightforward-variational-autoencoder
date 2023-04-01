# Straightforward Variational Autoencoder
The Variational Autoencoder (VAE) is an autoencoder which generates a code from which samples can be drawn. 
In order to have a code from which samples can be drawn, these criteria must be met: 
* The values of each feature in the code must normally distributed
* The features of the code must not be correlated

The reason for these conditions is shown late (TODO insert here)

The VAE achieves this by employing a complex probabilistic framework, which I personally find unintuitive. 
The idea of this work is to create an autoencoder that achieves the same goal as the VAE but using a more intuitive formulation. 
Specifically, instead of the probabilistic framework, I add regularization terms, which directly make sure that the autoencoder's code is well behaved. 

The loss function of my autoencoder is

    loss = reconstruction_loss + alpha*deviation_regularization_loss + beta*correlation_regularization_loss

The **reconstruction loss** is the regular autoencoder loss which makes sure that the autoencoder learns to properly compress data samples and reconstruct the original data samples when decompressing them. 

The **deviation regularization loss** measures how much each feature in the code deviates from the normal distribution. This is done by sorting the values of each feature within a batch and comparing how much the sorted values deviate from the CDF of the standard normal distribution. 

The **correlation regularization loss** measures how much features are correlated with each other. 

## Why regularize?

Why do we need the regularization losses? Wouldn't the reconstruction loss be sufficent? 

### No regularization at all

### No deviation regularization

### No correlation regularization

### Both

