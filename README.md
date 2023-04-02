# Straightforward Variational Autoencoder
The Variational Autoencoder (VAE) is an autoencoder which generates a code from which new samples can be drawn. 
In order to have a code from which useful samples can be drawn, these criteria must be met: 
* The values of each feature in the code must be normally distributed
* The features of the code must not be correlated

The reason for these conditions is shown [later](#why-regularize). 

The VAE achieves this by employing a complex probabilistic framework, which I personally find unintuitive. 
The idea of this work is to create an autoencoder that achieves the same goal as the VAE but using a more intuitive formulation. 
Specifically, instead of the probabilistic framework, I add regularization terms, which directly make sure that the autoencoder's code is well behaved. 

The loss function of my autoencoder is

    loss = reconstruction_loss + alpha*shape_regularization_loss + beta*correlation_regularization_loss

The **reconstruction loss** is the regular autoencoder loss which makes sure that the autoencoder learns to properly compress data samples and reconstruct the original data samples when decompressing them. 

The **shape regularization loss** measures how much each feature in the code deviates from the normal distribution. This is done by sorting the values of each feature within a batch and comparing how much the sorted values deviate from the CDF of the standard normal distribution. 

The **correlation regularization loss** measures how much features are correlated with each other. 

## Why regularize?

Why do we need the regularization losses? Wouldn't the reconstruction loss be sufficent? 

The following images are sampled from the code of different autoencoders, which were trained using different regularization losses. 

### No regularization at all
![no_reg1](https://user-images.githubusercontent.com/1943719/229309768-62a0e921-cc02-4391-be5e-14d20fd5c675.png)
![no_reg2](https://user-images.githubusercontent.com/1943719/229309773-2bee3b22-fb50-47d1-a447-007e1295ef97.png)
![no_reg3](https://user-images.githubusercontent.com/1943719/229309778-4d8ff5f2-f481-41c2-92c7-c118f24e01fe.png)

When not regularizing at all, the pictures are black. That's because the code is sampled assuming a standard normal distribution, with a standard deviation of 1. But in fact the code blew up during with values ranging from -1000 to +1000. 

### No shape regularization
![no_deviation1](https://user-images.githubusercontent.com/1943719/229309798-4089c5ba-1f54-4e98-9d29-4980966a1a00.png)
![no_deviation2](https://user-images.githubusercontent.com/1943719/229309803-15b4b6f3-0fb2-4ce7-8bf0-e0d614fbbdfa.png)
![no_deviation3](https://user-images.githubusercontent.com/1943719/229309808-c8bae1d9-6fe1-4356-b483-0fca11e7d4c7.png)

When not regularizing the shape of the distribution, the distribution of the autoencoder shifts from the standard normal distribution. Each feature follows a different distribution. When sampling from a standard normal distribution, we don't sample from the full distribution of the code and the results look bad. 

### No correlation regularization
![no_corr1](https://user-images.githubusercontent.com/1943719/229309816-2093c691-5f8c-4cde-878b-9a2bde5cafe3.png)
![no_corr2](https://user-images.githubusercontent.com/1943719/229309821-37021d9d-9414-4b4d-bf62-91c507f4fb15.png)
![no_corr3](https://user-images.githubusercontent.com/1943719/229309823-9efd4a2e-dcd0-4fec-a142-25f848c300e0.png)

When regularizing the shape of the distribution but not the correlation, results look better. But since we didn't regularize the correlation, it might be that we sample from the distribution in a way that the autonencoder isn't used to. For example, if *feature 5* and *feature 23* are closely correlated, but we sample *feature 5* as 1.15 and *feature 23* as -2.3, this would break the correlation which the autoencoder is used to and thus the images look bad. 

### Regularizing both the shape and the correlation
![both1](https://user-images.githubusercontent.com/1943719/229309832-4c94daa1-b19c-4aa1-be0c-372a38fd5d6d.png)
![both2](https://user-images.githubusercontent.com/1943719/229309840-6d5d8468-71fc-45e7-ba64-a56d35cac2bf.png)
![both3](https://user-images.githubusercontent.com/1943719/229309848-e77f534b-cd18-4b86-8174-8aa0235ff639.png)

When adding both regularization terms, one can actually sample from the code and meaningful images come out. These images are certainly not amazing but I think it shows that the proposed regularization terms somewhat work. 
