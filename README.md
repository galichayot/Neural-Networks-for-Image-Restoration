# Neural-Networks-for-Image-Restoration
l develop a general workflow for training networks to restore corrupted images, and then apply this workflow

The method that I implement consist of the following three steps:

1.Collect “clean” images, apply simulated random corruptions, and extract small patches.\

2.Train a neural network to map from corrupted patches to clean patches.

3. Given a corrupted image, use the trained network to restore the complete image by restoring each
patch separately, by applying the “ConvNet Trick” for approximating this process as learned in
class.
