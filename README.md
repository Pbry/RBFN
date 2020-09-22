# RBFN
Simple example of Radial Basis Function neural network implemented from scratch.
It has one hidden layer, the centers are calculated by k_means.
The activation function is a negative exponential of the distance between input and desired output.
The estimator of the cost's gradient is computed manually.

Originally it was meant for robot Poppy's learning (https://www.poppy-project.org/fr/).
One action of Poppy in a 3D space is determined by 13 parameters. Thus, in the "main", I train and test
the network for a regression problem on an arbitrary function 13D->3D.


Paloma Bry
06/2016
