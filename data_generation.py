import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable

import cmath
import numpy as np
from numpy.random import RandomState

from dataloader import custom_loader

# generate some test data and serialize it and save it
# dimensions n_samples x n_features - 250 x 1
# targets 250 annotations
prng = RandomState(3123)
data = prng.randn(250,1)
annotations = np.sin(prng.randn(250,1))

training_set = data[:200,:]
training_labels = annotations[:200,:]
validation_set = data[200:,:]
validation_labels = annotations[200:,:]

training_set_tensor = torch.Tensor(training_set)
training_labels_tensor = torch.Tensor(training_labels)
validation_set_tensor = torch.Tensor(validation_set)
validation_labels_tensor = torch.Tensor(validation_labels)

control_training = custom_loader(training_set_tensor, training_labels_tensor, 200)
control_validation = custom_loader(validation_set_tensor, validation_labels_tensor, 50)

# encrypted data dimensions (n_samples, n_features + n) where n = 1 in experiment
# so we have 250 samples x 2 features for both training and annotations

for (elem in data[0, :])   
 {      t = 0.01+1j        
 		x = (cmath.exp(t*elem)).real       
 		 y = (cmath.exp(t*elem).imag     }

for 






