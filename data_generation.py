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
multiplier = 3.14159265 * 15
prng = RandomState(3123)
data = prng.rand(5000,1) * multiplier
annotations = np.sin(data)

training_set = data[:4500,:]
training_labels = annotations[:4500,:]
validation_set = data[4500:,:]
validation_labels = annotations[4500:,:]

training_set_tensor = torch.Tensor(training_set)
training_labels_tensor = torch.Tensor(training_labels)
validation_set_tensor = torch.Tensor(validation_set)
validation_labels_tensor = torch.Tensor(validation_labels)

control_training = custom_loader(training_set_tensor, training_labels_tensor, 4500)
control_validation = custom_loader(validation_set_tensor, validation_labels_tensor, 500)

# encrypted data dimensions (n_samples, n_features + n) where n = 1 in experiment
# so we have 250 samples x 2 features for both training and annotations


encrypted_data = []
encrypted_annotations = []
t = 0.01+1j
for i in range(5000):
	point = data[i]
	# encrypted_data.append([multiplier * cmath.exp(t*point).real, \
	# 							multiplier * cmath.exp(t*point).imag])
	encrypted_data.append([point[0] + 1.4])
	point2 = annotations[i]
	# encrypted_annotations.append([cmath.exp(t*point2).real,cmath.exp(t*point2).imag])
	encrypted_annotations.append([point2[0] + 1.4])

encrypted_data = np.stack(encrypted_data)
encrypted_annotations = np.stack(encrypted_annotations)

test_training_set = torch.Tensor(encrypted_data[:4500,:])
test_training_labels = torch.Tensor(encrypted_annotations[:4500,:])
test_validation_set = torch.Tensor(encrypted_data[4500:,:])
test_validation_labels = torch.Tensor(encrypted_annotations[4500:,:])

test_training = custom_loader(test_training_set, test_training_labels, 4500)
test_validation = custom_loader(test_validation_set, test_validation_labels, 500)





