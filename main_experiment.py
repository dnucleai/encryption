import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_generation import control_training, control_validation, test_training, test_validation
from simple_nn import Net

control_neural_network = Net(1,1,1)
test_neural_network = Net(2,2,2)

 DataLoader(bAbiDataset(*d_train), batch_size=batch_size,
						shuffle=False, num_workers=4, pin_memory=gpu_available)

control_training = DataLoader(control_training, batch_size=4, shuffle=false)
control_validation = DataLoader(control_validation, batch_size=4, shuffle=false)

test_training = DataLoader(test_training, batch_size=4, shuffle=false)
test_validation = DataLoader(test_validation, batch_size=4, shuffle=false)

epochs = 5

for epoch in range(epochs):
	for idx_control, sample_control in enumerate(control_training):
		control_input, control_label = Variable(sample_control["training_data_point"]), 
										Variable(sample_control["label"])
		control_prediction = control_neural_network(control_input)
		
