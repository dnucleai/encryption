import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_generation import control_training, control_validation, test_training, test_validation
from simple_nn import Net

control_neural_network = Net(1,1,50)
test_neural_network = Net(2,2,50)

control_training = DataLoader(control_training, batch_size=4, shuffle=False)
control_validation = DataLoader(control_validation, batch_size=4, shuffle=False)

test_training = DataLoader(test_training, batch_size=4, shuffle=False)
test_validation = DataLoader(test_validation, batch_size=4, shuffle=False)

epochs = 25

optimizer_control = torch.optim.Adam(control_neural_network.parameters(), lr=.0001)
optimizer_test = torch.optim.Adam(test_neural_network.parameters(), lr=0.0001)
criterion = nn.MSELoss()

control_loss_list = []

# begin control experiment

for epoch in range(epochs):
	for idx_control, sample_control in enumerate(control_training):
		control_input, control_label = Variable(sample_control["training_data_point"]), \
										Variable(sample_control["label"])
		control_prediction = control_neural_network.forward(control_input)
		control_loss = criterion(control_prediction, control_label)

		if idx_control % 2000 == 0:
			print "epoch: ", epoch, "idx_control: ", idx_control, "loss: ", control_loss

		control_loss.backward()
		optimizer_control.step()
		optimizer_control.zero_grad()


for idx_control, sample_control in enumerate(control_validation):
	control_input, control_label = Variable(sample_control["training_data_point"]), \
									Variable(sample_control["label"])
	control_prediction = control_neural_network.forward(control_input)
	control_validation_loss = criterion(control_prediction, control_label)
	control_loss_list.append(control_validation_loss)

total_control_loss = sum(control_loss_list)

print "total control loss: ", total_control_loss

# begin experiment
for epoch in range(epochs):
	for idx_sample, sample_exp in enumerate(test_training):
		test_input, test_label = Variable(sample_control["training_data_point"]), \
										Variable(sample_control["label"])
		test_prediction = test_neural_network.forward(control_input)
		test_loss = criterion(test_prediction, test_label)
		if idx_sample % 2000 == 0:
			print "epoch: ", epoch, "idx_sample: ", idx_sample, "loss: ", test_loss

		test_loss.backward()
		optimizer_test.step()
		optimizer_test.zero_grad()

test_loss_list = []

for idx_sample, sample_exp in enumerate(test_validation):
		test_input, test_label = Variable(sample_control["training_data_point"]), \
										Variable(sample_control["label"])
		test_prediction = test_neural_network.forward(control_input)
		test_loss = criterion(test_prediction, test_label)
		test_loss_list.append(test_loss)

total_test_loss = sum(test_loss_list)

print "total test loss: ", total_test_loss

print "total control loss: ", total_control_loss

