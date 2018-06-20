import torch
from torch.utils.data import Dataset


class custom_loader(Dataset):
	"""
	bloody dataloader
	"""

	def __init__(self, training, annotations, length):
		"""
		Parameters:
		length : n_samples
		
		"""
		self.training = training
		self.annotations = annotations
		self.length = length

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		sample = {"training_data_point" : self.training[idx],
					"label" : self.annotations[idx]}
		return sample