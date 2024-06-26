import torch
import torch.nn as nn
from torch.nn import init


"""
	PC-GNN Model
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class PCALayer(nn.Module):
	"""
	One Pick-Choose-Aggregate layer
	"""

	# The __init__ method initializes the PC-GNN model and takes in three parameters: num_classes which represents the number of classes (2 in the paper), inter1 which is the inter-relation aggregator that outputs the final embedding, and lambda_1 which is a hyperparameter used in the loss function. The method initializes several instance variables including self.inter1, self.xent, self.weight, self.lambda_1, and self.epsilon.
	def __init__(self, num_classes, inter1, lambda_1):
		"""
		Initialize the PC-GNN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(PCALayer, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()

		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
		init.xavier_uniform_(self.weight)
		self.lambda_1 = lambda_1
		self.epsilon = 0.1

	# The forward method takes in three parameters: nodes which represents the input nodes, labels which represents the labels for the input nodes, and train_flag which is a boolean flag indicating whether the model is in training mode or not. The method first calls the inter1 method to obtain the embeddings and label scores. It then applies a linear transformation to the embeddings using the self.weight parameter and returns the resulting scores and label scores.
	def forward(self, nodes, labels, train_flag=True):
		embeds1, label_scores = self.inter1(nodes, labels, train_flag)
		scores = self.weight.mm(embeds1)
		return scores.t(), label_scores

	# This method computes the probability scores for the input nodes and labels using the sigmoid activation function. It returns the probability scores for both GNN logits and label logits.
	def to_prob(self, nodes, labels, train_flag=True):
		gnn_logits, label_logits = self.forward(nodes, labels, train_flag)
		gnn_scores = torch.sigmoid(gnn_logits)
		label_scores = torch.sigmoid(label_logits)
		return gnn_scores, label_scores

	# The loss method takes in the same parameters as the forward method and computes the loss function of the PC-GNN model. It first calls the forward method to obtain the scores and label scores. It then computes the Simi loss and GNN loss as defined in the paper and combines them using the lambda_1 hyperparameter to obtain the final loss.
	def loss(self, nodes, labels, train_flag=True):
		gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
		# Simi loss, Eq. (7) in the paper
		label_loss = self.xent(label_scores, labels.squeeze())
		# GNN loss, Eq. (10) in the paper
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		# the loss function of PC-GNN, Eq. (11) in the paper
		final_loss = gnn_loss + self.lambda_1 * label_loss
		return final_loss
