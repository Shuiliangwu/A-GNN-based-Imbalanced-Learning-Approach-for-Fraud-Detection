import time, datetime
import os
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import test_pcgnn, test_sage, load_data, pos_neg_split, normalize, pick_step
from src.model import PCALayer
from src.layers import InterAgg, IntraAgg
from src.graphsage import *


"""
	Training PC-GNN
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
"""

# The main purpose of the code is to train and test the PC-GNN model on imbalanced datasets and compare its performance against other GNN models like GraphSAGE and GCN.
class ModelHandler(object):

	# The class has an init method that takes the configuration as input and initializes the class with the given configuration, including loading the data, splitting it into train, validation, and test sets, normalizing the feature data, and setting the input graph for the model based on the selected model architecture.
	def __init__(self, config):
		args = argparse.Namespace(**config)
		# load graph, feature, and label
		# This method initializes the class with the given configuration. It loads the graph, feature, and label data using the load_data function from the utils.py file.
		[homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data_name, prefix=args.data_dir)

		# train_test split
		# It performs a train-test split on the data using the train_test_split function from the sklearn library
		np.random.seed(args.seed)
		random.seed(args.seed)
		if args.data_name == 'yelp':
			index = list(range(len(labels)))
			idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=args.train_ratio,
																	random_state=2, shuffle=True)
			idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio,
																	random_state=2, shuffle=True)

		elif args.data_name == 'amazon':  # amazon
			# 0-3304 are unlabeled nodes
			index = list(range(3305, len(labels)))
			idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[3305:], stratify=labels[3305:],
																	train_size=args.train_ratio, random_state=2, shuffle=True)
			idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
																	test_size=args.test_ratio, random_state=2, shuffle=True)

		print(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},'+
			f'valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}')
		print(f"Classification threshold: {args.thres}")
		print(f"Feature dimension: {feat_data.shape[1]}")


		# split pos neg sets for under-sampling
		# This function splits the training set into positive and negative examples based on their labels. The positive examples are those with a label of 1, while the negative examples are those with a label of 0. 
		train_pos, train_neg = pos_neg_split(idx_train, y_train)

		
		# if args.data == 'amazon':
		# The normalized matrix will have each row summing up to 1.
		feat_data = normalize(feat_data)
		# train_feats = feat_data[np.array(idx_train)]
		# scaler = StandardScaler()
		# scaler.fit(train_feats)
		# feat_data = scaler.transform(feat_data)
		args.cuda = not args.no_cuda and torch.cuda.is_available()
		os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

		# set input graph
		# This conditional statement is used to set the input graph for the model based on the selected model architecture. If the model is a SAGE or GCN model, then the input graph is a homogeneous graph represented by the homo variable. Otherwise, the input graph is a heterogeneous graph represented by the relation1, relation2, and relation3 variables.
		if args.model == 'SAGE' or args.model == 'GCN':
			adj_lists = homo
		else:
			adj_lists = [relation1, relation2, relation3]

		print(f'Model: {args.model}, multi-relation aggregator: {args.multi_relation}, emb_size: {args.emb_size}.')
		
		# The resulting data and configuration are stored in the dataset and args attributes of the class, respectively
		self.args = args
		self.dataset = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'homo': homo,
						'idx_train': idx_train, 'idx_valid': idx_valid, 'idx_test': idx_test,
						'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
						'train_pos': train_pos, 'train_neg': train_neg}


	# This method is responsible for training the model. It first initializes the model's input features and builds the one-layer models (PC-GNN, SAGE, or GCN) based on the selected model architecture. Then, it initializes the optimizer and loops through the epochs to train the model using mini-batch training. The method also performs model validation after every "valid_epochs" epochs and saves the best model based on the highest AUC score. After training, the best model is loaded, and the test performance metrics (F1-macro, F1 for positive class, F1 for negative class, AUC, and G-mean) are returned.
	def train(self):
		args = self.args
		feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']
		idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
		idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset['y_valid'], self.dataset['idx_test'], self.dataset['y_test']
		# initialize model input
		features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
		features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
		if args.cuda:
			features.cuda()

		# build one-layer models
		if args.model == 'PCGNN':
			intra1 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra2 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra3 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], 
							  adj_lists, [intra1, intra2, intra3], inter=args.multi_relation, cuda=args.cuda)
		elif args.model == 'SAGE':
			agg_sage = MeanAggregator(features, cuda=args.cuda)
			enc_sage = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_sage, gcn=False, cuda=args.cuda)
		elif args.model == 'GCN':
			agg_gcn = GCNAggregator(features, cuda=args.cuda)
			enc_gcn = GCNEncoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_gcn, gcn=True, cuda=args.cuda)

		if args.model == 'PCGNN':
			gnn_model = PCALayer(2, inter1, args.alpha)
		elif args.model == 'SAGE':
			# the vanilla GraphSAGE model as baseline
			enc_sage.num_samples = 5
			gnn_model = GraphSage(2, enc_sage)
		elif args.model == 'GCN':
			gnn_model = GCN(2, enc_gcn)

		if args.cuda:
			gnn_model.cuda()

		# The function then initializes the optimizer and sets up a directory to save the model. 
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

		timestamp = time.time()
		timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
		dir_saver = args.save_dir+timestamp
		path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.data_name, args.model))
		f1_mac_best, auc_best, ep_best = 0, 0, -1

		# train the model
		for epoch in range(args.num_epochs):
			sampled_idx_train = pick_step(idx_train, y_train, self.dataset['homo'], size=len(self.dataset['train_pos'])*2)
			
			random.shuffle(sampled_idx_train)

			num_batches = int(len(sampled_idx_train) / args.batch_size) + 1

			loss = 0.0
			epoch_time = 0

			# mini-batch training
			for batch in range(num_batches):
				start_time = time.time()
				i_start = batch * args.batch_size
				i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
				batch_nodes = sampled_idx_train[i_start:i_end]
				batch_label = self.dataset['labels'][np.array(batch_nodes)]
				optimizer.zero_grad()
				if args.cuda:
					loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
				else:
					loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
				loss.backward()
				optimizer.step()
				end_time = time.time()
				epoch_time += end_time - start_time
				loss += loss.item()

			print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')

			# Valid the model for every $valid_epoch$ epoch
			if epoch % args.valid_epochs == 0:
				if args.model == 'SAGE' or args.model == 'GCN':
					print("Valid at epoch {}".format(epoch))
					f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_sage(idx_valid, y_valid, gnn_model, args.batch_size, args.thres)
					if auc_val > auc_best:
						f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)
				else:
					print("Valid at epoch {}".format(epoch))
					f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_pcgnn(idx_valid, y_valid, gnn_model, args.batch_size, args.thres)
					if auc_val > auc_best:
						f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)

		print("Restore model from epoch {}".format(ep_best))
		print("Model path: {}".format(path_saver))
		gnn_model.load_state_dict(torch.load(path_saver))
		if args.model == 'SAGE' or args.model == 'GCN':
			f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_sage(idx_test, y_test, gnn_model, args.batch_size, args.thres)
		else:
			f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_pcgnn(idx_test, y_test, gnn_model, args.batch_size, args.thres)
		return f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test
