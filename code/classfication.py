import torch
import utils
import torch.nn as nn
from sklearn.model_selection import train_test_split
import random
import math
import config
from sklearn.metrics import f1_score


class Classification(nn.Module):
	def __init__(self, emb_size, num_classes):
		super(Classification, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.layer = nn.Sequential(
								nn.Linear(emb_size, num_classes)
								#nn.ReLU()
							)
	def init_params(self):
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, embeds):
		logists = torch.log_softmax(self.layer(embeds), 1)
		return logists



class Yelp_evaluation():
	def __init__(self):

		#load author label
		#id - label
		self.business_label = {}
		self.sample_num = 0
		with open('../data/yelp_business_category.txt') as infile:
			for line in infile.readlines():
				business, label = line.strip().split()[:2]
				business = int(business)
				label = int(label)

				self.business_label[business] = label
				self.sample_num += 1

	def train(self, model, emb):
		X = []
		Y = []
		for business in self.business_label:
			X.append(business)
			Y.append(self.business_label[business])

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
		model = train_classification(X_train, Y_train, emb, model)
		micro_f1, macro_f1 = evaluate(model, X_test, Y_test, emb)
		return  micro_f1, macro_f1


def train_classification(train_nodes, labels, features, classification, epochs=600):
	print('Training Classification ...')
	c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
	# train classification, detached from the current graph
	#classification.init_params()
	b_sz = 32
	for epoch in range(epochs):
		random.seed(0)
		random.shuffle(train_nodes)
		random.seed(0)
		random.shuffle(labels)
		batches = math.ceil(len(train_nodes) / b_sz)
		for index in range(batches):
			nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
			labels_batch = labels[index * b_sz : (index + 1) * b_sz]
			embs_batch = features[nodes_batch]

			logists = classification(torch.FloatTensor(embs_batch))
			loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss /= len(nodes_batch)
			# print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))
			loss.backward()

			nn.utils.clip_grad_norm_(classification.parameters(), 5)
			c_optimizer.step()
			c_optimizer.zero_grad()
	return classification

def evaluate(models, X_test, Y_test, embs):
	params = []

	for param in models.parameters():
		if param.requires_grad:
			param.requires_grad = False
			params.append(param)

	embs = embs[X_test]
	logists = models(torch.FloatTensor(embs))

	print(Y_test[0 : 20])

	_, Y_pred = torch.max(logists, 1)

	print(logists[0:20, :])
	print(Y_pred[0 : 20])



	micro_f1 = f1_score(Y_test, Y_pred, average='micro')
	macro_f1 = f1_score(Y_test, Y_pred, average='macro')
	print(micro_f1, macro_f1)
	return micro_f1, macro_f1

def yelp_classifiction(node_embed):
	yelp = Yelp_evaluation()
	model = Classification(64, 3)
	micro_f1, macro_f1 = yelp.train(model, node_embed)
	return micro_f1, macro_f1


if __name__ == "__main__":
	emb_1 =  utils.read_new_embeddings('../data/global_attk__seed_2_batch_32_moshi_0_dropout_0.2_W30_tanh_gtt_5_一层.txt')
	emb_2 = utils.read_embeddings(filename=config.pretrain_node_emb_filename_d,
                                                       n_node=config.node_size,
                                                       n_embed=config.n_emb)
	yelp_classifiction(emb_1)
	print("#################################")
	# yelp_classifiction(emb_2)