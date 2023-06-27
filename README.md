# CFRHNE

## conda environment configuration

You can configure the environment with the following command.
```
conda install --yes --file coarse_requirements.txt
conda install --yes --file requirements.txt
```

## run
You can run the program with the following command. Of course, you can also run the program with pycharm.
First, you need to use a coarse model to map the high-dimensional features of the graph to discrete vectors, and then use the embedding as input to obtain better embeddings using adversarial learning.
```
python C:\xxx\coarse_code\coarse.py
python C:\xxx\code\Fine.py
```

## config
We put all hyperparameters and dataset settings in config.py.
```

coarse model

pos_size = 4                                 #  the number of positive samples
neg_size = 32                                #  the number of negative samples
n_emb = 150                                  #  the dimension of embeddings

walk_len = walk_lens[dataset]                #  the length of walking
restart_ratio = restart[dataset]             #  the Probability of returning to the starting node
top_k = top_ks[dataset]                      #  the limition of node sequences
n_heads = head_set[dataset]                  #  the number of attention heads
walk_NUM = 40                                #  the number of walking

-------------------------------------------------------------------------------------------------------

fine model

batch_size = 32    
lambda_gen = 1e-5                 #  generator Regularization
lambda_dis = 1e-5                 #  discriminator Regularization
n_sample = 16                     #  sample size
lr_gen = 0.0001                   #  the learing rate of generator 
lr_dis = 0.0001                   #  the learing rate of discriminator
n_epoch = 40                      #  the number of training

sig = 1                           #  the variance of the generator Gaussian distribution
walk_num = 6                      #  the number of walking
walk_len = 8                      #  the length of walking

change = 10                       #  Swap high-order low-order samples every 5 times
d_epoch = 15                      #  Number of iterations in discriminator
g_epoch = 7                       #  Number of iterations in generator

inf = 40                          #  The farthest distance between two vectors

```

## data
**XX_test_0.2:** 20% of the edges in the network are used as the test set for link predtiction, the format is``` source_node target_node relation_type```

**XX_train_0.8:** 80% of the edges in the network are used as the train set for link predtiction, the format is ``` source_node target_node relation_type```

**XX_label:** The label value of some nodes in the network, the format is ```id label```

**link.dat:** all edges in network, the format is ``` source_node target_node relation_type```

**neg_0.2:** 20% of the fake edges in the network are used as the test set for link predtiction, the format is ``` source_node target_node relation_type```

**neg_0.8:** 80% of the fake edges in the network are used as the train set for link prediction, the format is ``` source_node target_node relation_type```

## evaluation
We have written corresponding evaluation methods for each dataset in acm_evaluation.py, aminer_evaluation.py, dblp_evaluation.py and yelp_evaluation.py.
