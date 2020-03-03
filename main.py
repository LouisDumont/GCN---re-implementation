import torch
import torch.nn as nn
# The idea here is not to use the DGL library

import numpy as np
import os
import random
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import time

from load_cora import load_cora
from gcn import GCN

### Load Data ###
data_path = os.getcwd() + '/cora_own/'

start = time.time()
X, A, labels = load_cora(data_path+'cora.content', data_path+'cora.cites')
print('Dataset extracted in {}'.format(time.time()-start))

nb_nodes = labels.size
nb_feat = X.shape[1]

### Add self-connections, compute A_hat ###
start = time.time()
A_norm = A + np.eye(nb_nodes)
D_norm = np.diag(np.sum(A_norm, axis=0))

D_invsqrt = np.linalg.inv(np.sqrt(D_norm))
A_hat = D_invsqrt @ A_norm @ D_invsqrt
print('Creating the normalized-Slef-cautious adjacency matrix {}'.format(time.time()-start))

### Sate random seeds for reproducability ###
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

### Parameters for the training ###
gcn_model = GCN(A_hat, 2, [1433,16,7])
lr = 0.01
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=lr)
nb_epochs = 200
loss_function = nn.CrossEntropyLoss()
L2_reg_lambda = 5e-2 # We obtain better results with this parameter (compared to the original one)
# We also found dropout to harm the final accuracy

### Select the labeled nodes for training (20 samples for each class) ###
train_labels_idx = []
for class_ in range(7):
    available = np.argwhere(labels==class_)
    available = np.squeeze(available)
    print(available.shape)
    samples = np.random.choice(available, size=20, replace=False)
    train_labels_idx += list(samples)

labels = torch.Tensor(labels).type(torch.LongTensor)
# Normalize data (expected by the Glorot init?)
# X = scale(X, axis=0) # scling X always hurts the final accuracy
X_tensor = torch.Tensor(X)
X_tensor = X_tensor.unsqueeze(0)

### Define training framework ###
def train(model):
    print(model)
    for i in range(nb_epochs):
        start = time.time()
        # predict
        predict_scores = model(X_tensor)

        # Define loss
        loss = loss_function(predict_scores[train_labels_idx], labels[train_labels_idx])
        loss += L2_reg_lambda * model.layers[0].weight.norm(2)

        # Back-propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss, train accuracy and total accuracy
        predict_labels = np.argmax(predict_scores.data, axis=1)
        train_accuracy = accuracy_score(predict_labels[train_labels_idx], labels.data[train_labels_idx])
        global_accuracy = accuracy_score(predict_labels, labels.data)

        if i%10 == 0:
            print('Epoch {}: Loss {}, Train Accuracy {}, Global Accuracy {}, Duration: {}'.format(i, loss.data, train_accuracy, global_accuracy, time.time()-start))

    pass

if __name__=='__main__':
    train(gcn_model)





