import torch
import torch.nn as nn
import torch.nn.functional as F



class GCN(nn.Module):
    def __init__(self, A_hat, nb_layers=2, nb_feats=[1433,16,7]):
        super(GCN, self).__init__()
        print('Creating model')
        self.A_hat = A_hat #torch.Tensor(A_hat)
        self.nb_feats= nb_feats
        self.nb_layers = nb_layers
        # With explicit matrices multiplications
        '''self.layers = nn.ParameterList()
        for i in range(self.nb_layers):
            self.layers.append(nn.Parameter(torch.empty(self.nb_feats[i], self.nb_feats[i+1])))
            nn.init.xavier_uniform_(self.layers[i])'''
        # Replacing matrices with linear layers
        self.layers = nn.ModuleList()
        for i in range(self.nb_layers):
            self.layers.append(nn.Linear(self.nb_feats[i], self.nb_feats[i+1]))
            self.layers.append(nn.Dropout(p=0.0))

        print('Nb layers in GCN model: {}'.format(len(self.layers)))


    def forward(self, X):
        X = torch.squeeze(X)

        # With explicit matrices multiplications
        '''for i, layer in enumerate(self.layers[:-1]):
            X = F.relu(torch.mm(self.A_hat, torch.mm(X, layer)))
        X = F.softmax(torch.mm(self.A_hat, torch.mm(X, self.layers[self.nb_layers-1])), dim=1)'''
        # Replacing matrices with linear layers
        for i in range(self.nb_layers-1):
            X = F.relu(torch.sparse.mm(self.A_hat, self.layers[2*i+1](self.layers[2*i](X))))
        X = F.softmax(torch.sparse.mm(self.A_hat, self.layers[-1](self.layers[-2](X))), dim=1)

        return X

