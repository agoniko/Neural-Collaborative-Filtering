import os
import time
import random
import argparse
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

class NeuMF(nn.Module):
    def __init__(self, factor_num, num_users, num_items, layers):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = factor_num

        #GMF component
        self.embedding_user_gmf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
        self.embedding_item_gmf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

        self.affine_output_gmf = nn.Linear(in_features=self.factor_num, out_features=8, bias = False)


        #MLP component
        self.layers = layers

        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.factor_num*2, self.layers[0]))

        for in_size, out_size in zip(self.layers[:-1], self.layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        ##Models fusion
        self.mixing_layers = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            #nn.Dropout(p = 0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            #nn.Dropout(p = 0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, user_indices, item_indices):
        #GMF forward
        # (batch_size , factor_num)
        user_embedding_gmf = self.embedding_user_gmf(user_indices)
        item_embedding_gmf = self.embedding_item_gmf(item_indices)

        # (bacth_size , factor_num)
        element_product = torch.mul(user_embedding_gmf, item_embedding_gmf)
        # (batch_size, 8)
        logits_gmf = self.affine_output_gmf(element_product)

        # (batch_size, 8)
        ratings_gmf = logits_gmf

        #MLP forward
        # (batch_size, factor_num)
        user_embedding_mlp = self.embedding_user_mlp(user_indices)

        # (bacth_size, factor_num)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        # (batch_size, 2* factor_num)
        vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            # (batch_size, in_size) -> (batch_size, out_size)
            vector = self.fc_layers[idx](vector)
            if idx != len(self.fc_layers)-1:
                vector = nn.ReLU()(vector)
            # vector = nn.BatchNorm1d()(vector)
            vector = nn.Dropout(p=0.5)(vector)
        # (batch_size, 8)
        ratings_mlp =  vector

        #Models fusion
        # (batch_size, 8) cat (batch_size, 8) -> (batch_size, 16)
        ratings = torch.cat([ratings_gmf, ratings_mlp], dim = 1)

        return self.mixing_layers(ratings).squeeze()       