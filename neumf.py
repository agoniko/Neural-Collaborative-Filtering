import torch
import torch.nn as nn


class NeuMF(nn.Module):
    def __init__(self, factor_num, num_users, num_items, layer_sizes, dropout):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = factor_num
        self.test = 8
        self.dropout = dropout

        # GMF component
        self.gmf_user_embed = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num),
            nn.Dropout(p=self.dropout[2]),
        )
        self.gmf_item_embed = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num),
            nn.Dropout(p=self.dropout[3]),
        )
        self.gmf_affine = nn.Linear(
            in_features=self.factor_num, out_features=8, bias=False
        )

        # MLP component
        self.mlp_user_embed = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num),
            nn.Dropout(p=self.dropout[4]),
        )
        self.mlp_item_embed = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num),
            nn.Dropout(p=self.dropout[5]),
        )
        layers = []
        layers.append(nn.Linear(self.factor_num * 2, layer_sizes[0]))
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        layers.pop()
        self.mlp_fc = nn.Sequential(*layers)

        # Combine models
        self.mixing_layers = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout[0]),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(p=dropout[1]),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_indices, item_indices):
        # GMF forward
        user_embedding_gmf = self.gmf_user_embed(user_indices)
        item_embedding_gmf = self.gmf_item_embed(item_indices)

        element_product = torch.mul(user_embedding_gmf, item_embedding_gmf)
        ratings_gmf = self.gmf_affine(element_product)

        # MLP forward
        user_embedding_mlp = self.mlp_user_embed(user_indices)
        item_embedding_mlp = self.mlp_item_embed(item_indices)

        vector = torch.cat((user_embedding_mlp, item_embedding_mlp), dim=-1)
        ratings_mlp = self.mlp_fc(vector)

        # Models fusion
        # (batch_size, 8) cat (batch_size, 8) -> (batch_size, 16)
        ratings = torch.cat((ratings_gmf, ratings_mlp), dim=1)
        return self.mixing_layers(ratings).squeeze()
