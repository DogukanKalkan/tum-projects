import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # TODO: Define model
        self.weight_normed_linear1 = torch.nn.utils.weight_norm(nn.Linear(in_features=latent_size + 3,out_features=512),dim=0)
        self.weight_normed_linear2 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512),dim=0)
        self.weight_normed_linear3 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512),dim=0)
        self.weight_normed_linear4 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512-(latent_size+3)),dim=0)
        self.weight_normed_linear5 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512),dim=0)
        self.weight_normed_linear6 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512),dim=0)
        self.weight_normed_linear7 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512),dim=0)
        self.weight_normed_linear8 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512),dim=0)
        self.linear9 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)


    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x = self.weight_normed_linear1(x_in)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.weight_normed_linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.weight_normed_linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.weight_normed_linear4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x_concat = torch.hstack((x,x_in))
        x = self.weight_normed_linear5(x_concat)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.weight_normed_linear6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.weight_normed_linear7(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.weight_normed_linear8(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear9(x)
        return x
