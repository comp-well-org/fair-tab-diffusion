import os
import sys
import dgl
import time
import json
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
from dgl.nn import GraphConv, SAGEConv
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, masked_select_nnz

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

warnings.filterwarnings('ignore')

################################################################################
# data
def preprocess(data_dir):
    xn_train = pd.read_csv(os.path.join(data_dir, 'xn_train.csv'), index_col=0)
    xn_eval = pd.read_csv(os.path.join(data_dir, 'xn_eval.csv'), index_col=0)
    xn_test = pd.read_csv(os.path.join(data_dir, 'xn_test.csv'), index_col=0)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), index_col=0)
    y_eval = pd.read_csv(os.path.join(data_dir, 'y_eval.csv'), index_col=0)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), index_col=0)
    
    # only the first column of y_train, y_eval, y_test is used
    y_train = y_train.iloc[:, 0]
    y_eval = y_eval.iloc[:, 0]
    y_test = y_test.iloc[:, 0]
    
    data_train = pd.concat([xn_train, y_train], axis=1)
    data_eval = pd.concat([xn_eval, y_eval], axis=1)
    data_test = pd.concat([xn_test, y_test], axis=1)
    # print(data_train.head())
    
    with open(os.path.join(data_dir, 'desc.json')) as f:
        desc = json.load(f)
    # print(json.dumps(desc, indent=4))
    
    categories = desc['n_unq_cat_od_x_lst'] + [desc['n_unq_y']]
    d_numerical = desc['d_num_x']
    X_train_num = data_train.iloc[:, :d_numerical].values
    X_eval_num = data_eval.iloc[:, :d_numerical].values
    X_test_num = data_test.iloc[:, :d_numerical].values
    
    X_train_cat = data_train.iloc[:, d_numerical:].values
    X_eval_cat = data_eval.iloc[:, d_numerical:].values
    X_test_cat = data_test.iloc[:, d_numerical:].values
    
    # convert X_train_cat, X_eval_cat, X_test_cat to one-hot encoding
    X_train_cat = categorical_to_onehot(X_train_cat, categories)
    X_eval_cat = categorical_to_onehot(X_eval_cat, categories)
    X_test_cat = categorical_to_onehot(X_test_cat, categories)
    
    X_num_sets = (X_train_num, X_eval_num, X_test_num)
    X_cat_sets = (X_train_cat, X_eval_cat, X_test_cat)
    
    return X_num_sets, X_cat_sets, categories, d_numerical

################################################################################
# model
class Encoder(nn.Module):
    def __init__(self, input_dim, encoder_dim, encoder_l, device):
        super(Encoder, self).__init__()
        encoder = nn.ModuleList([nn.Linear(input_dim, encoder_dim), nn.ReLU()])
        for _ in range(encoder_l - 2):
            encoder_dim_ = int(encoder_dim / 2)
            encoder.append(nn.Linear(encoder_dim, encoder_dim_))
            encoder.append(nn.ReLU())
            encoder_dim = encoder_dim_
        self.encoder = nn.Sequential(*encoder)
        self.encode_mu = nn.Linear(encoder_dim, input_dim)
        self.encode_logvar = nn.Linear(encoder_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu_z, logvar_z = self.encode_mu(h), self.encode_logvar(h)
        z = self.reparameterize(mu_z, logvar_z)
        return z, (mu_z, logvar_z)

class LearnedGraph(nn.Module):
    def __init__(self, input_dim, graph_prior, prior_mask, threshold, device):
        super(LearnedGraph, self).__init__()
        self.graph = nn.Parameter(torch.zeros(input_dim, input_dim, requires_grad=True, device=device))
        if all(i is not None for i in [graph_prior, prior_mask]):
            self.graph_prior = graph_prior.detach().clone().requires_grad_(False).to(device)
            self.prior_mask = prior_mask.detach().clone().requires_grad_(False).to(device)
            self.use_prior = True
        else:
            self.use_prior = False

        self.act = nn.Sigmoid()
        self.threshold = nn.Threshold(threshold, 0)
        self.device = device

    def forward(self, iter):
        if self.use_prior:
            graph = self.prior_mask * self.graph_prior + (1 - self.prior_mask) * self.graph
        else:
            graph = self.graph
        graph = self.act(graph)
        graph = graph.clone()
        graph = graph * (torch.ones(graph.shape[0]).to(self.device) - torch.eye(graph.shape[0]).to(self.device))
        graph = graph + torch.eye(graph.shape[0]).to(self.device)
        if iter > 50:
            graph = self.threshold(graph)
        else:
            graph = graph
        return graph

class GraphInputProcessorHomo(nn.Module):
    def __init__(self, input_dim, decoder_dim, het_encoding, device):
        super(GraphInputProcessorHomo, self).__init__()
        self.device = device
        self.het_encoding = het_encoding

        if het_encoding:
            feat_dim = input_dim + 1
        else:
            feat_dim = 1

        self.embedding_functions = []
        for _ in range(input_dim):
            self.embedding_functions.append(nn.Sequential(nn.Linear(feat_dim, decoder_dim), nn.Tanh()).to(device))

    def forward(self, z, adj):
        b_z = z.unsqueeze(-1)
        b_size, n_nodes, _ = b_z.shape

        if self.het_encoding:
            one_hot_encoding = torch.eye(n_nodes).to(self.device)
            b_encoding = torch.stack([one_hot_encoding for _ in range(b_size)], dim=0)
            b_z = torch.cat([b_z, b_encoding], dim=-1)

        b_z = [f(b_z[:, i]) for i, f in enumerate(self.embedding_functions)]
        b_z = torch.stack(b_z, dim=1)
        b_z = torch.flatten(b_z, start_dim=0, end_dim=1)

        edge_index = adj.nonzero().t()
        row, col = edge_index
        edge_weight = adj[row, col]

        g = dgl.graph((edge_index[0], edge_index[1]))
        b_adj = dgl.batch([g] * b_size)
        b_edge_weight = edge_weight.repeat(b_size)

        return (b_z, b_adj, b_edge_weight)

class GraphInputProcessorHet(nn.Module):
    def __init__(self, input_dim, decoder_dim, n_edge_types, het_encoding, device):
        super(GraphInputProcessorHet, self).__init__()
        self.n_edge_types = n_edge_types
        self.device = device
        self.het_encoding = het_encoding
        if het_encoding:
            feat_dim = input_dim + 1
        else:
            feat_dim = 1
        self.embedding_functions = []
        for _ in range(input_dim):
            self.embedding_functions.append(nn.Sequential(nn.Linear(feat_dim, decoder_dim), nn.Tanh()).to(device))

    def forward(self, z, adj):
        b_size, n_nodes = z.shape
        b_z = z.unsqueeze(-1)
        if self.het_encoding:
            one_hot_encoding = torch.eye(n_nodes).to(self.device)
            b_encoding = torch.stack([one_hot_encoding for _ in range(b_size)], dim=0)
            b_z = torch.cat([b_z, b_encoding], dim=-1)

        b_z = [f(b_z[:, i]) for i, f in enumerate(self.embedding_functions)]
        b_z = torch.stack(b_z, dim=1)
        b_size, n_nodes, n_feats = b_z.shape

        n_edge_types = self.n_edge_types
        edge_types = torch.arange(1, n_edge_types + 1, 1).reshape(n_nodes, n_nodes)

        b_adj = torch.stack([adj for _ in range(b_size)], dim=0)

        b_edge_index, b_edge_weights = dense_to_sparse(b_adj)
        r, c = b_edge_index
        b_edge_types = edge_types[r % n_nodes, c % n_nodes]
        b_z = b_z.reshape(b_size * n_nodes, n_feats)
        
        return (b_z, b_edge_index, b_edge_weights, b_edge_types)

class GraphDecoderHomo(nn.Module):
    def __init__(self, decoder_dim, decoder_l, decoder_arch, device):
        super(GraphDecoderHomo, self).__init__()
        decoder = nn.ModuleList([])
        if decoder_arch == 'gcn':
            for i in range(decoder_l):
                if i == decoder_l - 1:
                    decoder.append(GraphConv(decoder_dim, 1, norm='both', weight=True, bias=True))
                else:
                    decoder_dim_ = int(decoder_dim / 2)
                    decoder.append(
                        GraphConv(
                            decoder_dim, decoder_dim_, norm='both', 
                            weight=True, bias=True, activation=nn.Tanh(),
                        ),
                    )
                    decoder_dim = decoder_dim_
        elif decoder_arch == 'sage':
            for i in range(decoder_l):
                if i == decoder_l - 1:
                    decoder.append(SAGEConv(decoder_dim, 1, aggregator_type='mean', bias=True))
                else:
                    decoder_dim_ = int(decoder_dim / 2)
                    decoder.append(
                        SAGEConv(
                            decoder_dim, decoder_dim_, aggregator_type='mean',
                            bias=True, activation=nn.Tanh(),
                        ),
                    )
                    decoder_dim = decoder_dim_
        else:
            raise Exception('decoder can only be {het|gcn|sage}')
        self.decoder = nn.Sequential(*decoder)

    def forward(self, graph_input, b_size):
        b_z, b_adj, b_edge_weight = graph_input
        for layer in self.decoder:
            b_z = layer(b_adj, feat=b_z, edge_weight=b_edge_weight)
        x_hat = b_z.reshape(b_size, -1)
        return x_hat

class GraphDecoderHet(nn.Module):
    def __init__(self, decoder_dim, decoder_l, n_edge_types, device):
        super(GraphDecoderHet, self).__init__()
        decoder = nn.ModuleList([])
        for i in range(decoder_l):
            if i == decoder_l - 1:
                decoder.append(RGCNConv(decoder_dim, 1, num_relations=n_edge_types + 1, root_weight=False))
            else:
                decoder_dim_ = int(decoder_dim / 2)
                decoder.append(RGCNConv(decoder_dim, decoder_dim_, num_relations=n_edge_types + 1, root_weight=False))
                decoder.append(nn.ReLU())
                decoder_dim = decoder_dim_
        self.decoder = nn.Sequential(*decoder)

    def forward(self, graph_input, b_size):
        b_z, b_edge_index, b_edge_weights, b_edge_types = graph_input
        h = b_z
        for layer in self.decoder:
            if not isinstance(layer, nn.ReLU):
                h = layer(h, b_edge_index, b_edge_types, b_edge_weights)
            else:
                h = layer(h)
        x_hat = h.reshape(b_size, -1)
        return x_hat

_WITH_PYG_LIB = False

def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout='coo')

class RGCNConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = 'mean',
        root_weight: bool = True,
        is_sorted: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)
        self._WITH_PYG_LIB = torch.cuda.is_available() and _WITH_PYG_LIB

        if num_bases is not None and num_blocks is not None:
            raise ValueError(
                'Can not apply both basis-decomposition and block-diagonal-decomposition at the same time.',
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.is_sorted = is_sorted

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        if num_bases is not None:
            self.weight = Parameter(
                torch.Tensor(num_bases, in_channels[0], out_channels),
            )
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))

        elif num_blocks is not None:
            if in_channels[0] % num_blocks != 0 and out_channels % num_blocks != 0:
                raise AssertionError(
                    'Channels must be divisible by num_blocks, for RGCNConv.',
                )
            self.weight = Parameter(
                torch.Tensor(
                    num_relations,
                    num_blocks,
                    in_channels[0] // num_blocks,
                    out_channels // num_blocks,
                ),
            )
            self.register_parameter('comp', None)
        else:
            self.weight = Parameter(
                torch.Tensor(num_relations, in_channels[0], out_channels),
            )
            self.register_parameter('comp', None)
        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

    def forward(
        self,
        x: Union[OptTensor, Tuple[OptTensor, Tensor]],
        edge_index: Adj,
        edge_type: OptTensor = None,
        edge_weight: OptTensor = None,
    ):
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)
        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]
        size = (x_l.size(0), x_r.size(0))
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        if edge_type is None:
            raise AssertionError('edge_type cannot be None for RGCNConv.')
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)
        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels,
            )

        if self.num_blocks is not None:
            if x_l.dtype == torch.long and self.num_blocks is not None:
                raise ValueError(
                    'Block-diagonal decomposition not supported for non-continuous input features.',
                )
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                out = out + h.contiguous().view(-1, self.out_channels)
        else:
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                if edge_weight is not None:
                    tmp_weight = edge_weight[edge_type == i]
                else:
                    tmp_weight = None
                h = self.propagate(
                    tmp,
                    x=x_l,
                    edge_type_ptr=None,
                    edge_weight=tmp_weight,
                    size=size,
                )
                out = out + (h @ weight[i])
        root = self.root
        if root is not None:
            out = out + (root[x_r] if x_r.dtype == torch.long else x_r @ root)
        if self.bias is not None:
            out = out + self.bias
        return out

class Goggle(nn.Module):
    def __init__(
        self, input_dim, encoder_dim=64, encoder_l=2, het_encoding=True,
        decoder_dim=64, decoder_l=2, threshold=0.1, decoder_arch='gcn',
        graph_prior=None, prior_mask=None,
        device='cpu',
    ):
        super(Goggle, self).__init__()
        self.input_dim = input_dim
        self.device = device
        self.learned_graph = LearnedGraph(input_dim, graph_prior, prior_mask, threshold, device)
        self.encoder = Encoder(input_dim, encoder_dim, encoder_l, device)
        if decoder_arch == 'het':
            n_edge_types = input_dim * input_dim
            self.graph_processor = GraphInputProcessorHet(input_dim, decoder_dim, n_edge_types, het_encoding, device)
            self.decoder = GraphDecoderHet(decoder_dim, decoder_l, n_edge_types, device)
        else:
            self.graph_processor = GraphInputProcessorHomo(input_dim, decoder_dim, het_encoding, device)
            self.decoder = GraphDecoderHomo(decoder_dim, decoder_l, decoder_arch, device)

    def forward(self, x, iter):
        z, (mu_z, logvar_z) = self.encoder(x)
        b_size, _ = z.shape
        adj = self.learned_graph(iter)
        graph_input = self.graph_processor(z, adj)
        x_hat = self.decoder(graph_input, b_size)
        return x_hat, adj, mu_z, logvar_z

    def sample(self, count):
        with torch.no_grad():
            mu = torch.zeros(self.input_dim)
            sigma = torch.ones(self.input_dim)
            q = torch.distributions.Normal(mu, sigma)
            z = q.rsample(sample_shape=torch.Size([count])).squeeze().to(self.device)

            self.learned_graph.eval()
            self.graph_processor.eval()
            self.decoder.eval()

            adj = self.learned_graph(100)
            graph_input = self.graph_processor(z, adj)
            synth_x = self.decoder(graph_input, count)
        return synth_x

class GoggleLoss(nn.Module):
    def __init__(self, alpha=1, beta=0, graph_prior=None, device='cpu'):
        super(GoggleLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.device = device
        self.alpha = alpha
        self.beta = beta
        if graph_prior is not None:
            self.use_prior = True
            self.graph_prior = torch.Tensor(graph_prior).requires_grad_(False).to(device)
        else:
            self.use_prior = False

    def forward(self, x_recon, x, mu, logvar, graph):
        loss_mse = self.mse_loss(x_recon, x)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if self.use_prior:
            loss_graph = (graph - self.graph_prior).norm(p=1) / torch.numel(graph)
        else:
            loss_graph = graph.norm(p=1) / torch.numel(graph)
        loss = loss_mse + self.alpha * loss_kld + self.beta * loss_graph
        return loss, loss_mse, loss_kld, loss_graph

class GoggleModel:
    def __init__(
        self, ds_name, input_dim, encoder_dim=256, encoder_l=2,
        het_encoding=True, decoder_dim=256, decoder_l=2, threshold=0.1,
        decoder_arch='gcn', graph_prior=None, prior_mask=None, device='cpu',
        alpha=0.1, beta=0.1, seed=42, iter_opt=True,
        **kwargs,
    ):
        self.ds_name = ds_name
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        self.learning_rate = kwargs.get('learning_rate', 5e-3)
        self.weight_decay = kwargs.get('weight_decay', 1e-3)
        self.epochs = kwargs.get('epochs', 1000)
        self.batch_size = kwargs.get('batch_size', 1024)
        self.patience = kwargs.get('patience', 50)
        self.logging_epoch = kwargs.get('logging', 100)
        self.loss = GoggleLoss(alpha, beta, graph_prior, device)
        self.model = Goggle(
            input_dim, encoder_dim, encoder_l, het_encoding,
            decoder_dim, decoder_l, threshold, decoder_arch,
            graph_prior, prior_mask, device,
        ).to(device)
        self.iter_opt = iter_opt
        if iter_opt:
            gl_params = ['learned_graph.graph']
            graph_learner_params = list(
                filter(lambda kv: kv[0] in gl_params, self.model.named_parameters()),
            )
            graph_autoencoder_params = list(
                filter(lambda kv: kv[0] not in gl_params, self.model.named_parameters()),
            )
            self.optimiser_gl = torch.optim.Adam(
                [param[1] for param in graph_learner_params],
                lr=self.learning_rate,
                weight_decay=0,
            )
            self.optimiser_ga = torch.optim.Adam(
                [param[1] for param in graph_autoencoder_params],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimiser = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

    def evaluate(self, data_loader, epoch):
        with torch.no_grad():
            eval_loss, rec_loss, kld_loss, graph_loss = 0.0, 0.0, 0.0, 0.0
            num_samples = 0
            for _, data in enumerate(data_loader):
                self.model.eval()
                data = data[0].to(self.device)
                x_hat, adj, mu_z, logvar_z = self.model(data, epoch)
                loss, loss_rec, loss_kld, loss_graph = self.loss(
                    x_hat, data, mu_z, logvar_z, adj,
                )
                eval_loss += loss.item()
                rec_loss += loss_rec.item()
                kld_loss += loss_kld.item()
                graph_loss += loss_graph.item() * data.shape[0]
                num_samples += data.shape[0]
            eval_loss /= num_samples
            rec_loss /= num_samples
            kld_loss /= num_samples
            graph_loss /= num_samples
            return eval_loss, rec_loss, kld_loss, graph_loss

    def fit(self, train_loader, model_save_path):
        best_loss = np.inf
        for epoch in range(self.epochs):
            train_loss, num_samples = 0.0, 0
            train_loss_epoch = 0.0
            for i, data in enumerate(train_loader):
                data = data.float()
                if self.iter_opt:
                    if i % 2 == 0:
                        self.model.train()
                        self.optimiser_ga.zero_grad()
                        data = data.to(self.device)
                        x_hat, adj, mu_z, logvar_z = self.model(data, epoch)
                        loss, _, _, _ = self.loss(x_hat, data, mu_z, logvar_z, adj)
                        loss.backward(retain_graph=True)
                        self.optimiser_ga.step()
                        train_loss += loss.item()
                        num_samples += data.shape[0]
                    else:
                        self.model.train()
                        self.optimiser_gl.zero_grad()
                        data = data.to(self.device)
                        x_hat, adj, mu_z, logvar_z = self.model(data, epoch)
                        loss, _, _, _ = self.loss(x_hat, data, mu_z, logvar_z, adj)
                        loss.backward(retain_graph=True)
                        self.optimiser_gl.step()
                        train_loss += loss.item()
                        num_samples += data.shape[0]
                else:
                    data = data[0].to(self.device)
                    self.model.train()
                    self.optimiser.zero_grad()
                    x_hat, adj, mu_z, logvar_z = self.model(data, epoch)
                    loss, _, _, _ = self.loss(x_hat, data, mu_z, logvar_z, adj)
                    loss.backward(retain_graph=True)
                    self.optimiser.step()
                    train_loss += loss.item()
                    num_samples += data.shape[0]
                    
                train_loss_epoch = train_loss / num_samples
                if epoch == self.epochs - 1 and i == len(train_loader) - 1:
                    print(f'training -> epoch: {epoch + 1}/{self.epochs}, loss: {train_loss_epoch:.4f} -- best: {best_loss:.4f}')
                else:
                    print(f'training -> epoch: {epoch + 1}/{self.epochs}, loss: {train_loss_epoch:.4f} -- best: {best_loss:.4f}', end='\r')
                
            train_loss /= num_samples
            if train_loss <= best_loss:
                best_loss = train_loss
                torch.save(self.model.state_dict(), model_save_path)

    def sample(self, count):
        X_synth = self.model.sample(count)
        X_synth = X_synth.cpu().detach().numpy()
        return X_synth

################################################################################
# utils
def categorical_to_onehot(cat_matrix, categories):
    cat_matrix = cat_matrix.astype(int)
    # create a list to store the one-hot encoded values
    onehot = []
    # iterate over the columns of the categorical matrix
    for i in range((cat_matrix.shape[1])):
        # create a one-hot encoded matrix for the i-th column
        onehot_i = np.eye(categories[i])[cat_matrix[:, i]]
        # append the one-hot encoded matrix to the list
        onehot.append(onehot_i)
    # concatenate the one-hot encoded matrices along the columns
    return np.concatenate(onehot, axis=1)

def onehot_to_categorical(onehot_matrix, categories):
    # create a list to store the categorical values
    categorical = []
    # iterate over the columns of the one-hot matrix
    st = 0
    for i in range(len(categories)):
        ed = st + categories[i]
        # create a categorical matrix for the i-th column
        categorical_i = np.argmax(onehot_matrix[:, st:ed], axis=1)
        # append the categorical matrix to the list
        categorical.append(categorical_i)
        st = ed
    return np.stack(categorical, axis=1)

################################################################################
# main
def main():
    # global variables
    device = 'cuda:1'
    
    # TODO: configs
    dataname = 'adult'
    lr = 1e-3
    n_epochs = 1
    batch_size = 256
    n_samples = 1000
    
    # data
    dataset_dir = f'/rdf/db/public-tabular-datasets/{dataname}/'
    ckpt_dir = f'./ckpt/{dataname}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    X_num_sets, X_cat_sets, categories, d_numerical = preprocess(dataset_dir)
    X_train_num, X_eval_num, X_test_num = X_num_sets
    X_train_cat, X_eval_cat, X_test_cat = X_cat_sets
    X_train_num = torch.tensor(X_train_num.astype(np.float32)).float()
    X_train_cat = torch.tensor(X_train_cat.astype(np.int32)).long()
    categories = np.array(categories)
    X_train = torch.cat([X_train_num, X_train_cat], dim=1)
    
    # model
    gen = GoggleModel(
        ds_name=dataname, input_dim=X_train.shape[1], encoder_dim=2048,
        encoder_l=4, het_encoding=True, decoder_dim=2048,
        decoder_l=4, threshold=0.1, decoder_arch='gcn',
        graph_prior=None, prior_mask=None, beta=1,
        learning_rate=lr, seed=42, epochs=n_epochs,
        batch_size=batch_size, device=device,
    )
    
    num_params = sum(p.numel() for p in gen.model.encoder.parameters() if p.requires_grad)
    print(f'number of parameters in encoder: {num_params}')
    
    # training
    start_time = time.time()
    train_loader = DataLoader(X_train, batch_size=gen.batch_size, shuffle=True)
    gen.fit(train_loader, f'{ckpt_dir}/model.pt')
    end_time = time.time()
    print(f'training time: {(end_time - start_time):.2f}s')
    
    # sampling
    gen.model.load_state_dict(torch.load(f'{ckpt_dir}/model.pt'))
    start_time = time.time()
    samples = gen.sample(n_samples)
    sample_con = samples[:, :d_numerical]
    sample_dis = samples[:, d_numerical:]
    sample_dis = onehot_to_categorical(sample_dis, categories)
    sample = np.concatenate([sample_con, sample_dis], axis=1)
    end_time = time.time()
    print(f'sampling time: {(end_time - start_time):.2f}s')
    
    sample = pd.DataFrame(sample)
    print(sample.head(3))

if __name__ == '__main__':
    main()
