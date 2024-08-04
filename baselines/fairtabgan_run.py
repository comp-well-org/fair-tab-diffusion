import os
import sys
import copy
import json
import time
import warnings
import argparse
import torch
import numpy as np
import pandas as pd
import skops.io as sio
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import TensorDataset, DataLoader

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

from lib import load_config, copy_file
from src.evaluate.skmodels import default_sk_clf

warnings.filterwarnings('ignore')

def get_ohe_data(df, S, Y, S_under, Y_desire):
    df_int = df.select_dtypes(['float', 'integer']).values
    continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)

    scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
    df_int = scaler.fit_transform(df_int)

    df_cat = df.select_dtypes('object')
    df_cat_names = list(df.select_dtypes('object').columns)
    numerical_array = df_int
    ohe = OneHotEncoder()
    ohe_array = ohe.fit_transform(df_cat)

    cat_lens = [i.shape[0] for i in ohe.categories_]
    discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))

    S_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(S)])
    Y_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(Y)])

    if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(S)][0] == S_under:
        underpriv_index = 0
        priv_index = 1
    else:
        underpriv_index = 1
        priv_index = 0
    if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(Y)][0] == Y_desire:
        desire_index = 0
        undesire_index = 1
    else:
        desire_index = 1
        undesire_index = 0

    final_array = np.hstack((numerical_array, ohe_array.toarray()))
    return {
        'ohe': ohe,
        'scaler': scaler,
        'discrete_columns_ordereddict': discrete_columns_ordereddict,
        'continuous_columns_list': continuous_columns_list,
        'final_array': final_array,
        'S_start_index': S_start_index,
        'Y_start_index': Y_start_index,
        'underpriv_index': underpriv_index,
        'priv_index': priv_index,
        'undesire_index': undesire_index,
        'desire_index': desire_index,
    }

def get_original_data(df_transformed, df_orig, ohe, scaler):
    df_ohe_int = df_transformed[:, :df_orig.select_dtypes(['float', 'integer']).shape[1]]
    df_ohe_int = scaler.inverse_transform(df_ohe_int)
    df_ohe_cats = df_transformed[:, df_orig.select_dtypes(['float', 'integer']).shape[1]:]
    df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
    df_int = pd.DataFrame(df_ohe_int, columns=df_orig.select_dtypes(['float', 'integer']).columns)
    df_cat = pd.DataFrame(df_ohe_cats, columns=df_orig.select_dtypes('object').columns)
    return pd.concat([df_int, df_cat], axis=1)

def prepare_data(
    data_df, S, Y, S_under, Y_desire,
    train_idx_relative, 
    eval_idx_relative, 
    test_idx_relative,
    batch_size=64,
):
    ans_dict = get_ohe_data(data_df, S, Y, S_under, Y_desire)
    df_transformed = ans_dict['final_array']
    input_dim = df_transformed.shape[1]

    train_data = df_transformed[train_idx_relative].copy()
    eval_data = df_transformed[eval_idx_relative].copy()
    test_data = df_transformed[test_idx_relative].copy()

    train_ds = TensorDataset(torch.from_numpy(train_data).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_ds = TensorDataset(torch.from_numpy(eval_data).float())
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=True)
    test_ds = TensorDataset(torch.from_numpy(test_data).float())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    
    solution = copy.deepcopy(ans_dict)
    solution['train_data'] = train_loader
    solution['eval_data'] = eval_loader
    solution['test_data'] = test_loader
    solution['train_idx'] = train_idx_relative
    solution['eval_idx'] = eval_idx_relative
    solution['test_idx'] = test_idx_relative
    solution['input_dim'] = input_dim
    solution['batch_size'] = batch_size
    solution['train_loader'] = train_loader
    solution['eval_loader'] = eval_loader
    solution['test_loader'] = test_loader
    return solution

class Generator(nn.Module):
    def __init__(self, input_dim, continuous_columns, discrete_columns):
        super(Generator, self).__init__()
        self._input_dim = input_dim
        self._discrete_columns = discrete_columns
        self._num_continuous_columns = len(continuous_columns)
        self.lin1 = nn.Linear(self._input_dim, self._input_dim)
        self.lin_numerical = nn.Linear(self._input_dim, self._num_continuous_columns)
        self.lin_cat = nn.ModuleDict()
        for key, value in self._discrete_columns.items():
            self.lin_cat[key] = nn.Linear(self._input_dim, value)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x_numerical = F.relu(self.lin_numerical(x))
        x_cat = []
        for key in self.lin_cat:
            x_cat.append(F.gumbel_softmax(self.lin_cat[key](x), tau=0.2))
        x_final = torch.cat((x_numerical, *x_cat), 1)
        return x_final

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self._input_dim = input_dim
        self.dense1 = nn.Linear(self._input_dim, self._input_dim)
        self.dense2 = nn.Linear(self._input_dim, self._input_dim)

    def forward(self, x):
        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))
        return x

class FairLossFunc(nn.Module):
    def __init__(self, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index):
        super(FairLossFunc, self).__init__()
        self._S_start_index = S_start_index
        self._Y_start_index = Y_start_index
        self._underpriv_index = underpriv_index
        self._priv_index = priv_index
        self._undesire_index = undesire_index
        self._desire_index = desire_index

    def forward(self, x, crit_fake_pred, lamda):
        G = x[:, self._S_start_index:self._S_start_index + 2]
        H = x[:, self._Y_start_index:self._Y_start_index + 2]
        disp = -1.0 * lamda * (torch.mean(G[:, self._underpriv_index] * H[:, self._desire_index]) / (
            x[:, self._S_start_index + self._underpriv_index].sum()) - torch.mean(
            G[:, self._priv_index] * H[:, self._desire_index]) / (
                                   x[:, self._S_start_index + self._priv_index].sum())) - 1.0 * torch.mean(
            crit_fake_pred)
        return disp

def get_gradient(crit, real, fake, epsilon):
    mixed_data = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_data)
    gradient = torch.autograd.grad(
        inputs=mixed_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    return crit_loss

def train(
    data_df, S, Y, S_under, Y_desire,
    train_idx_relative, 
    eval_idx_relative, 
    test_idx_relative,
    device,
    batch_size=64,
    epochs=500, 
    fair_epochs=10, 
    lamda=0.5,
    lr=0.0002,
):
    sol_dict = prepare_data(
        data_df, S, Y, S_under, Y_desire,
        train_idx_relative,
        eval_idx_relative,
        test_idx_relative,
        batch_size,
    )
    input_dim = sol_dict['input_dim']
    continuous_columns = sol_dict['continuous_columns_list']
    discrete_columns = sol_dict['discrete_columns_ordereddict']
    S_start_index = sol_dict['S_start_index']
    Y_start_index = sol_dict['Y_start_index']
    underpriv_index = sol_dict['underpriv_index']
    priv_index = sol_dict['priv_index']
    undesire_index = sol_dict['undesire_index']
    desire_index = sol_dict['desire_index']
    train_loader = sol_dict['train_loader']

    generator = Generator(
        input_dim, continuous_columns, discrete_columns,
    ).to(device)
    critic = Critic(input_dim).to(device)
    second_critic = FairLossFunc(
        S_start_index, Y_start_index, underpriv_index, 
        priv_index, undesire_index, desire_index,
    ).to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    gen_optimizer_fair = torch.optim.Adam(generator.parameters(), lr=lr / 2, betas=(0.5, 0.999))
    crit_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.999))

    critic_losses = []
    cur_step = 0
    
    for i in range(epochs):
        if i == epochs - 1:
            print(f'epoch {i + 1}/{epochs}')
        else:
            print(f'epoch {i + 1}/{epochs}', end='\r')
        for data in train_loader:
            data[0] = data[0].to(device)
            batch_size = data[0].shape[0]
            crit_repeat = 4
            mean_iteration_critic_loss = 0
            for k in range(crit_repeat):
                crit_optimizer.zero_grad()
                fake_noise = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake = generator(fake_noise)
                crit_fake_pred = critic(fake.detach())
                crit_real_pred = critic(data[0])
                epsilon = torch.rand(batch_size, input_dim, device=device, requires_grad=True)
                gradient = get_gradient(critic, data[0], fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda=10)
                mean_iteration_critic_loss += crit_loss.item() / crit_repeat
                crit_loss.backward(retain_graph=True)
                crit_optimizer.step()
            if cur_step > 50:
                critic_losses += [mean_iteration_critic_loss]
            if i + 1 <= (epochs - fair_epochs):
                gen_optimizer.zero_grad()
                fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake_2 = generator(fake_noise_2)
                crit_fake_pred = critic(fake_2)
                gen_loss = get_gen_loss(crit_fake_pred)
                gen_loss.backward()
                gen_optimizer.step()
            if i + 1 > (epochs - fair_epochs):
                gen_optimizer_fair.zero_grad()
                fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake_2 = generator(fake_noise_2)
                crit_fake_pred = critic(fake_2)
                gen_fair_loss = second_critic(fake_2, crit_fake_pred, lamda)
                gen_fair_loss.backward()
                gen_optimizer_fair.step()
            cur_step += 1
    
    return {
        'generator': generator,
        'critic': critic,
        'ohe': sol_dict['ohe'],
        'scaler': sol_dict['scaler'],
        'train_data': sol_dict['train_data'],
        'eval_data': sol_dict['eval_data'],
        'test_data': sol_dict['test_data'],
        'input_dim': input_dim,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--exp_name', type=str, default='check')
    
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError('config file is required')
    
    # configs
    exp_config = config['exp']
    data_config = config['data']
    fair_config = config['fair']
    train_config = config['train']
    sample_config = config['sample']
    eval_config = config['eval']
    
    seed = exp_config['seed']
    sensitive = fair_config['sensitive']
    underprevileged = fair_config['underprivileged']
    label = fair_config['label']
    desirable = fair_config['desirable']
    batch_size = train_config['batch_size']
    n_epochs = train_config['n_epochs']
    fair_epochs = train_config['fair_epochs']
    lr = train_config['lr']
    n_seeds = sample_config['n_seeds']

    # message
    print(json.dumps(config, indent=4))
    print('-' * 80)
    
    # experimental directory
    exp_dir = os.path.join(
        exp_config['home'], 
        data_config['name'],
        exp_config['method'],
        args.exp_name,
    )
    copy_file(
        os.path.join(exp_dir), 
        args.config,
    )
    
    # data
    dataset_dir = os.path.join(data_config['path'], data_config['name'])
    ckpt_dir = os.path.join(exp_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    with open(os.path.join(dataset_dir, 'desc.json'), 'r') as f:
        description = json.load(f)
    d_num_x = description['d_num_x']
    label_cols = [pd.read_csv(os.path.join(dataset_dir, 'y_train.csv'), index_col=0).columns.tolist()[0]]

    df_name = os.path.join(dataset_dir, 'd_all.csv')
    train_name = os.path.join(dataset_dir, 'd_train.csv')
    eval_name = os.path.join(dataset_dir, 'd_eval.csv')
    test_name = os.path.join(dataset_dir, 'd_test.csv')
    S = sensitive
    Y = label
    underprivileged_value = underprevileged
    desirable_value = desirable

    all_df = pd.read_csv(df_name, index_col=0)
    train_df = pd.read_csv(train_name, index_col=0)
    eval_df = pd.read_csv(eval_name, index_col=0)
    test_df = pd.read_csv(test_name, index_col=0)
    all_idx = all_df.index
    train_idx = train_df.index 
    eval_idx = eval_df.index
    test_idx = test_df.index
    size_of_fake_data = len(train_df)
    
    # reletive postion
    train_idx_relative = [i for i in range(len(all_idx)) if all_idx[i] in train_idx]
    eval_idx_relative = [i for i in range(len(all_idx)) if all_idx[i] in eval_idx]
    test_idx_relative = [i for i in range(len(all_idx)) if all_idx[i] in test_idx]
    
    S_under = underprivileged_value
    Y_desire = desirable_value
    all_df[S] = all_df[S].astype(object)
    all_df[Y] = all_df[Y].astype(object)
    
    device = torch.device(exp_config['device'])

    start_time = time.time()
    solution = train(
        all_df, S, Y, S_under, Y_desire,
        train_idx_relative,
        eval_idx_relative,
        test_idx_relative,
        device=device,
        batch_size=batch_size,
        epochs=n_epochs,
        fair_epochs=fair_epochs,
        lamda=0.5,
        lr=lr,
    )
    end_time = time.time()
    print(f'training time: {(end_time - start_time):.2f}s')
    
    generator = solution['generator']
    input_dim = solution['input_dim']
    ohe = solution['ohe']
    scaler = solution['scaler']
    
    cat_encoder = sio.load(os.path.join(dataset_dir, 'cat_encoder.skops'))
    label_encoder = sio.load(os.path.join(dataset_dir, 'label_encoder.skops'))

    # sampling with seeds
    start_time = time.time()
    for i in range(n_seeds):
        random_seed = seed + i
        torch.manual_seed(random_seed)
        fake_numpy_array = generator(torch.randn(size=(size_of_fake_data, input_dim), device=device)).cpu().detach().numpy()
        fake_df = get_original_data(fake_numpy_array, all_df, ohe, scaler)
        x_syn_num = fake_df.iloc[:, :d_num_x]
        x_syn_cat = fake_df.iloc[:, d_num_x: -1]
        y_syn = fake_df.iloc[:, -1]
        
        # transform categorical data
        x_cat_cols = x_syn_cat.columns
        x_syn_cat = cat_encoder.transform(x_syn_cat)
        x_syn_cat = pd.DataFrame(x_syn_cat, columns=x_cat_cols)
        x_syn = pd.concat([x_syn_num, x_syn_cat], axis=1)
        y_syn = label_encoder.transform(pd.DataFrame(y_syn))
        y_syn = pd.DataFrame(y_syn, columns=label_cols)

        synth_dir = os.path.join(exp_dir, f'synthesis/{random_seed}')
        if not os.path.exists(synth_dir):
            os.makedirs(synth_dir)

        x_syn.to_csv(os.path.join(synth_dir, 'x_syn.csv'))
        y_syn.to_csv(os.path.join(synth_dir, 'y_syn.csv'))
        print(f'seed: {random_seed}, x_syn: {x_syn.shape}, y_syn: {y_syn.shape}')
    end_time = time.time()
    print(f'sampling time: {(end_time - start_time):.2f}s')
    
    # evaluation
    x_eval = pd.read_csv(
        os.path.join(data_config['path'], data_config['name'], 'x_eval.csv'),
        index_col=0,
    )
    c_eval = pd.read_csv(
        os.path.join(data_config['path'], data_config['name'], 'y_eval.csv'),
        index_col=0,
    )
    y_eval = c_eval.iloc[:, 0]
    
    # evaluate classifiers trained on synthetic data
    metric = {}
    for clf_choice in eval_config['sk_clf_choice']:
        aucs = []
        for i in range(n_seeds):
            random_seed = seed + i
            synth_dir = os.path.join(exp_dir, f'synthesis/{random_seed}')
            
            # read synthetic data
            x_syn = pd.read_csv(os.path.join(synth_dir, 'x_syn.csv'), index_col=0)
            c_syn = pd.read_csv(os.path.join(synth_dir, 'y_syn.csv'), index_col=0)
            y_syn = c_syn.iloc[:, 0]
            
            # train classifier
            clf = default_sk_clf(clf_choice, random_seed)
            clf.fit(x_syn, y_syn)
            y_pred = clf.predict_proba(x_eval)[:, 1]
            aucs.append(roc_auc_score(y_eval, y_pred))
        metric[clf_choice] = (np.mean(aucs), np.std(aucs))
    with open(os.path.join(exp_dir, 'metric.json'), 'w') as f:
        json.dump(metric, f, indent=4)

if __name__ == '__main__':
    main()
