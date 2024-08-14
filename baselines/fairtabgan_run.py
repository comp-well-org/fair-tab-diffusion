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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import TensorDataset, DataLoader

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

from lib import load_config, copy_file, load_json
from src.evaluate.metrics import evaluate_syn_data, print_metric
from constant import DB_PATH, EXPS_PATH

warnings.filterwarnings('ignore')

FAIR_CONFIG = {
    'adult': {
        'sensitive': 'sex',
        'underprivileged': 'Female',
        'label': 'class',
        'desirable': '>50K',
    },
    'german': {
        'sensitive': 'Sex',
        'underprivileged': 'female',
        'label': 'Risk',
        'desirable': 'good',
    },
    'bank': {
        'sensitive': 'age-group',
        'underprivileged': 'young',
        'label': 'y',
        'desirable': 'no',
    },
    'compass': {
        'sensitive': 'sex',
        'underprivileged': '0',
        'label': 'is_recid',
        'desirable': '-1',
    },
    'law': {
        'sensitive': 'gender',
        'underprivileged': 'female',
        'label': 'ugpagt3',
        'desirable': 'TRUE',
    },
}

def get_ohe_data(df, ss, yy, s_under, y_desire, num_cols, cat_cols):
    df_int = df[num_cols].values
    continuous_columns_list = num_cols

    scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
    df_int = scaler.fit_transform(df_int)
    
    df_cat = df[cat_cols].values
    df_cat_names = cat_cols
    
    numerical_array = df_int
    ohe = OneHotEncoder()
    ohe_array = ohe.fit_transform(df_cat)

    cat_lens = [i.shape[0] for i in ohe.categories_]
    discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))

    s_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(ss)])
    y_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(yy)])

    if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(ss)][0] == s_under:
        underpriv_index = 0
        priv_index = 1
    else:
        underpriv_index = 1
        priv_index = 0
    if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(yy)][0] == y_desire:
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
        's_start_index': s_start_index,
        'y_start_index': y_start_index,
        'underpriv_index': underpriv_index,
        'priv_index': priv_index,
        'undesire_index': undesire_index,
        'desire_index': desire_index,
    }

def get_original_data(df_transformed, d_num_x, num_cols, cat_cols, ohe, scaler):
    df_ohe_int = df_transformed[:, :d_num_x]
    df_ohe_int = scaler.inverse_transform(df_ohe_int)
    df_ohe_cats = df_transformed[:, d_num_x:]
    df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
    
    df_int = pd.DataFrame(df_ohe_int, columns=num_cols)
    
    df_cat = pd.DataFrame(df_ohe_cats, columns=cat_cols)
    
    data_df = pd.concat([df_int, df_cat], axis=1)
    return data_df

def prepare_data(
    data_df, 
    num_cols,
    cat_cols,
    ss, yy, s_under, y_desire, 
    train_idx_relative, 
    eval_idx_relative, 
    test_idx_relative,
    batch_size=64,
):
    ans_dict = get_ohe_data(data_df, ss, yy, s_under, y_desire, num_cols, cat_cols)
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
            if '.' in key:
                key = key.replace('.', '_')
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
    def __init__(self, s_start_index, y_start_index, underpriv_index, priv_index, undesire_index, desire_index):
        super(FairLossFunc, self).__init__()
        self._s_start_index = s_start_index
        self._y_start_index = y_start_index
        self._underpriv_index = underpriv_index
        self._priv_index = priv_index
        self._undesire_index = undesire_index
        self._desire_index = desire_index

    def forward(self, x, crit_fake_pred, lamda):
        G = x[:, self._s_start_index:self._s_start_index + 2]
        H = x[:, self._y_start_index:self._y_start_index + 2]
        disp = -1.0 * lamda * (torch.mean(G[:, self._underpriv_index] * H[:, self._desire_index]) / (
            x[:, self._s_start_index + self._underpriv_index].sum()) - torch.mean(
            G[:, self._priv_index] * H[:, self._desire_index]) / (
                                   x[:, self._s_start_index + self._priv_index].sum())) - 1.0 * torch.mean(
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
    data_df, 
    num_cols,
    cat_cols,
    ss, yy, s_under, y_desire,
    train_idx_relative, 
    eval_idx_relative, 
    test_idx_relative,
    device,
    batch_size=64,
    epochs=500, 
    fair_epochs=10, 
    lamda=0.5,
    lr=0.0002,
    exp_dir='.',
):
    sol_dict = prepare_data(
        data_df, 
        num_cols,
        cat_cols,
        ss, yy, s_under, y_desire,
        train_idx_relative,
        eval_idx_relative,
        test_idx_relative,
        batch_size,
    )
    input_dim = sol_dict['input_dim']
    continuous_columns = sol_dict['continuous_columns_list']
    discrete_columns = sol_dict['discrete_columns_ordereddict']
    s_start_index = sol_dict['s_start_index']
    y_start_index = sol_dict['y_start_index']
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
        s_start_index, y_start_index, underpriv_index, 
        priv_index, undesire_index, desire_index,
    ).to(device)
    
    num_params_generator = sum(p.numel() for p in generator.parameters())
    num_params_critic = sum(p.numel() for p in critic.parameters())
    num_params_second_critic = sum(p.numel() for p in second_critic.parameters())
    num_params = num_params_generator + num_params_critic + num_params_second_critic
    with open(os.path.join(exp_dir, 'params.txt'), 'w') as f:
        f.write(f'number of parameters: {num_params}')

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
            for _ in range(crit_repeat):
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
    parser.add_argument('--train', action='store_true', help='training', default=True)
    parser.add_argument('--sample', action='store_true', help='sampling', default=True)
    parser.add_argument('--eval', action='store_true', help='evaluation', default=True)
    
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError('config file is required')
    
    # configs
    exp_config = config['exp']
    data_config = config['data']
    train_config = config['train']
    sample_config = config['sample']
    eval_config = config['eval']
    fair_config = FAIR_CONFIG[data_config['name']]
    
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
    print(f'config file: {args.config}')
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
    data_desc = load_json(os.path.join(dataset_dir, 'desc.json'))
    d_types = data_desc['d_types']
    d_num_x = data_desc['d_num_x']
    num_col_names = data_desc['num_col_names']
    cat_col_names = data_desc['cat_col_names']
    label_cols = [data_desc['label_col_name']]
    cat_label_cols = cat_col_names + label_cols

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

    if args.train:
        # train
        start_time = time.time()
        solution = train(
            all_df, 
            num_col_names,
            cat_label_cols,
            S, Y, S_under, Y_desire,
            train_idx_relative,
            eval_idx_relative,
            test_idx_relative,
            device=device,
            batch_size=batch_size,
            epochs=n_epochs,
            fair_epochs=fair_epochs,
            lamda=0.5,
            lr=lr,
            exp_dir=exp_dir,
        )
        end_time = time.time()
        with open(os.path.join(exp_dir, 'time.txt'), 'w') as f:
            time_msg = f'training time: {end_time - start_time:.2f} seconds with {n_epochs} epochs'
            f.write(time_msg)
        print()
            
    if args.sample:
        # sampling
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
            fake_df = get_original_data(fake_numpy_array, d_num_x, num_col_names, cat_label_cols, ohe, scaler)  
            
            x_syn_num = fake_df.iloc[:, :d_num_x]
            x_syn_cat = fake_df.iloc[:, d_num_x: -1]
            y_syn = fake_df.iloc[:, -1]
            y_syn = y_syn.astype(str)
            
            # FIXME: this is only for the law datast, please fix this
            if list(y_syn.unique()) == ['False', 'True']:
                # capitalize
                y_syn = y_syn.replace({'True': 'TRUE', 'False': 'FALSE'})
            y_syn = pd.DataFrame(y_syn)
            
            # transform categorical data
            x_cat_cols = x_syn_cat.columns
            x_syn_cat = x_syn_cat.astype(str)
            x_syn_cat = cat_encoder.transform(x_syn_cat)
            x_syn_cat = pd.DataFrame(x_syn_cat, columns=x_cat_cols)
            x_syn = pd.concat([x_syn_num, x_syn_cat], axis=1)
            y_syn = label_encoder.transform(y_syn)
            y_syn = pd.DataFrame(y_syn, columns=label_cols)

            synth_dir = os.path.join(exp_dir, f'synthesis/{random_seed}')
            if not os.path.exists(synth_dir):
                os.makedirs(synth_dir)

            x_syn.to_csv(os.path.join(synth_dir, 'x_syn.csv'))
            y_syn.to_csv(os.path.join(synth_dir, 'y_syn.csv'))
            print(f'seed: {random_seed}, x_syn: {x_syn.shape}, y_syn: {y_syn.shape}')
        end_time = time.time()
        with open(os.path.join(exp_dir, 'time.txt'), 'a') as f:
            time_msg = f'\nsampling time: {end_time - start_time:.2f} seconds with {n_seeds} seeds'
            f.write(time_msg)
        print()
    
    if args.eval:
        # evaluate classifiers trained on synthetic data
        synth_dir_list = []
        for i in range(n_seeds):
            synth_dir = os.path.join(exp_dir, f'synthesis/{seed + i}')
            if os.path.exists(synth_dir):
                synth_dir_list.append(synth_dir)

        sst_col_names = data_desc['sst_col_names']
        metric = evaluate_syn_data(
            data_dir=os.path.join(data_config['path'], data_config['name']),
            exp_dir=exp_dir,
            synth_dir_list=synth_dir_list,
            sk_clf_lst=eval_config['sk_clf_choice'],
            sens_cols=sst_col_names,
        )

        with open(os.path.join(exp_dir, 'metric.json'), 'w') as f:
            json.dump(metric, f, indent=4)

        # print metric
        print_metric(metric)

if __name__ == '__main__':
    main()
