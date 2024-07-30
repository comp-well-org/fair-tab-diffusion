import os
import json
import torch
import random
import logging
import warnings
import typing as tp
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from dataclasses import dataclass
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings('ignore')

################################################################################
# data
class GReaTDataset(Dataset):
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        row = self._data.fast_slice(key, 1)
        shuffle_idx = list(range(row.num_columns))
        random.shuffle(shuffle_idx)
        shuffled_text = ', '.join(
            ['%s is %s' % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip()) for i in shuffle_idx],
        )
        tokenized_text = self.tokenizer(shuffled_text)
        return tokenized_text
        
    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)

@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch['labels'] = batch['input_ids'].clone()
        return batch

def _pad(x, length: int, pad_value=50256):
    return [pad_value] * (length - len(x)) + x

def _pad_tokens(tokens):
    max_length = len(max(tokens, key=len))
    tokens = [_pad(t, max_length) for t in tokens]
    return tokens

################################################################################
# utils
class GReaTStart:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_start_tokens(self, n_samples: int) -> tp.List[tp.List[int]]:
        raise NotImplementedError('this has to be overwritten but the subclasses')

class CategoricalStart(GReaTStart):
    def __init__(self, tokenizer, start_col: str, start_col_dist: dict):
        super().__init__(tokenizer)
        assert isinstance(start_col, str), ''
        assert isinstance(start_col_dist, dict), ''
        self.start_col = start_col
        self.population = list(start_col_dist.keys())
        self.weights = list(start_col_dist.values())

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.population, self.weights, k=n_samples)
        start_text = [self.start_col + ' is ' + str(s) + ',' for s in start_words]
        start_tokens = _pad_tokens(self.tokenizer(start_text)['input_ids'])
        return start_tokens

class ContinuousStart(GReaTStart):
    def __init__(self, tokenizer, start_col: str, start_col_dist: tp.List[float], noise: float = .01, decimal_places: int = 5):
        super().__init__(tokenizer)

        assert isinstance(start_col, str), ''
        assert isinstance(start_col_dist, list), ''

        self.start_col = start_col
        self.start_col_dist = start_col_dist
        self.noise = noise
        self.decimal_places = decimal_places

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.start_col_dist, k=n_samples)
        start_text = [self.start_col + ' is ' + format(s, f'.{self.decimal_places}f') + ',' for s in start_words]
        start_tokens = _pad_tokens(self.tokenizer(start_text)['input_ids'])
        return start_tokens

class RandomStart(GReaTStart):
    def __init__(self, tokenizer, all_columns: tp.List[str]):
        super().__init__(tokenizer)
        self.all_columns = all_columns

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.all_columns, k=n_samples)
        start_text = [s + ' is ' for s in start_words]
        start_tokens = _pad_tokens(self.tokenizer(start_text)['input_ids'])
        return start_tokens
    
def _array_to_dataframe(data: tp.Union[pd.DataFrame, np.ndarray], columns=None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data

    assert isinstance(data, np.ndarray), 'input needs to be a pandas dataframe or a numpy array!'
    assert columns, 'To convert the data into a pandas dataframe, a list of column names has to be given!'
    assert len(columns) == len(data[0]), '%d column names are given, but array has %d columns!' % (len(columns), len(data[0]))

    return pd.DataFrame(data=data, columns=columns)

def _get_column_distribution(df: pd.DataFrame, col: str) -> tp.Union[list, dict]:
    if df[col].dtype == 'float':
        col_dist = df[col].to_list()
    else:
        col_dist = df[col].value_counts(1).to_dict()
    return col_dist

def _convert_tokens_to_text(tokens: tp.List[torch.Tensor], tokenizer: AutoTokenizer) -> tp.List[str]:
    # convert tokens to text
    text_data = [tokenizer.decode(t) for t in tokens]

    # clean text
    text_data = [d.replace('<|endoftext|>', '') for d in text_data]
    text_data = [d.replace('\n', ' ') for d in text_data]
    text_data = [d.replace('\r', '') for d in text_data]

    return text_data

def _convert_text_to_tabular_data(text: tp.List[str], df_gen: pd.DataFrame) -> pd.DataFrame:
    columns = df_gen.columns.to_list()
        
    # convert text to tabular data
    for t in text:
        features = t.split(',')
        td = dict.fromkeys(columns)
        
        # transform all features back to tabular data
        for f in features:
            values = f.strip().split(' is ')
            if values[0] in columns and not td[values[0]]:
                try:
                    td[values[0]] = [values[1]]
                except IndexError:
                    pass
        df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)
    return df_gen

def _seed_worker(_):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)

class GReaTTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError('trainer: training requires a train_dataset')

        data_collator = self.data_collator
        train_dataset = self.train_dataset
        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=_seed_worker,
        )

################################################################################
# model
class GReaT:
    def __init__(
        self, llm: str, experiment_dir: str = 'trainer_great', epochs: int = 100,
        batch_size: int = 8, efficient_finetuning: str = '', **train_kwargs,
    ):
        # load model and tokenizer from hugging face
        self.efficient_finetuning = efficient_finetuning
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.llm)

        if self.efficient_finetuning == 'lora':
            lora_config = LoraConfig(
                r=16,  # only training 0.16% of the parameters of the model
                lora_alpha=32,
                target_modules=['c_attn'],  # this is specific for gpt2 model, to be adapted
                lora_dropout=0.05,
                bias='none',
                task_type=TaskType.CAUSAL_LM,  # this is specific for gpt2 model, to be adapted
            )
            # prepare int-8 model for training
            self.model = prepare_model_for_kbit_training(self.model)
            # add lora adaptor
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # set the training hyperparameters
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_hyperparameters = train_kwargs

        # needed for the sampling process
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None

    def fit(
        self, data: tp.Union[pd.DataFrame, np.ndarray], column_names: tp.Optional[tp.List[str]] = None,
        conditional_col: tp.Optional[str] = None, resume_from_checkpoint: tp.Union[bool, str] = False,
    ) -> GReaTTrainer:
        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        # convert dataframe into hugging face dataset object
        logging.info('convert data into hugging face dataset object...')
        great_ds = GReaTDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer)

        # set training hyperparameters
        logging.info('create `GReaT` trainer...')
        training_args = TrainingArguments(
            self.experiment_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            **self.train_hyperparameters,
        )
        great_trainer = GReaTTrainer(
            self.model, training_args, train_dataset=great_ds, tokenizer=self.tokenizer,
            data_collator=GReaTDataCollator(self.tokenizer),
        )

        # start training
        logging.info('start training...')
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return great_trainer

    def sample(
        self, n_samples: int, start_col: tp.Optional[str] = '', start_col_dist: tp.Optional[tp.Union[dict, list]] = None, 
        temperature: float = 0.7, k: int = 100, max_length: int = 500, device: str = 'cuda',
    ) -> pd.DataFrame:
        great_start = self._get_start_sampler(start_col, start_col_dist)

        # move model to device
        self.model.to(device)

        # init empty DataFrame for the generated samples
        df_gen = pd.DataFrame(columns=self.columns)

        # start generation process
        with tqdm(total=n_samples) as pbar:
            already_generated = 0
            while n_samples > df_gen.shape[0]:
                start_tokens = great_start.get_start_tokens(k)
                start_tokens = torch.tensor(start_tokens).to(device)

                # generate tokens
                tokens = self.model.generate(input_ids=start_tokens, max_length=max_length, do_sample=True, temperature=temperature, pad_token_id=50256)

                # convert tokens back to tabular data
                text_data = _convert_tokens_to_text(tokens, self.tokenizer)
                df_gen = _convert_text_to_tabular_data(text_data, df_gen)

                # print(len(text_data[0]))
                # print(text_data[0])
                # print(df_gen.iloc[0])

                # remove rows with flawed numerical values
                for i_num_cols in self.num_cols:
                    df_gen = df_gen[pd.to_numeric(df_gen[i_num_cols], errors='coerce').notnull()]

                df_gen[self.num_cols] = df_gen[self.num_cols].astype(float)

                # remove rows with missing values
                df_gen = df_gen.drop(df_gen[df_gen.isna().any(axis=1)].index)
                # update process bar
                pbar.update(df_gen.shape[0] - already_generated)
                already_generated = df_gen.shape[0]
        df_gen = df_gen.reset_index(drop=True)
        return df_gen.head(n_samples)

    def great_sample(self, starting_prompts: tp.Union[str, list[str]], temperature: float = 0.7, max_length: int = 100, device: str = 'cuda') -> pd.DataFrame:
        self.model.to(device)
        starting_prompts = [starting_prompts] if isinstance(
            starting_prompts, str) else starting_prompts
        generated_data = []

        # generate a sample for each starting point
        for prompt in tqdm(starting_prompts):
            start_token = torch.tensor(self.tokenizer(prompt)['input_ids']).to(device)

            # generate tokens
            gen = self.model.generate(input_ids=torch.unsqueeze(start_token, 0), max_length=max_length, do_sample=True, temperature=temperature, pad_token_id=50256)
            generated_data.append(torch.squeeze(gen))

        # convert text back to tabular data
        decoded_data = _convert_tokens_to_text(generated_data, self.tokenizer)
        df_gen = _convert_text_to_tabular_data(
            decoded_data, pd.DataFrame(columns=self.columns))

        return df_gen

    def save(self, path: str):
        # make directory
        if os.path.isdir(path):
            print(f'directory {path} already exists and is overwritten now')
        else:
            os.mkdir(path)

        # save attributes
        with open(path + '/config.json', 'w') as f:
            attributes = self.__dict__.copy()
            attributes.pop('tokenizer')
            attributes.pop('model')

            # ndarray is not json serializable and therefore has to be converted into a list
            if isinstance(attributes['conditional_col_dist'], np.ndarray):
                attributes['conditional_col_dist'] = list(
                    attributes['conditional_col_dist'])

            json.dump(attributes, f)

        # save model weights
        torch.save(self.model.state_dict(), path + '/model.pt')

    def load_finetuned_model(self, path: str):
        self.model.load_state_dict(torch.load(path))

    @classmethod
    def load_from_dir(cls, path: str):
        assert os.path.isdir(path), f'directory {path} does not exist'

        # load attributes
        with open(path + '/config.json', 'r') as f:
            attributes = json.load(f)

        # create new be_great model instance
        great = cls(attributes['llm'])

        # set all attributes
        for k, v in attributes.items():
            setattr(great, k, v)

        # load model weights
        great.model.load_state_dict(torch.load(path + '/model.pt', map_location='cpu'))

        return great

    def _update_column_information(self, df: pd.DataFrame):
        # update the column names (and numerical columns for some sanity checks after sampling)
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    def _update_conditional_information(self, df: pd.DataFrame, conditional_col: tp.Optional[str] = None):
        assert conditional_col is None or isinstance(conditional_col, str), (f'the column name has to be a string and not {type(conditional_col)}')
        assert conditional_col is None or conditional_col in df.columns, (f'the column name {conditional_col} is not in the feature names of the given dataset')

        # take the distribution of the conditional column for a starting point in the generation process
        self.conditional_col = conditional_col if conditional_col else df.columns[-1]
        self.conditional_col_dist = _get_column_distribution(df, self.conditional_col)

    def _get_start_sampler(
        self, start_col: tp.Optional[str],
        start_col_dist: tp.Optional[tp.Union[tp.Dict, tp.List]],
    ) -> GReaTStart:
        if start_col and start_col_dist is None:
            raise ValueError(f'start column {start_col} was given, but no corresponding distribution')
        if start_col_dist is not None and not start_col:
            raise ValueError(f'start column distribution {start_col} was given, the column name is missing')

        assert start_col is None or isinstance(start_col, str), f'the column name has to be a string and not {type(start_col)}'
        assert start_col_dist is None or isinstance(start_col_dist, dict) or isinstance(start_col_dist, list), (
            f'the distribution of the start column on has to be a list or a dict and not {type(start_col_dist)}'
        )
        start_col = start_col if start_col else self.conditional_col
        start_col_dist = start_col_dist if start_col_dist else self.conditional_col_dist

        if isinstance(start_col_dist, dict):
            return CategoricalStart(self.tokenizer, start_col, start_col_dist)
        elif isinstance(start_col_dist, list):
            return ContinuousStart(self.tokenizer, start_col, start_col_dist)
        else:
            return RandomStart(self.tokenizer, self.columns)

################################################################################
# main
def main():
    # global variables
    device = 'cuda:1'
    
    # TODO: configs
    dataname = 'adult'
    n_epochs = 1
    batch_size = 8
    n_samples = 2000
    
    # data
    train_df = pd.read_csv(f'/rdf/db/public-tabular-datasets/{dataname}/d_train.csv', index_col=0)
    
    # model
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    great = GReaT(
        'distilgpt2',                         
        epochs=n_epochs,          
        experiment_dir=ckpt_dir,
        batch_size=batch_size,
        save_strategy='no',
        logging_strategy='no',
    )
    
    # training
    great.fit(train_df)
    great.save(ckpt_dir)
    
    # sampling
    great.load_finetuned_model(f'{ckpt_dir}/model.pt')
    
    df = _array_to_dataframe(train_df, columns=None)
    great._update_column_information(df)
    great._update_conditional_information(df, conditional_col=None)    
    samples = great.sample(n_samples, k=100, device=device)
    print(samples.head(3))

if __name__ == '__main__':
    main()
