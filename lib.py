import os
import json
import tomli
import shutil
import tomli_w
import skops.io as sio

def load_config(path) -> dict:
    with open(path, 'rb') as f:
        return tomli.load(f)
    
def copy_file(exp_dir, file_path) -> None:
    temp = os.path.join(exp_dir, os.path.basename(file_path))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        shutil.copy(file_path, exp_dir, follow_symlinks=True)
    else:
        if os.path.exists(temp):
            temp_file = os.path.join(exp_dir, os.path.basename(file_path) + '_temp')
            shutil.copy(file_path, temp_file, follow_symlinks=True)
            os.remove(temp)
            shutil.move(temp_file, temp)
        else:
            shutil.copy(file_path, exp_dir, follow_symlinks=True)

def write_config(config, config_file_path) -> None:
    with open(config_file_path, 'wb') as f:
        tomli_w.dump(config, f)

def load_json(js_path) -> dict:
    with open(js_path, 'r') as f:
        return json.load(f)

def write_json(js, js_path) -> None:
    with open(js_path, 'w') as f:
        json.dump(js, f, indent=4)

def load_encoder(encoder_path) -> object:
    return sio.load(encoder_path)
