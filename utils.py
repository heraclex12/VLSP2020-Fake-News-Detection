import numpy as np
import torch
import os


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(model, tokenizer, checkpoint_path, epoch='best'):
    torch.save(model.state_dict(), os.path.join(checkpoint_path, f'model_{epoch}.bin'))
    # save configurations
    model.config.to_json_file(os.path.join(checkpoint_path, 'config.json'))
    # save exact vocabulary utilized
    tokenizer.save_vocabulary(checkpoint_path)
