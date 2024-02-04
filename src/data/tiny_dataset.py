import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import Dataset
from core.model_config import ModelConfig
from .tiny_tokenizer import BPETokenizer

class TinyTextDataset(Dataset):
    def __init__(self, config: ModelConfig, text):

        self.config = config
        self.tokenizer = BPETokenizer()
        ids, u_ids = self.tokenizer(text)
        self.vocab_size = u_ids.size()[0]
        #self_vocab_list = u_ids
        self.data = ids

    def get_vocab_size(self):
        return self.vocab_size
    def get_vocab_list(self):
        return self.vocab_list

    def get_block_size(self):
        return self.config.n_positions

    def __len__(self):
        return len(self.data) - self.config.n_positions

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        ids = self.data[idx:idx + self.config.n_positions + 1]
        # return as tensors
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y



"""

from datasets import load_dataset
import torch
from tokenizer import BPETokenizer, Encoder
# List available datasets
dataset = load_dataset("tiny_shakespeare")
text = dataset["train"][0]["text"]

#encoder = Encoder()
#rs= encoder.encode_and_show_work(text)

#unique_values, counts = torch.unique(rs['tokens'], return_counts=True)

#print(rs['tokens'])

tokenizer = BPETokenizer()
r, u, c = tokenizer(text)
print(r.size())
print(u.size())

#for i in range(u.size(0)):
#    element = r[i, j].item()
#    element = u[i:i+1]
    # Do something with the element
#    print(element)
#    txt=tokenizer.decode(element)
#    print(txt)
    

#reverse
#txt=tokenizer.decode(r)

#print(txt)

"""