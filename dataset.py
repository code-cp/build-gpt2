import torch 
import numpy as np 
import os 
import tiktoken

def process_text(filename): 
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines1 = [line.strip().replace('MORTY: ', '') for line in lines if line.startswith('MORTY:')]
    lines2 = [line.strip().replace('Morty: ', '') for line in lines if line.startswith('Morty:')]
    data = "".join(lines1 + lines2)

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(data)
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    print(f"data has {len(tokens_np_uint16):,} tokens")

    result_filename = os.path.splitext(filename)[0] + ".npy"
    print(f"save tokens to {result_filename}")
    np.save(result_filename, tokens_np_uint16)

def load_tokens(filename): 
    npt = np.load(filename)

    # Earlier version of PyTorch may have difficulty converting from uint16 to long. 
    # we added npt = npt.astype(np.int32) 
    # to use numpy to convert uint16 to int32 
    # before converting to torch tensor and then converting to long
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)

    return ptt 

class DataLoaderLite: 
    def __init__(self, data_root, batch_size, sequence_len, split): 
        self.batch_size = batch_size 
        self.sequence_len = sequence_len 
        assert split in ["train", "val"]
        
        data_dir = os.path.join(data_root, split)
        filename = data_dir + ".npy"
        if not os.path.exists(filename):
            process_text(data_dir + ".txt")

        self.tokens = load_tokens(filename)
        self.reset()

    def reset(self):
        self.current_position = 0

    def next_batch(self): 
        batch_size, sequence_len = self.batch_size, self.sequence_len 
        buf = self.tokens[
            self.current_position : 
            self.current_position + batch_size * sequence_len + 1 
            ]
        
        # inputs 
        x = (buf[:-1]).view(batch_size, sequence_len)
        # targets 
        y = (buf[1:]).view(batch_size, sequence_len)

        # advance the position in the tensor
        self.current_position += batch_size * sequence_len 
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (batch_size * sequence_len + 1) > len(self.tokens): 
            self.current_position = 0 

        return x, y 
