import torch 
import numpy as np 
import os 

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

        # get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards 
        
        assert len(shards) > 0, f"no shards for split {split}"
        print(f"found {len(shards)} shards for split {split}")

        self.reset()

    def reset(self): 
        # init at shard zero
        self.current_shard = 0 
        self.tokens = load_tokens(self.shards[self.current_shard])
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
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0 

        return x, y 