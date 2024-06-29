from dataclasses import dataclass 

@dataclass 
class GPT2Config: 
    # sequence len 
    block_size: int = 1024 
    # num. of tokens, 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    vocab_size: int = 50257 
    n_layer: int = 12 
    n_head: int = 12
    n_embd: int = 768 

@dataclass 
class TrainConfig: 
    batch_size: int = 8 
    sequence_len: int = 512 
    max_lr: float = 6e-4 
    min_lr: float = 6e-5 
    warmup_steps: float = 715 
    # if data is 10B tokens and batch size 0.5M tokens, 
    # 19,073 steps is ~1 epoch
    max_steps: float = 19073 
    val_loss_steps: int = 20 