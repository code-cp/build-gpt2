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
    batch_size: int = 1
    total_batch_size: int = 4096
    sequence_len: int = 1024 
    max_lr: float = 5e-5
    min_lr: float = 5e-6 
    warmup_steps: float = 5 
    # max_steps: float = 100
    max_steps: float = 50
    val_loss_steps: int = 5 
    log_dir: str = "log"
    save_model_interval: int = 20
    get_eval_interval: int = 10 