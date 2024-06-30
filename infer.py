import os 
import torch 
from torch.nn import functional as F 
import tiktoken

from config import TrainConfig, GPT2Config 
from utils import choose_device 
from model import GPT2 

def infer(): 
    train_config = TrainConfig()
    log_dir = train_config.log_dir 
    files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.pt')]
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    assert len(files) > 0, f"No saved model"

    ckpt_path = files[0]
    print(f"loading model from {ckpt_path}")
    device, device_type = choose_device()
    checkpoint = torch.load(ckpt_path, map_location=device)

    gpt2_config = GPT2Config()
    model = GPT2(gpt2_config)
    # model = GPT2.from_pretrained("gpt2")
    model.to(device)

    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

    model.eval()
    num_return_sequences = 4
    max_length = 32

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hi Rick, ")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(2501)

    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad(): 
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # xgen size is B, T, vocab_size
                logits, loss = model(xgen)
            # take the logits at the last position
            # logits size is B, vocab_size
            logits = logits[:, -1, :]
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            # ix size is (B, 1)
            # multinomial returns a tensor where each row 
            # contains num_samples indices sampled from the multinomial
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            # gather the corresponding 
            # xcol size is (B, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences): 
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"sample {i}: {decoded}")
        

if __name__ == "__main__": 
    infer()