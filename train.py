import torch 
import math 
import os 
import time 

from config import TrainConfig, GPT2Config 
from dataset import DataLoaderLite 
from model import GPT2 

def choose_device(): 
    device = "cpu"
    if torch.cuda.is_available(): 
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
        device = "mps"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"using device: {device} with type {device_type}")
    return device, device_type

def get_learning_rate(config, iteration): 
    max_lr = config.max_lr 
    min_lr = config.min_lr 
    warmup_steps = config.warmup_steps 
    max_steps = config.max_steps 

    # 1) linear warmup for warmup_iters steps
    if iteration < warmup_steps: 
        return max_lr * (iteration + 1) / warmup_steps 
    
    # 2) if it > lr_decay_iters, return min learning rate
    if iteration > max_steps: 
        return min_lr 
    
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iteration - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1 
    # coeff starts at 1 and goes to 0
    coeff = .5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train(): 
    device, device_type = choose_device()
    # from ghost in the shell 
    torch.manual_seed(2501)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(2501)

    # 2**19, ~0.5M, in number of tokens
    total_batch_size = 524288

    train_config = TrainConfig()
    batch_size = train_config.batch_size 
    sequence_len = train_config.sequence_len 
    assert total_batch_size % (batch_size * sequence_len) == 0
    grad_accum_steps = total_batch_size // (batch_size * sequence_len)

    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(batch_size=batch_size, sequence_len=sequence_len, split="train")
    val_loader = DataLoaderLite(batch_size=batch_size, sequence_len=sequence_len, split="val")

    torch.set_float32_matmul_precision('high')

    gpt2_config = GPT2Config(vocab_size=50304)
    model = GPT2(gpt2_config)
    model.to(device)
    model = torch.compile(model)

    optimizer = model.configure_optimizers(weight_decay=.1, learning_rate=train_config.max_lr, device_type=device_type)

    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    # open for writing to clear the file
    with open(log_file, "w") as f: 
        pass 

    for step in range(train_config.max_steps): 
        t0 = time.time()

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = .0 

        # accumulate the gradients 
        for micro_step in range(grad_accum_steps): 
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16): 
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss /= grad_accum_steps 
            loss_accum += loss.detach()
            loss.backward()

        # update the parameters 
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        learning_rate = get_learning_rate(train_config, step)
        for param_group in optimizer.param_groups: 
            param_group["lr"] = learning_rate 
        optimizer.step()
        if device_type == "cuda": 
            # wait for the GPU to finish work
            torch.cuda.synchronize()
        t1 = time.time()

        dt = t1 - t0 
        tokens_processed = train_loader.batch_size * train_loader.sequence_len * grad_accum_steps 
        tokens_per_sec = tokens_processed / dt 

        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {learning_rate:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

        # once in a while evaluate our validation loss and save model 
        last_step = (step == train_config.max_steps - 1)
        if step % 250 == 0 or last_step: 
            model.eval()
            val_loader.reset()
            with torch.no_grad(): 
                val_loss_accum = .0 
                for _ in range(train_config.val_loss_steps): 
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16): 
                        logits, loss = model(x, y)
                    loss = loss / train_config.val_loss_steps 
                    val_loss_accum += loss.detach()

            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f: 
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step): 
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": model.state_dict(), 
                    "config": model.config, 
                    "step": step, 
                    "val_loss": val_loss_accum.item(), 
                }
                torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__": 
    train()