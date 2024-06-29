import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import inspect

class CausalSelfAttention(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        
        assert config.n_embd % config.n_head == 0 

        # project input to key, query, value 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT2_SCALE_INIT = 1 

        # regularization 
        self.n_head = config.n_head 
        self.n_embd = config.n_embd 

    def forward(self, x): 
        batch_size, sequence_len, embd_dim = x.size() 

        # map input to q, k, v 
        qkv = self.c_attn(x) 
        q, k, v = qkv.split(self.n_embd, dim=2)

        # move head to the batch dim 
        # nh is number of heads 
        # hs is head size 
        # number of channels = nh * hs = embd_dim
        # in GPT2 124M, n_head = 12, hs = 64, so nh * hs = 768 channels 
        
        # q size is B x T x nh x hs
        # which becomes B x nh x T x hs after transpose 
        q = q.view(batch_size, sequence_len, self.n_head, embd_dim // self.n_head).transpose(1, 2)
        k = k.view(batch_size, sequence_len, self.n_head, embd_dim // self.n_head).transpose(1, 2)
        v = v.view(batch_size, sequence_len, self.n_head, embd_dim // self.n_head).transpose(1, 2)

        # flash attention 
        # y size is B x nh x T x hs
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # re-assemble all head outputs 
        # y size B x T x nh x hs after transpose 
        # then becomes B x T x C after view 
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_len, embd_dim)
        # output projection 
        # y size is still B x T x C
        y = self.c_proj(y)

        return y 
    
class MLP(nn.Module): 

    def __init__(self, config): 
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.GPT2_SCALE_INIT = 1 

    def forward(self, x): 
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x 
    
class Block(nn.Module): 

    def __init__(self, config): 
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x): 
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 
    
class GPT2(nn.Module): 

    def __init__(self, config): 
        super().__init__()
        self.config = config 

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd), 
                wpe = nn.Embedding(config.block_size, config.n_embd), 
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # share the embedding 
        self.transformer.wte.weight = self.lm_head.weight 

        self.apply(self._init_weights)

    def _init_weights(self, module): 
        default_std = .02 
        if isinstance(module, nn.Linear): 
            std = default_std 
            if hasattr(module, 'GPT2_SCALE_INIT'): 
                std *= (2 * self.config.n_layer) ** -0.5 
            torch.nn.init.normal_(module.weight, mean=.0, std=std)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=.0, std=default_std)

    def forward(self, idx, targets=None): 
        # idx size is batch size x sequence len 
        batch_size, sequence_len = idx.size()
        assert sequence_len <= self.config.block_size, f"Cannot process sequence len {sequence_len}, since block size is only {self.config.block_size}"

        # pos size is sequence len 
        pos = torch.arange(0, sequence_len, dtype=torch.long, device=idx.device)
        # pos embd size is sequence len x n_embd 
        pos_emb = self.transformer.wpe(pos)
        # token embd size is batch size x sequence len x n_embd 
        tok_emb = self.transformer.wte(idx)
        # broadcast in batch dim  
        x = tok_emb + pos_emb 

        for block in self.transformer.h: 
            x = block(x)
        
        x = self.transformer.ln_f(x)
        # logits size is batch size x sequence len x vocab_size  
        logits = self.lm_head(x)

        loss = None 
        if targets is not None: 
            # view(-1) flattens the vector
            # it has to construct a new view with only 1 dimension and infer the dimension
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss 
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type): 
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay}, 
            {'params': nodecay_params, 'weight_decay': .0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params) 
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters 
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(.9, .95), eps=1e-8, fused=use_fused)

        return optimizer 