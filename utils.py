import torch 

def choose_device(): 
    device = "cpu"
    if torch.cuda.is_available(): 
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
        device = "mps"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"using device: {device} with type {device_type}")
    return device, device_type
