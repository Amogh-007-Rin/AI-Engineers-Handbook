"""Offline Diffusers scheduler contract; no pretrained model required."""

import torch
from diffusers import DDPMScheduler


def add_noise(sample, noise, timestep):
    sample, noise = torch.as_tensor(sample, dtype=torch.float32), torch.as_tensor(noise, dtype=torch.float32)
    if sample.shape != noise.shape or sample.ndim != 4:
        raise ValueError("sample and noise must share NCHW shape")
    scheduler = DDPMScheduler(num_train_timesteps=20)
    timestep = torch.tensor([timestep], dtype=torch.long)
    if timestep.item() < 0 or timestep.item() >= scheduler.config.num_train_timesteps:
        raise ValueError("timestep out of range")
    return scheduler.add_noise(sample, noise, timestep)
