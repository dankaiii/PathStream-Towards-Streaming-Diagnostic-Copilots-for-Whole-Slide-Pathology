"""Session-averaged distributed updates shared by PathStream and the SFT baselines."""
import contextlib
import math

import torch
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup


def optimizer_and_schedule(parameters, config, steps):
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.01, fused=True)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(steps * config["warmup_ratio"]), num_training_steps=steps)
    return optimizer, scheduler


def backward_sessions(ddp, parameters, batches, session_loss, global_size, full_size):
    # A partial batch may assign zero sessions to a rank. Sum its gradients
    # explicitly so every real session still contributes exactly once.
    partial = global_size < full_size
    if partial:
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
    scale = (1 if partial else dist.get_world_size()) / global_size
    loss_sum = torch.zeros((), device=parameters[0].device)
    for i, batch in enumerate(batches):
        sync = ddp.no_sync() if partial or i + 1 < len(batches) else contextlib.nullcontext()
        with sync:
            loss = session_loss(ddp.module if partial else ddp, batch)
            (loss * scale).backward()
        loss_sum += loss.detach()
    if partial:
        for parameter in parameters:
            dist.all_reduce(parameter.grad)
    dist.all_reduce(loss_sum)
    return float(loss_sum / global_size)
