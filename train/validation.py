"""Evaluate the training objective and track the best validation checkpoint."""
import torch
import torch.distributed as dist

from losses.diagnosis_attention import ATTENTION_GROUPS


CE_GROUPS = ("gate_ce", "description_ce", "diagnosis_ce")


def move_batch(batch, device):
    items = batch.pop("attention_items")
    tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    tensors["attention_items"] = [
        {key: value if key == "kind" else value.to(device, non_blocking=True)
         for key, value in item.items()} for item in items
    ]
    return tensors


def averages(names, sums, counts):
    return {name: float(value / count) if count else None
            for name, value, count in zip(names, sums, counts)}


def validate(model, loader, device):
    model.eval()
    totals = torch.zeros(12, dtype=torch.float64, device=device)
    with torch.random.fork_rng(devices=[device]), torch.inference_mode():
        for batch in loader:
            counts = batch.pop("batch_counts").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, ce, attention = model(**move_batch(batch, device))
            totals[0] += loss.double()
            totals[1] += 1
            # Average each component over sessions that supply its supervision.
            totals[2:5] += ce[:3].double() / counts.clamp_min(1)
            totals[5:8] += counts > 0
            totals[8:10] += attention[:2].double() / attention[2:].clamp_min(1)
            totals[10:12] += attention[2:] > 0
    dist.all_reduce(totals)
    model.train()
    return {"loss": float(totals[0] / totals[1]), "sessions": int(totals[1]),
            **averages(CE_GROUPS, totals[2:5], totals[5:8]),
            **averages(ATTENTION_GROUPS, totals[8:10], totals[10:12])}


class EarlyStopping:
    def __init__(self):
        self.best_loss = float("inf")
        self.best_step = None
        self.bad_evaluations = 0

    def observe(self, loss, step):
        if loss < self.best_loss:
            self.best_loss, self.best_step = loss, step
            self.bad_evaluations = 0
            return True
        self.bad_evaluations += 1
        return False
