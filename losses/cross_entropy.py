"""Per-session full-vocabulary CE with gate/description/diagnosis token weights."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def chunked_loss(head, hidden, labels, kinds, chunk_size, disabled_kinds=(),
                 token_weights=(1.0, 1.0, 1.0)):
    mask = labels != -100
    hidden, targets, kinds = hidden[mask], labels[mask], kinds[mask]
    total = hidden.new_zeros((), dtype=torch.float32)
    metrics = hidden.new_zeros(6, dtype=torch.float32)
    weights = hidden.new_tensor([0.0, *token_weights], dtype=torch.float32)[kinds]
    denominator = weights.sum()

    def score(h, y, k, w):
        logits = head(h).float()
        ce = F.cross_entropy(logits, y, reduction="none")
        correct = (logits.argmax(-1) == y).float()
        sums = torch.stack([ce[k == group].sum() for group in (1, 2, 3)])
        accuracy = torch.stack([correct[k == group].sum() for group in (1, 2, 3)])
        # CE removal keeps the full weighted denominator; metrics stay unweighted.
        objective = ce * w
        for group in disabled_kinds:
            objective = objective.masked_fill(k == group, 0)
        return objective.sum() / denominator, torch.cat((sums, accuracy)).detach()

    for start in range(0, targets.numel(), chunk_size):
        value, detail = checkpoint(
            score, hidden[start:start + chunk_size], targets[start:start + chunk_size],
            kinds[start:start + chunk_size], weights[start:start + chunk_size], use_reentrant=False,
        )
        total = total + value
        metrics = metrics + detail
    return total, metrics


def task_mean_loss(head, hidden, labels, kinds, chunk_size, task_weights, disabled_kinds=()):
    mask = labels != -100
    hidden, targets, kinds = hidden[mask], labels[mask], kinds[mask]
    counts = torch.bincount(kinds, minlength=4).float()
    coefficients = hidden.new_tensor([0.0, *task_weights], dtype=torch.float32)
    coefficients = coefficients * (counts > 0)
    coefficients = coefficients / coefficients.sum()
    # Ablate after Full normalization so surviving groups keep their coefficients.
    coefficients[list(disabled_kinds)] = 0
    token_weights = coefficients[kinds] / counts[kinds]
    total = hidden.new_zeros((), dtype=torch.float32)
    metrics = hidden.new_zeros(6, dtype=torch.float32)

    def score(h, y, k, weights):
        logits = head(h).float()
        ce = F.cross_entropy(logits, y, reduction="none")
        correct = (logits.argmax(-1) == y).float()
        sums = torch.stack([ce[k == group].sum() for group in (1, 2, 3)])
        accuracy = torch.stack([correct[k == group].sum() for group in (1, 2, 3)])
        return (ce * weights).sum(), torch.cat((sums, accuracy)).detach()

    for start in range(0, targets.numel(), chunk_size):
        value, detail = checkpoint(
            score, hidden[start:start + chunk_size], targets[start:start + chunk_size],
            kinds[start:start + chunk_size], token_weights[start:start + chunk_size],
            use_reentrant=False,
        )
        total = total + value
        metrics = metrics + detail
    return total, metrics
