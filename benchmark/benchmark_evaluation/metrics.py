"""Shared benchmark metrics. Reported scores use the paper's ×100 scale."""
from collections import Counter
import re
from statistics import mean


def percent_mean(values):
    return 100 * mean(values) if values else None


def invalid_output(turn):
    native, reported = turn.get("native_decision"), turn.get("decision")
    bounded_chunk = turn.get("native_chunk_budget") and (native or reported) in ("update", "silent")
    return (turn.get("protocol_error") or "protocol_error" in (native, reported)
            or (turn.get("hit_generation_limit") and not bounded_chunk)
            or turn.get("api_status") not in (None, "completed", "stop"))


def decision(turn):
    if turn is None:
        return "unexecuted"
    if invalid_output(turn):
        return "error"
    native, reported = turn.get("native_decision"), turn.get("decision")
    if native in ("update", "silent"):
        return native
    if turn.get("native_silence"):
        return "silent"
    if reported in ("update", "silent"):
        return reported
    text = turn.get("raw_response", turn.get("response"))
    if not isinstance(text, str) or not text.strip():
        return "error"
    return "silent" if text.strip().casefold().rstrip(".。!！") == "silent" else "update"


def gate_scores(pairs):
    """Pool scored frames before averaging the two class F1 scores."""
    counts = Counter(pairs)
    result = {}
    for label in ("update", "silent"):
        correct = counts[label, label]
        expected = sum(n for (gold, _), n in counts.items() if gold == label)
        predicted = sum(n for (_, pred), n in counts.items() if pred == label)
        total = expected + predicted
        result[label + "_f1"] = 200 * correct / total if total else None
    values = list(result.values())
    result["macro_f1"] = mean(values) if None not in values else None
    return result


def rouge_l_f1(reference, response):
    """Word-level longest common subsequence F1, before table scaling."""
    left, right = (re.findall(r"[a-z0-9]+", text.lower()) for text in (reference, response))
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for word in left:
        current = [0]
        for j, other in enumerate(right):
            length = previous[j] + 1 if word == other else max(previous[j + 1], current[-1])
            current.append(length)
        previous = current
    return 2 * previous[-1] / (len(left) + len(right))


def conch_encoder(checkpoint, device):
    """Load CONCH v1 once and return a callable encoding one nonempty description."""
    import torch
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=str(checkpoint), device=device)
    model.eval()
    tokenizer = get_tokenizer()
    chunk_size = 127 - tokenizer.num_special_tokens_to_add(pair=False)

    @torch.inference_mode()
    def encode(text):
        ids = tokenizer.encode(text.strip(), add_special_tokens=False)
        rows = []
        for start in range(0, len(ids), chunk_size):
            chunk = [tokenizer.bos_token_id, *ids[start:start + chunk_size], tokenizer.eos_token_id]
            rows.append(chunk + [tokenizer.pad_token_id] * (128 - len(chunk)))
        vectors = model.encode_text(torch.tensor(rows, device=device), normalize=True)
        return torch.nn.functional.normalize(vectors.float().mean(dim=0), dim=0)

    return encode


def stream_scores(streams, encode):
    """Score every gate after frame zero and every same-frame reference description."""
    gates, rouge, conch = [], [], []
    for actions, references, turns in streams:
        for i, action in enumerate(actions):
            turn = turns.get(i)
            prediction = decision(turn)
            if i > 0:
                gates.append((action, prediction))
            if action == "update":
                reference = references[i]
                response = turn["response"].strip() if prediction == "update" else ""
                rouge.append(rouge_l_f1(reference, response))
                cosine = (encode(reference) @ encode(response)).clamp(-1, 1).item() if response else 0.0
                conch.append(cosine)
    return {"gate_macro_f1": gate_scores(gates)["macro_f1"],
            "description_rouge_l": percent_mean(rouge), "description_conch": percent_mean(conch)}
