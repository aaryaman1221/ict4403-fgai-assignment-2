"""Word-level autoregressive text generation for ICT4403 Assignment 2.

The same data split, context length, embedding size, hidden size, optimizer,
batch size, and training-step budget are used for each recurrent architecture.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import ssl
import time
import urllib.request
from collections import Counter
from pathlib import Path

import torch
import certifi
from torch import nn
from torch.nn import functional as F


DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
ARCHITECTURES = ("rnn", "bidir_rnn", "lstm", "gru")
SPECIAL_UNK = "<unk>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return
    print(f"Downloading Tiny Shakespeare from {DATA_URL}")
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "ICT4403-assignment/1.0"})
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=60, context=ssl_context) as response:
        path.write_bytes(response.read())


def tokenize(text: str) -> list[str]:
    text = text.replace("’", "'").lower()
    return TOKEN_PATTERN.findall(text)


def make_vocabulary(tokens: list[str], max_vocab: int, min_freq: int) -> tuple[dict[str, int], list[str]]:
    counts = Counter(tokens)
    words = [word for word, count in counts.most_common() if count >= min_freq]
    words = words[: max(0, max_vocab - 1)]
    vocab_words = [SPECIAL_UNK, *words]
    return {word: i for i, word in enumerate(vocab_words)}, vocab_words


def encode(tokens: list[str], vocab: dict[str, int]) -> torch.Tensor:
    unknown = vocab[SPECIAL_UNK]
    return torch.tensor([vocab.get(token, unknown) for token in tokens], dtype=torch.long)


def sample_batch(data: torch.Tensor, batch_size: int, context: int, device: torch.device):
    starts = torch.randint(0, len(data) - context - 1, (batch_size,))
    x = torch.stack([data[s : s + context] for s in starts])
    y = torch.stack([data[s + context] for s in starts])
    return x.to(device), y.to(device)


def fixed_windows(data: torch.Tensor, context: int, max_windows: int = 5000):
    starts = torch.arange(0, len(data) - context, max(1, context // 2))
    if len(starts) > max_windows:
        keep = torch.linspace(0, len(starts) - 1, max_windows).long()
        starts = starts[keep]
    x = torch.stack([data[s : s + context] for s in starts])
    y = torch.stack([data[s + context] for s in starts])
    return x, y


class WordGenerator(nn.Module):
    def __init__(self, kind: str, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if kind in ("rnn", "bidir_rnn"):
            self.recurrent = nn.RNN(
                embedding_dim, hidden_dim, batch_first=True,
                bidirectional=(kind == "bidir_rnn"),
            )
            directions = 2 if kind == "bidir_rnn" else 1
        elif kind == "lstm":
            self.recurrent = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            directions = 1
        elif kind == "gru":
            self.recurrent = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
            directions = 1
        else:
            raise ValueError(f"Unknown architecture: {kind}")
        self.output = nn.Linear(hidden_dim * directions, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, state = self.recurrent(self.embedding(x))
        hidden = state[0] if isinstance(state, tuple) else state
        if self.kind == "bidir_rnn":
            summary = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            summary = hidden[-1]
        return self.output(summary)


@torch.no_grad()
def evaluate(model: nn.Module, data: torch.Tensor, context: int, device: torch.device):
    model.eval()
    x, y = fixed_windows(data, context)
    losses, correct, total = [], 0, 0
    batch_size = 256
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        logits = model(xb)
        losses.append(F.cross_entropy(logits, yb, reduction="sum").item())
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += len(yb)
    mean_loss = sum(losses) / max(total, 1)
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20)), "accuracy": correct / max(total, 1)}


@torch.no_grad()
def generate(model, seed_text, vocab, vocab_words, context, temperature, count, device):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero")
    model.eval()
    seed_tokens = tokenize(seed_text)
    if not seed_tokens:
        seed_tokens = ["to", "be"]
    ids = [vocab.get(token, vocab[SPECIAL_UNK]) for token in seed_tokens]
    for _ in range(count):
        window = ids[-context:]
        logits = model(torch.tensor([window], dtype=torch.long, device=device))[0]
        probabilities = F.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1).item()
        ids.append(next_id)
    return detokenize([vocab_words[i] for i in ids])


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]}'’])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = text.replace(" ' ", "'")
    return text


def save_plots(results: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    display_names = {"rnn": "Simple RNN", "bidir_rnn": "Bidirectional RNN", "lstm": "LSTM", "gru": "GRU"}
    for row in results:
        epochs = [item["epoch"] for item in row["history"]]
        losses = [item["val_loss"] for item in row["history"]]
        ax.plot(epochs, losses, marker="o", label=display_names[row["model"]])
    ax.set(title="Validation loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_loss.png", dpi=160)
    plt.close(fig)

    names = [display_names[row["model"]] for row in results]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    measures = (
        ("test_perplexity", "Test perplexity"),
        ("test_accuracy", "Test accuracy"),
        ("train_seconds", "Training time (s)"),
    )
    for ax, (key, title) in zip(axes, measures):
        ax.bar(names, [row[key] for row in results], color="#5577aa")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)


def train_model(kind, train_ids, val_ids, vocab_size, args, device):
    model = WordGenerator(kind, vocab_size, args.embedding_dim, args.hidden_dim).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for _ in range(args.steps_per_epoch):
            x, y = sample_batch(train_ids, args.batch_size, args.context, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
        val_metrics = evaluate(model, val_ids, args.context, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": running_loss / args.steps_per_epoch,
            "val_loss": val_metrics["loss"],
            "val_perplexity": val_metrics["perplexity"],
            "val_accuracy": val_metrics["accuracy"],
        })
        print(f"{kind:10s} epoch {epoch + 1:02d}/{args.epochs}: "
              f"train loss {history[-1]['train_loss']:.4f}, "
              f"val loss {val_metrics['loss']:.4f}, "
              f"val accuracy {val_metrics['accuracy']:.4f}")
    elapsed = time.perf_counter() - start_time
    model.eval()
    return model, history, elapsed, parameter_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tinyshakespeare.txt"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=4403)
    parser.add_argument("--context", type=int, default=24)
    parser.add_argument("--max-vocab", type=int, default=10000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.6, 0.9, 1.2])
    parser.add_argument("--generated-words", type=int, default=70)
    parser.add_argument("--quick", action="store_true", help="Short smoke-sized run; not recommended for final results")
    args = parser.parse_args()
    if args.quick:
        args.epochs, args.steps_per_epoch = 1, 10

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    download_dataset(args.data)
    raw_text = args.data.read_text(encoding="utf-8")
    tokens = tokenize(raw_text)
    train_end, val_end = int(len(tokens) * 0.80), int(len(tokens) * 0.90)
    train_tokens, val_tokens, test_tokens = tokens[:train_end], tokens[train_end:val_end], tokens[val_end:]
    vocab, vocab_words = make_vocabulary(train_tokens, args.max_vocab, args.min_freq)
    train_ids = encode(train_tokens, vocab)
    val_ids = encode(val_tokens, vocab)
    test_ids = encode(test_tokens, vocab)
    print(f"Dataset: {args.data} | {len(raw_text):,} characters, {len(tokens):,} tokens, "
          f"{len(vocab):,} vocabulary items | device: {device}")

    results = []
    sample_records = []
    for model_index, kind in enumerate(ARCHITECTURES):
        # Keep initialization streams reproducible while using each model's own seed.
        set_seed(args.seed + model_index)
        model, history, train_seconds, parameter_count = train_model(
            kind, train_ids, val_ids, len(vocab), args, device
        )
        test_metrics = evaluate(model, test_ids, args.context, device)
        results.append({
            "model": kind,
            "parameters": parameter_count,
            "train_seconds": train_seconds,
            "best_val_loss": min(row["val_loss"] for row in history),
            "test_loss": test_metrics["loss"],
            "test_perplexity": test_metrics["perplexity"],
            "test_accuracy": test_metrics["accuracy"],
            "history": history,
        })
        torch.save({
            "model": model.state_dict(),
            "vocab": vocab,
            "config": vars(args) | {"data": str(args.data), "output": str(args.output)},
        }, args.output / f"{kind}.pt")
        for seed_text in ("Friends, Romans, countrymen", "To be, or not to be"):
            for temperature in args.temperatures:
                sample_records.append({
                    "model": kind,
                    "seed": seed_text,
                    "temperature": temperature,
                    "text": generate(model, seed_text, vocab, vocab_words, args.context,
                                     temperature, args.generated_words, device),
                })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["model", "parameters", "train_seconds", "best_val_loss", "test_loss", "test_perplexity", "test_accuracy"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in results])
    (args.output / "metrics.json").write_text(json.dumps({
        "dataset": {"source": DATA_URL, "characters": len(raw_text), "tokens": len(tokens),
                    "vocabulary": len(vocab), "train_tokens": len(train_ids),
                    "validation_tokens": len(val_ids), "test_tokens": len(test_ids)},
        "settings": {"seed": args.seed, "context": args.context, "embedding_dim": args.embedding_dim,
                     "hidden_dim": args.hidden_dim, "batch_size": args.batch_size,
                     "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
                     "learning_rate": args.learning_rate, "device": str(device)},
        "models": results,
    }, indent=2), encoding="utf-8")
    with (args.output / "generated_samples.md").open("w", encoding="utf-8") as handle:
        handle.write("# Generated text samples\n\nSampling uses the same two seed sequences for every model.\n")
        for record in sample_records:
            handle.write(f"\n## {record['model']} | temperature {record['temperature']} | seed: {record['seed']}\n\n")
            handle.write(f"> {record['text']}\n")
    save_plots(results, args.output)
    print(f"Saved metrics, model checkpoints, and samples in {args.output}")


if __name__ == "__main__":
    main()
