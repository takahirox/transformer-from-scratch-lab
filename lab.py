"""Small, inspectable Stage 0-7 experiments. Requires PyTorch."""
import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def seed(value):
    random.seed(value)
    torch.manual_seed(value)


def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
    weights = scores.softmax(-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=16, heads=2):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.projections = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.output = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        batch, length, dim = x.shape
        q, k, v = [layer(x).reshape(batch, length, self.heads, dim // self.heads)
                   .transpose(1, 2) for layer in self.projections]
        if mask is not None and mask.ndim == 3:
            mask = mask[:, None]
        y, weights = attention(q, k, v, mask)
        return self.output(y.transpose(1, 2).contiguous().reshape(batch, length, dim)), weights


class Position(nn.Module):
    def __init__(self, dim, max_length=4096):
        super().__init__()
        pos = torch.arange(max_length).float()[:, None]
        freq = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000) / dim))
        enc = torch.zeros(max_length, dim)
        enc[:, 0::2] = torch.sin(pos * freq)
        enc[:, 1::2] = torch.cos(pos * freq[:enc[:, 1::2].size(1)])
        self.register_buffer("enc", enc[None])

    def forward(self, x):
        if x.size(1) > self.enc.size(1):
            raise ValueError("sequence length exceeds position capacity")
        return x + self.enc[:, :x.size(1)].to(x.dtype)


class Block(nn.Module):
    def __init__(self, dim=16, heads=2):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        y, _ = self.attn(self.norm1(x))
        x = x + y
        return x + self.ff(self.norm2(x))


class Classifier(nn.Module):
    def __init__(self, kind, dim=16, heads=2):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(2, dim)
        if kind == "transformer":
            self.pos = Position(dim)
            self.model = Block(dim, heads)
        elif kind == "rnn":
            self.model = nn.RNN(dim, dim, batch_first=True)
        elif kind == "lstm":
            self.model = nn.LSTM(dim, dim, batch_first=True)
        else:
            raise ValueError(kind)
        self.head = nn.Linear(dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        if self.kind == "transformer":
            x = self.model(self.pos(x))[:, -1]
        else:
            _, hidden = self.model(x)
            x = hidden[0][-1] if self.kind == "lstm" else hidden[-1]
        return self.head(x)


def data(batch, length, device, generator):
    x = torch.randint(0, 2, (batch, length), generator=generator)
    return x.to(device), x[:, 0].to(device)


def attention_experiment():
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    q, k, v = x, x, x @ torch.diag(torch.tensor([1., 2.]))
    y, weights = attention(q, k, v)
    reference = F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
    torch.testing.assert_close(y, reference, atol=1e-6, rtol=1e-6)
    return {"input": x.tolist(), "q": q.tolist(), "k": k.tolist(), "v": v.tolist(),
            "scores": (q @ k.T).tolist(), "scaled_scores": ((q @ k.T) / math.sqrt(2)).tolist(),
            "weights": weights.tolist(), "row_sums": weights.sum(-1).tolist(),
            "output": y.tolist(), "matches_pytorch": True}


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def train(args, device):
    results = []
    for kind in ("rnn", "lstm", "transformer"):
        seed(args.seed)
        model = Classifier(kind, args.dim, args.heads).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.003)
        rng = torch.Generator().manual_seed(args.seed + 100)
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for _ in range(args.steps):
                x, y = data(args.batch, args.length, device, rng)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                x, y = data(512, args.length, device,
                            torch.Generator().manual_seed(args.seed + 9000))
                acc = (model(x).argmax(-1) == y).float().mean().item()
            results.append(dict(device=str(device), model=kind, epoch=epoch,
                                loss=round(statistics.mean(losses), 6), accuracy=round(acc, 6),
                                parameters=sum(p.numel() for p in model.parameters())))
    return results


def benchmark(args, device):
    rows = []
    for length in args.bench_lengths:
        for kind in ("rnn", "lstm", "transformer"):
            seed(args.seed)
            model = Classifier(kind, args.dim, args.heads).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.003)
            x, y = data(args.bench_batch, length, device,
                        torch.Generator().manual_seed(args.seed))
            def run(training):
                if training:
                    model.train()
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(x), y)
                    loss.backward()
                    opt.step()
                else:
                    model.eval()
                    with torch.no_grad():
                        model(x)
                synchronize(device)
            for training in (False, True):
                for _ in range(args.warmup):
                    run(training)
                times = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    run(training)
                    times.append((time.perf_counter() - start) * 1000)
                rows.append(dict(device=str(device), model=kind, length=length,
                                 mode="train_step" if training else "forward",
                                 median_ms=round(statistics.median(times), 5),
                                 min_ms=round(min(times), 5), max_ms=round(max(times), 5),
                                 repeats=len(times),
                                 attention_weights_estimated_bytes_fp32=(
                                     4 * args.bench_batch * args.heads * length**2
                                     if kind == "transformer" else 0)))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(env, train_rows, bench_rows, args):
    lines = ["# Stage 7: evidence-based synthesis", "",
             "Small synthetic experiments; not a reproduction of paper-scale training.", "",
             "## Environment", "", json.dumps(env, indent=2), "",
             "## Attention and architecture", "",
             "Stages 1-2 explicitly print Q, K, V, scores, scaled scores, softmax weights and output "
             "in attention.json, with a PyTorch reference check. Stages 3-4 implement multi-head "
             "attention, sinusoidal positions, residuals, layer normalization, feed-forward layers "
             "and a trainable classifier without nn.Transformer.", "",
             "## Long-range copy task (Stage 5)", "",
             "All models predict the first binary token from the final sequence representation. "
             "The validation sequence is held out and shared between models.", "",
             "| Device | Model | Final loss | Validation accuracy | Parameters |",
             "|---|---|---:|---:|---:|"]
    for row in train_rows:
        if row["epoch"] == args.epochs:
            lines.append(f"| {row['device']} | {row['model']} | {row['loss']:.4f} | "
                         f"{row['accuracy']:.3f} | {row['parameters']} |")
    lines += ["", "## Runtime (Stage 6)", "",
              "Times are wall-clock statistics after warmup. Forward and full optimizer steps "
              "are timed separately, with synchronization on MPS.", "",
              "| Device | Model | Length | Mode | Median ms | Min ms | Max ms |",
              "|---|---|---:|---|---:|---:|---:|"]
    for row in bench_rows:
        lines.append(f"| {row['device']} | {row['model']} | {row['length']} | "
                     f"{row['mode']} | {row['median_ms']:.4f} | "
                     f"{row['min_ms']:.4f} | {row['max_ms']:.4f} |")
    lines += ["", "## Interpretation and limitations", "",
              "- Explicit full self-attention stores O(L squared) weights per head; the CSV "
              "contains a theoretical fp32 weight-matrix size, NOT observed peak memory.",
              "- Recurrent models process sequence positions sequentially; attention exposes "
              "parallel computation across positions. Real speed depends on kernels and hardware.",
              "- Models differ in parameter count. One toy task and a few epochs cannot "
              "establish architecture-wide superiority or robust long-range learning.",
              "- Any missing MPS rows mean that MPS was not run. Only present rows are measurements.",
              "- For stronger conclusions, test more seeds, longer sequences and actual Apple Silicon MPS.",
              "", "## Conclusion", "",
              "The experiments let readers inspect attention directly and compare observed learning "
              "and timing rather than assuming the original paper's claims automatically apply "
              "to this small regime.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--device", choices=["cpu", "mps", "both"], default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--bench-lengths", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--bench-batch", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if (args.heads < 1 or args.dim < 1 or args.dim % args.heads or
        min(args.length, args.batch, args.epochs, args.steps, args.bench_batch,
            args.repeats) < 1 or args.warmup < 0 or not args.bench_lengths or
        min(args.bench_lengths) < 1 or max([args.length] + args.bench_lengths) > 4096):
        p.error("invalid lengths/counts; dimension must be divisible by heads")
    if args.device != "cpu" and not torch.backends.mps.is_available():
        p.error("MPS is unavailable; choose --device cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    seed(args.seed)
    env = dict(python=sys.version.split()[0], torch=torch.__version__, os=platform.platform(),
               cpu_count=os.cpu_count(), processor=platform.processor(),
               torch_threads=torch.get_num_threads(),
               mps_available=torch.backends.mps.is_available(), seed=args.seed,
               command_options={**vars(args), "output": str(args.output)})
    (args.output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    (args.output / "attention.json").write_text(json.dumps(attention_experiment(), indent=2) + "\n")
    devices = [torch.device("cpu"), torch.device("mps")] if args.device == "both" else [torch.device(args.device)]
    train_rows = [r for dev in devices for r in train(args, dev)]
    bench_rows = [r for dev in devices for r in benchmark(args, dev)]
    write_csv(args.output / "training.csv", train_rows)
    write_csv(args.output / "benchmark.csv", bench_rows)
    (args.output / "synthesis.md").write_text(report(env, train_rows, bench_rows, args), encoding="utf-8")
    print(f"Completed Stage 0-7 baseline; results: {args.output}")


if __name__ == "__main__":
    main()
