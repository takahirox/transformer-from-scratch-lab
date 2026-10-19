"""Run the integrated baseline: python -m baseline.run --output results/quick."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F

from .models import SequenceClassifier, scaled_attention
from .checkpoint import save_transformer

KINDS = ("rnn", "lstm", "transformer")


@dataclass
class Config:
    seed: int = 7
    threads: int = 1
    classes: int = 8
    width: int = 32
    heads: int = 4
    batch_size: int = 32
    length: int = 32
    steps: int = 200
    eval_size: int = 256
    log_every: int = 20
    learning_rate: float = 0.003
    lengths: tuple = (16, 32, 64)
    warmup: int = 3
    repeats: int = 10
    devices: str = "auto"

    def validate(self):
        positive = (self.threads, self.classes, self.width, self.heads, self.batch_size,
                    self.steps, self.eval_size, self.log_every, self.repeats)
        if min(positive) <= 0 or self.classes < 2 or self.width % self.heads:
            raise ValueError("positive sizes required; classes >= 2; width divisible by heads")
        if not self.lengths or min(self.lengths) < 2 or self.length < 2:
            raise ValueError("all sequence lengths must be >= 2")
        if self.warmup < 1 or self.learning_rate <= 0:
            raise ValueError("warmup and learning rate must be positive")
        if self.devices not in ("auto", "cpu"):
            raise ValueError("devices must be auto or cpu")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment(config):
    cpu = platform.processor() or platform.machine()
    physical_memory = None
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        physical_memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    source_root = Path(__file__).resolve().parent
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "pytorch": torch.__version__,
        "os": platform.platform(), "machine": platform.machine(), "cpu": cpu,
        "logical_cpu_count": os.cpu_count(), "physical_memory_bytes": physical_memory,
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "dtype": "float32", "training_device": "cpu", "config": asdict(config),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source_root.glob("*.py"))},
    }


def attention_evidence():
    # Same tiny example as the original stage script, preserved separately.
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    wq, wk = torch.eye(2), torch.eye(2)
    wv = torch.diag(torch.tensor([1., 2.]))
    q, k, v = x @ wq, x @ wk, x @ wv
    raw_scores = q @ k.T
    scaled_scores = raw_scores / (q.shape[-1] ** 0.5)
    output, weights = scaled_attention(q, k, v)
    reference = F.scaled_dot_product_attention(q, k, v)
    error = (output - reference).abs().max().item()
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(weights.sum(-1), torch.ones(3))
    tensors = {"X": x, "W_Q": wq, "W_K": wk, "W_V": wv, "Q": q, "K": k, "V": v,
               "QK_transpose": raw_scores, "scaled_scores": scaled_scores,
               "weights": weights, "output": output}
    return {
        "axes": "rows: query position; weight columns: key position; Q/K/V columns: features",
        "tensors": {name: {"shape": list(t.shape), "values": t.tolist()} for name, t in tensors.items()},
        "row_sums": weights.sum(-1).tolist(), "reference_max_abs_error": error,
        "reference": "torch.nn.functional.scaled_dot_product_attention; no mask, dropout=0",
    }


def recall_data(count, length, classes, generator):
    tokens = torch.randint(classes, (count, length), generator=generator)
    return tokens, tokens[:, 0].clone()


def make_model(kind, config):
    return SequenceClassifier(kind, config.classes, config.width, config.heads,
                              max(config.length, *config.lengths))


def evaluate(model, tokens, targets, batch_size):
    model.eval()
    loss_sum, correct = 0., 0
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            logits = model(tokens[start:start + batch_size])
            y = targets[start:start + batch_size]
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(-1) == y).sum().item()
    return {"eval_loss": loss_sum / len(tokens), "eval_accuracy": correct / len(tokens)}


def train_comparison(config, checkpoint=None):
    # Separate CPU generators: evaluation never consumes the training RNG stream.
    eval_generator = torch.Generator().manual_seed(config.seed + 1)
    eval_x, eval_y = recall_data(config.eval_size, config.length, config.classes, eval_generator)
    records = []
    for kind in KINDS:
        seed_everything(config.seed)
        model = make_model(kind, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        generator = torch.Generator().manual_seed(config.seed + 2)
        history = [{"step": 0, "train_loss": None,
                    **evaluate(model, eval_x, eval_y, config.batch_size)}]
        accumulated = 0.
        interval_steps = 0
        for step in range(1, config.steps + 1):
            # Reset generator identically per architecture: exactly the same minibatches.
            x, y = recall_data(config.batch_size, config.length, config.classes, generator)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            accumulated += loss.item()
            interval_steps += 1
            if step % config.log_every == 0 or step == config.steps:
                history.append({"step": step, "train_loss": accumulated / interval_steps,
                                **evaluate(model, eval_x, eval_y, config.batch_size)})
                accumulated, interval_steps = 0., 0
        # Also evaluate dependency distance without claiming retraining at each length.
        length_results = []
        for length in config.lengths:
            g = torch.Generator().manual_seed(config.seed + 100 + length)
            x, y = recall_data(config.eval_size, length, config.classes, g)
            length_results.append({"length": length, **evaluate(model, x, y, config.batch_size)})
        records.append({"model": kind, "parameters": sum(p.numel() for p in model.parameters()),
                        "history": history, "evaluation_by_length": length_results})
        if kind == "transformer" and checkpoint is not None:
            save_transformer(model, asdict(config), checkpoint)
    return {"task": "first-token recall from final position; iid uniform distractors",
            "chance_accuracy": 1 / config.classes, "training_device": "cpu",
            "optimizer": "Adam; gradient norm clipped to 1.0; no dropout",
            "data_seeds": {"evaluation": config.seed + 1, "training": config.seed + 2},
            "models": records}


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def timed_samples(operation, device, warmup, repeats):
    for _ in range(warmup):
        operation()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        operation()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    return {"samples_ms": samples, "median_ms": statistics.median(samples),
            "min_ms": min(samples), "max_ms": max(samples)}


def benchmark(config):
    devices = ["cpu"]
    if config.devices == "auto" and torch.backends.mps.is_available():
        devices.append("mps")
    rows = []
    for device in devices:
        for kind in KINDS:
            for length in config.lengths:
                seed_everything(config.seed)
                model = make_model(kind, config).to(device)
                g = torch.Generator().manual_seed(config.seed + length)
                x, y = recall_data(config.batch_size, length, config.classes, g)
                x, y = x.to(device), y.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

                def forward():
                    with torch.no_grad():
                        model(x)

                def train_step():
                    optimizer.zero_grad(set_to_none=True)
                    F.cross_entropy(model(x), y).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                forward_times = timed_samples(forward, device, config.warmup, config.repeats)
                model.train()
                train_times = timed_samples(train_step, device, config.warmup, config.repeats)
                parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                memory = {
                    "parameter_tensor_bytes_exact": parameter_bytes,
                    "gradient_tensor_bytes_estimate": parameter_bytes,
                    "adam_two_moment_tensor_bytes_estimate": 2 * parameter_bytes,
                    "one_attention_matrix_bytes_estimate": (
                        config.batch_size * config.heads * length * length * 4 if kind == "transformer" else 0),
                    "mps_current_tensor_bytes_snapshot": (
                        torch.mps.current_allocated_memory() if device == "mps" else None),
                }
                rows.append({"device": device, "model": kind, "length": length,
                             "forward": forward_times, "train_step": train_times, "memory": memory})
                # Release each case before measuring the next device/architecture/length.
                del optimizer, model, x, y
                if device == "mps":
                    torch.mps.empty_cache()
    return {
        "devices_run": devices,
        "mps_status": "measured" if "mps" in devices else (
            "not run: --devices cpu" if config.devices == "cpu" else "not run: MPS unavailable"),
        "method": "perf_counter wall time; per-sample device synchronization; inputs already on device; "
                  "eval/no_grad forward; train step includes zero_grad, cross entropy, backward, clipping, Adam; "
                  "fresh untrained model per case, reused/mutated through warmup and samples",
        "memory_method": "Exact parameter storage and analytical tensor estimates, NOT peak/process memory. "
                         "One attention matrix is B*H*L^2*4 bytes; excludes other activations, gradients, "
                         "allocator/workspace overhead. MPS snapshot is live tensor allocation after train steps, "
                         "not a peak; CPU allocator memory is not measured.",
        "rows": rows,
    }


def synthesis(env, attention, training, benchmarks):
    config = env["config"]
    lines = [
        "# Stage 0–7 baseline synthesis", "",
        "Generated from [environment.json](environment.json), [attention.json](attention.json), "
        "[training.json](training.json), and [benchmark.json](benchmark.json).", "",
        "## Measurements", "",
        f"Environment: Python {env['python']}, PyTorch {env['pytorch']}; {env['os']}; {env['cpu']}. "
        f"CPU threads: {env['torch_threads']}; seed: {config['seed']}; float32. "
        f"MPS available: {env['mps_available']}. MPS status: {benchmarks['mps_status']}. "
        f"PYTORCH_ENABLE_MPS_FALLBACK: {env['mps_fallback_env']}.", "",
        "Stages 1–2: explicit projections, raw/scaled scores, weights, shapes and weighted values "
        f"are in attention.json. Max absolute error against PyTorch SDPA: {attention['reference_max_abs_error']:.3g}; "
        f"weight row sums: {attention['row_sums']}.", "",
        "Stages 3–4: the trained Transformer uses learned Q/K/V and output projections, "
        "split/concatenated heads, fixed sinusoidal positions, pre-layer normalization, "
        "residual connections and a two-linear-layer ReLU feed-forward network.", "",
        f"Stage 5: first-token recall from the final position, {config['classes']} classes "
        f"(chance {training['chance_accuracy']:.1%}), length {config['length']}, "
        f"{config['steps']} Adam steps, batch {config['batch_size']}, learning rate {config['learning_rate']}. "
        "All models receive identical minibatches and held-out examples; widths and step budgets match, "
        "parameter counts differ. Training is on CPU.", "",
        "| Model | Parameters | Initial eval loss | Final train loss | Final eval loss | Final eval accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in training["models"]:
        first, last = record["history"][0], record["history"][-1]
        lines.append(f"| {record['model']} | {record['parameters']} | {first['eval_loss']:.4f} | "
                     f"{last['train_loss']:.4f} | {last['eval_loss']:.4f} | {last['eval_accuracy']:.1%} |")
    lines += ["", "Observed held-out loss changes (final minus initial; negative means improvement): " +
              "; ".join(f"{r['model']} {r['history'][-1]['eval_loss'] - r['history'][0]['eval_loss']:+.4f}"
                        for r in training["models"]) +
              ". These are outcomes of this budget and seed, not architecture-wide rankings."]
    lines += ["", "Held-out accuracy by dependency distance (same trained weights; no retraining):", "",
              "| Model | Length | Accuracy |", "| --- | ---: | ---: |"]
    for record in training["models"]:
        for result in record["evaluation_by_length"]:
            lines.append(f"| {record['model']} | {result['length']} | {result['eval_accuracy']:.1%} |")
    lines += ["", f"Stage 6: {config['warmup']} warmup calls and {config['repeats']} timed samples per operation. "
              + benchmarks["method"] + ".", "",
              "| Device | Model | Length | Forward median (ms) | Train-step median (ms) | One attention matrix estimate (bytes) |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in benchmarks["rows"]:
        lines.append(f"| {row['device']} | {row['model']} | {row['length']} | "
                     f"{row['forward']['median_ms']:.3f} | {row['train_step']['median_ms']:.3f} | "
                     f"{row['memory']['one_attention_matrix_bytes_estimate']} |")
    lengths = sorted(set(config["lengths"]))
    if len(lengths) > 1:
        lines += ["", f"Observed forward median ratios, length {lengths[-1]} / {lengths[0]}:", ""]
        for device in benchmarks["devices_run"]:
            for kind in KINDS:
                selected = {r["length"]: r for r in benchmarks["rows"]
                            if r["device"] == device and r["model"] == kind}
                ratio = selected[lengths[-1]]["forward"]["median_ms"] / selected[lengths[0]]["forward"]["median_ms"]
                lines.append(f"- {device} {kind}: {ratio:.2f}x.")
    lines += ["", benchmarks["memory_method"], "", "## Theory and interpretation", "",
              "Dense attention forms a score for every query/key pair: O(L²D) work and O(HL²) "
              "score storage per batch item. Recurrent layers take L sequential transitions with "
              "O(LD²) work. Attention directly connects distant positions within a layer; recurrent "
              "models carry information through intermediate states. These are structural claims, "
              "not measured asymptotic rates or guarantees of better accuracy.", "",
              "The tables show this run's learning and timing behavior. Any architecture below chance "
              "or failing to learn is retained as evidence of this small budget and setup. Final-position "
              "recall tests information transport across distractors, not language modeling. Timing "
              "rankings at these short lengths cannot establish a universal speed advantage.", "",
              "## Limitations and unrun cases", "",
              "One seed, one synthetic task, one layer, small widths and budgets; no hyperparameter "
              "search, matched parameter budgets, confidence intervals or large-length scaling claim. "
              "Repeated timing samples describe within-run variability, not independent experiments. "
              "CPU training uses deterministic algorithms; warn-only mode records the possibility of "
              "nondeterministic device kernels. Exact cross-version/device reproducibility is not promised. "
              "Evaluation at other lengths is generalization, not a comparison of models trained at those lengths.", "",
              f"MPS: {benchmarks['mps_status']}. No MPS training comparison was performed. "
              "Larger training, longer sequences and repeated Apple Silicon CPU/MPS trials remain optional "
              "follow-up experiments. Memory estimates are not measured peaks. This integrated baseline "
              "does not complete the richer stage Issues #2–#8.", ""]
    return "\n".join(lines)


def run(config, output, checkpoint=None):
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave an old report implying this run completed.
    for name in ("environment.json", "attention.json", "training.json", "benchmark.json", "report.md"):
        (output / name).unlink(missing_ok=True)
    if checkpoint is not None:
        Path(checkpoint).unlink(missing_ok=True)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed_everything(config.seed)
    env = environment(config)
    write_json(output / "environment.json", env)
    attention = attention_evidence()
    write_json(output / "attention.json", attention)
    training = train_comparison(config, checkpoint)
    write_json(output / "training.json", training)
    timings = benchmark(config)
    write_json(output / "benchmark.json", timings)
    (output / "report.md").write_text(synthesis(env, attention, training, timings))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/quick"))
    parser.add_argument("--checkpoint", type=Path,
                        help="save the final trained Transformer weights/settings for browser export")
    parser.add_argument("--devices", choices=("cpu", "auto"), default="auto",
                        help="CPU always; auto also benchmarks MPS when available")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--length", type=int, default=32, help="training sequence length")
    parser.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    config = Config(seed=args.seed, steps=args.steps, length=args.length, lengths=tuple(args.lengths),
                    warmup=args.warmup, repeats=args.repeats, devices=args.devices)
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved baseline artifacts to {run(config, args.output, args.checkpoint)}")


if __name__ == "__main__":
    main()
