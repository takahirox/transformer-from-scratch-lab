"""Export trained recall weights: python -m baseline.export --help."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .checkpoint import load_transformer

MAX_BROWSER_LENGTH = 128
TOLERANCE = {"atol": 1e-4, "rtol": 1e-3}


class BrowserModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        return self.model(tokens, return_attention=True)


def fixed_inputs(length, classes):
    """No random dependency: endpoints, mixed distractors and repeated tokens."""
    return [[i % classes for i in range(length)],
            [(classes - 1 - i) % classes for i in range(length)],
            [0] * length]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_checkpoint(checkpoint_path, output, lengths=None):
    model, checkpoint = load_transformer(checkpoint_path)
    settings = checkpoint["model_settings"]
    lengths = sorted(set(lengths if lengths is not None else checkpoint["training_config"]["lengths"]))
    if not lengths or any(length < 2 or length > min(settings["max_length"], MAX_BROWSER_LENGTH)
                          for length in lengths):
        raise ValueError(f"export lengths must be 2..{min(settings['max_length'], MAX_BROWSER_LENGTH)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # A failed export must not leave an old manifest advertising a complete bundle.
    (output / "manifest.json").unlink(missing_ok=True)
    wrapper = BrowserModel(model).eval()
    artifacts, cases = [], []
    for length in lengths:
        path = output / f"recall-{length}.onnx"
        example = torch.tensor([fixed_inputs(length, settings["classes"])[0]], dtype=torch.int32)
        # Deliberately fixed batch/length: constant folding removes runtime shape
        # calculations that would otherwise need CPU kernels in a WebGPU session.
        torch.onnx.export(wrapper, (example,), path, input_names=["tokens"],
                          output_names=["logits", "attention"], opset_version=17,
                          dynamo=False, external_data=False)
        onnx.checker.check_model(onnx.load(path))
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for tokens in fixed_inputs(length, settings["classes"]):
            tensor = torch.tensor([tokens], dtype=torch.int32)
            with torch.no_grad():
                logits, attention = wrapper(tensor)
                # Prove the instrumented path predicts exactly the normal CLI result.
                torch.testing.assert_close(logits, model(tensor), rtol=0, atol=0)
            actual = session.run(None, {"tokens": tensor.numpy()})
            errors = {}
            for name, expected, result in zip(("logits", "attention"), (logits, attention), actual):
                np.testing.assert_allclose(result, expected.numpy(), atol=1e-5, rtol=1e-5)
                errors[name] = float(np.max(np.abs(result - expected.numpy())))
            prediction = int(logits.argmax(-1).item())
            if int(actual[0].argmax(-1).item()) != prediction:
                raise ValueError("ONNX prediction differs from PyTorch")
            cases.append({"length": length, "tokens": tokens, "prediction": prediction,
                          "logits": logits.flatten().tolist(), "attention": attention.flatten().tolist(),
                          "native_onnx_max_abs_error": errors})
        artifacts.append({"length": length, "file": path.name, "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(output / "reference.json", {"tolerance": TOLERANCE, "cases": cases})
    manifest = {
        "format_version": 1, "task": "first-token recall from the final position",
        "model_settings": settings, "training_config": checkpoint["training_config"],
        "vocabulary": [str(i) for i in range(settings["classes"])],
        "input": {"name": "tokens", "dtype": "int32", "batch": 1},
        "outputs": {"logits": "[1, classes]", "attention": "[1, heads, query, key]"},
        "attention": "bidirectional; rows = query positions, columns = key positions; no padding",
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "versions": {"pytorch": str(torch.__version__), "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__, "opset": 17},
        "models": artifacts, "reference": "reference.json",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("web/model"))
    parser.add_argument("--lengths", type=int, nargs="+", help="supported fixed lengths; max 128 and checkpoint capacity")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        manifest = export_checkpoint(args.checkpoint, args.output, args.lengths)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Exported {len(manifest['models'])} models and PyTorch references to {args.output}")


if __name__ == "__main__":
    main()
