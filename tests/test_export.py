"""Checkpoint and ONNX integration checks (optional export dependencies)."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from baseline.checkpoint import load_transformer
from baseline.models import SequenceClassifier
from baseline.run import Config, evaluate, recall_data, seed_everything, train_comparison


class CheckpointTests(unittest.TestCase):
    def test_actual_trained_weights_reload_with_attention(self):
        torch.set_num_threads(1)
        config = Config(classes=4, width=8, heads=2, length=7, lengths=(3, 7),
                        batch_size=4, steps=4, eval_size=12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transformer.pt"
            records = train_comparison(config, path)
            model, checkpoint = load_transformer(path)
            self.assertEqual(checkpoint["model_settings"]["max_length"], 7)
            seed_everything(config.seed)
            initial = SequenceClassifier("transformer", 4, 8, 2, 7)
            self.assertFalse(torch.equal(model.embedding.weight, initial.embedding.weight))
            tokens, targets = recall_data(12, 7, 4, torch.Generator().manual_seed(config.seed + 1))
            metrics = evaluate(model, tokens, targets, 4)
            self.assertEqual(metrics["eval_accuracy"], records["models"][-1]["history"][-1]["eval_accuracy"])
            self.assertEqual(metrics["eval_loss"], records["models"][-1]["history"][-1]["eval_loss"])
            observed = []
            model.sequence.attention.register_forward_hook(lambda module, args, output: observed.append(output[1]))
            logits, attention = model(tokens, return_attention=True)
            torch.testing.assert_close(logits, model(tokens), atol=0, rtol=0)
            torch.testing.assert_close(attention, observed[0], atol=0, rtol=0)
            torch.testing.assert_close(attention.sum(-1), torch.ones(12, 2, 7))


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("onnx", "onnxruntime", "numpy")),
                     "install requirements-export.txt for ONNX integration checks")
class ExportTests(unittest.TestCase):
    def test_export_reload_numerical_equivalence_and_length_limits(self):
        from baseline.export import export_checkpoint

        torch.set_num_threads(1)
        config = Config(classes=4, width=7, heads=1, length=7, lengths=(3, 7),
                        batch_size=4, steps=3, eval_size=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trained.pt"
            train_comparison(config, path)
            output = Path(directory) / "browser"
            manifest = export_checkpoint(path, output)
            self.assertEqual([entry["length"] for entry in manifest["models"]], [3, 7])
            self.assertEqual(manifest["vocabulary"], ["0", "1", "2", "3"])
            reference = json.loads((output / "reference.json").read_text())
            self.assertEqual(len(reference["cases"]), 6)
            for case in reference["cases"]:
                self.assertEqual(len(case["attention"]), case["length"] ** 2)
                self.assertTrue(all(error < 1e-5 for error in case["native_onnx_max_abs_error"].values()))
            for lengths in ([], [1], [8], [129]):
                with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                    export_checkpoint(path, output, lengths)


if __name__ == "__main__":
    unittest.main()
