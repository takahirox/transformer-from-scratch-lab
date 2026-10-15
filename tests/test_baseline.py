"""Lightweight CPU checks: python -m unittest discover -s tests -v."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from baseline.models import MultiHeadAttention, PositionalEncoding, SequenceClassifier, scaled_attention
from baseline.run import Config, attention_evidence, recall_data, run, train_comparison


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tiny_numerical_example(self):
        evidence = attention_evidence()
        weights = torch.tensor(evidence["tensors"]["weights"]["values"])
        # Independently calculated first row: exp(1/sqrt(2)), 1, exp(1/sqrt(2)).
        expected = torch.tensor([0.40111209, 0.19777581, 0.40111209])
        torch.testing.assert_close(weights[0], expected)
        output = torch.tensor(evidence["tensors"]["output"]["values"])
        torch.testing.assert_close(output[0], torch.tensor([0.80222419, 1.19777581]))
        self.assertLess(evidence["reference_max_abs_error"], 1e-6)

    def test_attention_reference_and_causal_mask(self):
        torch.manual_seed(3)
        q, k, v = (torch.randn(2, 3, 5, 4) for _ in range(3))
        mask = torch.ones(5, 5, dtype=torch.bool).tril()
        actual, weights = scaled_attention(q, k, v, mask)
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.equal(weights[..., ~mask], torch.zeros_like(weights[..., ~mask])))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3, 5))
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, :, 3:] += 100
        changed_v[:, :, 3:] -= 100
        changed, _ = scaled_attention(q, changed_k, changed_v, mask)
        torch.testing.assert_close(actual[:, :, :3], changed[:, :, :3])
        with self.assertRaises(ValueError):
            scaled_attention(q, k, v, torch.zeros_like(mask))
        with self.assertRaises(ValueError):
            scaled_attention(q, k, v, mask.float())

    def test_multihead_matches_reference_and_backpropagates(self):
        torch.manual_seed(4)
        custom = MultiHeadAttention(8, 2)
        reference = nn.MultiheadAttention(8, 2, dropout=0, batch_first=True)
        with torch.no_grad():
            reference.in_proj_weight.copy_(torch.cat([custom.q.weight, custom.k.weight, custom.v.weight]))
            reference.in_proj_bias.copy_(torch.cat([custom.q.bias, custom.k.bias, custom.v.bias]))
            reference.out_proj.load_state_dict(custom.output.state_dict())
        x = torch.randn(2, 5, 8, requires_grad=True)
        mask = torch.ones(5, 5, dtype=torch.bool).tril()
        actual, weights = custom(x, mask)
        expected, expected_weights = reference(x, x, x, attn_mask=~mask, average_attn_weights=False)
        self.assertEqual(weights.shape, (2, 2, 5, 5))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(weights, expected_weights)
        actual.square().mean().backward()
        for parameter in custom.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            MultiHeadAttention(7, 2)

    def test_positions_shapes_and_model_gradients(self):
        positions = PositionalEncoding(7, 9)
        encoded = positions(torch.zeros(2, 9, 7))
        self.assertEqual(encoded.shape, (2, 9, 7))
        self.assertFalse(torch.equal(encoded[:, 0], encoded[:, 1]))
        self.assertEqual(list(positions.parameters()), [])
        with self.assertRaises(ValueError):
            positions(torch.zeros(1, 10, 7))
        for kind in ("rnn", "lstm", "transformer"):
            with self.subTest(kind=kind):
                model = SequenceClassifier(kind, classes=4, width=8, heads=2, max_length=9)
                embedded = []

                def capture_embedding(module, inputs, output):
                    output.retain_grad()
                    embedded.append(output)

                model.embedding.register_forward_hook(capture_embedding)
                tokens = torch.randint(4, (3, 9))
                logits = model(tokens)
                self.assertEqual(logits.shape, (3, 4))
                F.cross_entropy(logits, tokens[:, 0]).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                # Last-position prediction has a differentiable path to the first input.
                self.assertGreater(embedded[0].grad[:, 0].abs().sum().item(), 0)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def small_config(self):
        return Config(classes=4, width=8, heads=2, batch_size=4, length=8,
                      steps=3, eval_size=12, log_every=2, lengths=(4, 8),
                      warmup=1, repeats=2, devices="cpu")

    def test_task_and_training_are_reproducible(self):
        x, y = recall_data(20, 8, 4, torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(x[:, 0], y))
        self.assertTrue((x[:, -1] != y).any())
        config = self.small_config()
        first, second = train_comparison(config), train_comparison(config)
        self.assertEqual(first, second)
        for model in first["models"]:
            self.assertEqual([h["step"] for h in model["history"]], [0, 2, 3])
            self.assertGreater(model["parameters"], 0)

    def test_end_to_end_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = run(self.small_config(), directory)
            self.assertEqual({p.name for p in output.iterdir()},
                             {"environment.json", "attention.json", "training.json", "benchmark.json", "report.md"})
            data = {p.stem: json.loads(p.read_text()) for p in output.glob("*.json")}
            self.assertEqual(data["environment"]["config"]["seed"], 7)
            self.assertEqual(data["environment"]["training_device"], "cpu")
            self.assertTrue(data["environment"]["source_sha256"])
            self.assertEqual(data["benchmark"]["devices_run"], ["cpu"])
            self.assertEqual(len(data["benchmark"]["rows"]), 6)
            for row in data["benchmark"]["rows"]:
                for operation in ("forward", "train_step"):
                    self.assertEqual(len(row[operation]["samples_ms"]), 2)
                    self.assertGreater(row[operation]["median_ms"], 0)
                memory = row["memory"]
                self.assertIsNone(memory["mps_current_tensor_bytes_snapshot"])
                expected = 4 * 2 * row["length"] ** 2 * 4 if row["model"] == "transformer" else 0
                self.assertEqual(memory["one_attention_matrix_bytes_estimate"], expected)
            report = (Path(directory) / "report.md").read_text()
            for name in ("environment", "attention", "training", "benchmark"):
                self.assertIn(f"[{name}.json]({name}.json)", report)
            self.assertIn("not run: --devices cpu", report)
            self.assertIn("not measured peaks", report)

    def test_invalid_config(self):
        for changes in ({"width": 7}, {"lengths": ()}, {"repeats": 0}, {"warmup": 0}, {"length": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Config(**changes).validate()


if __name__ == "__main__":
    unittest.main()
