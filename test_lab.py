"""Fast CPU regression tests for the educational lab."""
import unittest
import torch
from torch.nn import functional as F
from lab import attention, attention_experiment, MultiHeadAttention, Position, Block, Classifier, data


class LabTests(unittest.TestCase):
    def test_reference_attention(self):
        result = attention_experiment()
        self.assertTrue(result["matches_pytorch"])
        for total in result["row_sums"]:
            self.assertAlmostEqual(total, 1, places=5)

    def test_attention_mask(self):
        q = k = v = torch.eye(3)
        mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
        out, weights = attention(q, k, v, mask)
        self.assertEqual(tuple(out.shape), (3, 3))
        self.assertTrue(torch.all(weights[~mask] == 0))

    def test_shapes_and_gradients(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 16)
        attn = MultiHeadAttention(16, 2)
        y, weights = attn(x)
        self.assertEqual(tuple(y.shape), (2, 5, 16))
        self.assertEqual(tuple(weights.shape), (2, 2, 5, 5))
        block = Block(16, 2)
        out = block(Position(16)(x))
        self.assertEqual(tuple(out.shape), (2, 5, 16))
        out.sum().backward()
        self.assertTrue(all(p.grad is not None for p in block.parameters()))

    def test_models_train(self):
        x, y = data(4, 7, torch.device("cpu"), torch.Generator().manual_seed(3))
        for kind in ("rnn", "lstm", "transformer"):
            model = Classifier(kind)
            pred = model(x)
            self.assertEqual(tuple(pred.shape), (4, 2))
            F.cross_entropy(pred, y).backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
