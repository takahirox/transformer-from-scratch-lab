"""Versioned CPU checkpoint for the existing recall Transformer."""

from pathlib import Path

import torch

from .models import SequenceClassifier


def save_transformer(model, config, path):
    settings = {"kind": "transformer", "classes": config["classes"],
                "width": config["width"], "heads": config["heads"],
                "max_length": max(config["length"], *config["lengths"])}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "model_settings": settings,
                "training_config": config,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)


def load_transformer(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    settings = checkpoint["model_settings"]
    if settings["kind"] != "transformer":
        raise ValueError("browser export requires the recall Transformer")
    model = SequenceClassifier(**settings)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), checkpoint
