"""Tests for src.config -- TrainConfig, ViewParams, serialization."""

import json
import tempfile
import os

import pytest

from src.config import TrainConfig, ViewParams


# ---------- Defaults ----------

def test_train_config_defaults():
    """Default TrainConfig values match expected."""
    cfg = TrainConfig()
    assert cfg.dim == 256
    assert cfg.epochs == 4
    assert cfg.neg == 10
    assert cfg.lr == 0.025
    assert cfg.seed == 2025
    assert cfg.optimizer == "sgd"
    assert cfg.views == ["tag", "text"]


# ---------- CLI parsing ----------

def test_train_config_from_args():
    """from_args correctly parses CLI-style arguments."""
    cfg = TrainConfig.from_args(["--dim", "128", "--epochs", "2", "--lr", "0.01"])
    assert cfg.dim == 128
    assert cfg.epochs == 2
    assert cfg.lr == 0.01


# ---------- Serialization ----------

def test_train_config_to_dict():
    """to_dict produces a plain dict with all fields."""
    cfg = TrainConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["dim"] == 256
    assert "tag" in d
    assert "text" in d
    assert isinstance(d["tag"], dict)


def test_train_config_json_roundtrip():
    """save() then from_json() produces equivalent config."""
    cfg = TrainConfig(dim=64, epochs=2, lr=0.001)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        cfg.save(tmp_path)
        cfg2 = TrainConfig.from_json(tmp_path)
        assert cfg2.dim == 64
        assert cfg2.epochs == 2
        assert cfg2.lr == 0.001
    finally:
        os.unlink(tmp_path)


# ---------- ViewParams ----------

def test_view_params_tag():
    """view_params('tag') returns the tag ViewParams."""
    cfg = TrainConfig()
    vp = cfg.view_params("tag")
    assert isinstance(vp, ViewParams)
    assert vp.window == 5  # tag default


def test_view_params_text():
    """view_params('text') returns the text ViewParams."""
    cfg = TrainConfig()
    vp = cfg.view_params("text")
    assert isinstance(vp, ViewParams)
    assert vp.window == 4  # text default
    assert vp.forward_only is True
