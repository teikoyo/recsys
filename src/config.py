"""Dataclass configuration for WS-SGNS training and evaluation."""

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from .constants import NS_POWER_DEFAULT, TOP_K_DEFAULT, CORPUS_SIZE_DEFAULT


@dataclass
class ViewParams:
    """Per-view training parameters."""
    window: int = 5
    keep_prob: float = 1.0
    forward_only: bool = False
    ctx_cap: int = 0
    batch_pairs: int = 204800
    max_pairs: int = 20_000_000
    max_sents: Optional[int] = None


@dataclass
class TrainConfig:
    """Training configuration for WS-SGNS."""
    tmp_dir: str = "./tmp"
    views: List[str] = field(default_factory=lambda: ["tag", "text"])
    dim: int = 256
    epochs: int = 4
    neg: int = 10
    lr: float = 0.025
    ns_power: float = NS_POWER_DEFAULT
    seed: int = 2025
    accum: int = 1
    optimizer: str = "sgd"
    sparse: bool = False
    amp: bool = True
    tf32: bool = True
    tag: ViewParams = field(default_factory=ViewParams)
    text: ViewParams = field(default_factory=lambda: ViewParams(
        window=4, keep_prob=0.35, forward_only=True, ctx_cap=4
    ))
    log_every: int = 200
    eval_samples_per_view: int = 3
    eval_topk: int = 5
    save_epoch_emb: bool = True
    emb_dtype: str = "float32"
    log_level: str = "INFO"

    @classmethod
    def from_args(cls, args=None) -> "TrainConfig":
        """Build config from argparse args (backward-compatible CLI bridge).

        Args:
            args: Pre-parsed args or None to parse sys.argv.

        Returns:
            Populated TrainConfig instance.
        """
        p = argparse.ArgumentParser(
            description="WS-SGNS Trainer",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Core settings
        p.add_argument("--tmp_dir", type=str, default="./tmp")
        p.add_argument("--views", type=str, default="tag,text")
        p.add_argument("--epochs", type=int, default=4)
        p.add_argument("--dim", type=int, default=256)
        p.add_argument("--neg", type=int, default=10)
        p.add_argument("--lr", type=float, default=0.025)
        p.add_argument("--ns_power", type=float, default=NS_POWER_DEFAULT)
        p.add_argument("--seed", type=int, default=2025)
        p.add_argument("--accum", type=int, default=1)

        # Optimizer
        p.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "sparse_adam", "adagrad"])
        p.add_argument("--sparse", type=lambda s: s.lower() in ["true", "1", "yes"],
                        default=False)

        # Precision
        p.add_argument("--amp", type=lambda s: s.lower() in ["true", "1", "yes"],
                        default=True)
        p.add_argument("--tf32", type=lambda s: s.lower() in ["true", "1", "yes"],
                        default=True)

        # Tag view
        p.add_argument("--window_tag", type=int, default=5)
        p.add_argument("--keep_prob_tag", type=float, default=1.0)
        p.add_argument("--forward_only_tag",
                        type=lambda s: s.lower() in ["true", "1", "yes"],
                        default=False)
        p.add_argument("--ctx_cap_tag", type=int, default=0)
        p.add_argument("--batch_pairs_tag", type=int, default=204800)
        p.add_argument("--max_pairs_tag", type=int, default=20_000_000)
        p.add_argument("--max_sents_tag", type=int, default=None)

        # Text view
        p.add_argument("--window_text", type=int, default=4)
        p.add_argument("--keep_prob_text", type=float, default=0.35)
        p.add_argument("--forward_only_text",
                        type=lambda s: s.lower() in ["true", "1", "yes"],
                        default=True)
        p.add_argument("--ctx_cap_text", type=int, default=4)
        p.add_argument("--batch_pairs_text", type=int, default=204800)
        p.add_argument("--max_pairs_text", type=int, default=20_000_000)
        p.add_argument("--max_sents_text", type=int, default=None)

        # Logging and checkpointing
        p.add_argument("--log_every", type=int, default=200)
        p.add_argument("--eval_samples_per_view", type=int, default=3)
        p.add_argument("--eval_topk", type=int, default=5)
        p.add_argument("--save_epoch_emb",
                        type=lambda s: s.lower() in ["true", "1", "yes"],
                        default=True)
        p.add_argument("--emb_dtype", type=str, default="float32",
                        choices=["float32", "float16"])
        p.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

        # Config file
        p.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")

        a = p.parse_args(args)

        # If config file provided, load it
        if a.config:
            return cls.from_json(a.config)

        views = [v.strip().lower() for v in a.views.split(",")]

        return cls(
            tmp_dir=a.tmp_dir,
            views=views,
            dim=a.dim,
            epochs=a.epochs,
            neg=a.neg,
            lr=a.lr,
            ns_power=a.ns_power,
            seed=a.seed,
            accum=a.accum,
            optimizer=a.optimizer,
            sparse=a.sparse,
            amp=a.amp,
            tf32=a.tf32,
            tag=ViewParams(
                window=a.window_tag,
                keep_prob=a.keep_prob_tag,
                forward_only=a.forward_only_tag,
                ctx_cap=a.ctx_cap_tag,
                batch_pairs=a.batch_pairs_tag,
                max_pairs=a.max_pairs_tag,
                max_sents=a.max_sents_tag,
            ),
            text=ViewParams(
                window=a.window_text,
                keep_prob=a.keep_prob_text,
                forward_only=a.forward_only_text,
                ctx_cap=a.ctx_cap_text,
                batch_pairs=a.batch_pairs_text,
                max_pairs=a.max_pairs_text,
                max_sents=a.max_sents_text,
            ),
            log_every=a.log_every,
            eval_samples_per_view=a.eval_samples_per_view,
            eval_topk=a.eval_topk,
            save_epoch_emb=a.save_epoch_emb,
            emb_dtype=a.emb_dtype,
            log_level=a.log_level,
        )

    @classmethod
    def from_json(cls, path) -> "TrainConfig":
        """Load config from a JSON file.

        Args:
            path: Path to JSON config file.

        Returns:
            Populated TrainConfig instance.
        """
        with open(path) as f:
            data = json.load(f)

        tag_data = data.pop("tag", {})
        text_data = data.pop("text", {})

        cfg = cls(**data)
        if tag_data:
            cfg.tag = ViewParams(**tag_data)
        if text_data:
            cfg.text = ViewParams(**text_data)
        return cfg

    def to_dict(self) -> dict:
        """Convert to a plain dict."""
        return asdict(self)

    def save(self, path) -> None:
        """Save config to a JSON file.

        Args:
            path: Output file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def view_params(self, view_name: str) -> ViewParams:
        """Get ViewParams for a named view.

        Args:
            view_name: "tag" or "text".

        Returns:
            The corresponding ViewParams.
        """
        if view_name == "tag":
            return self.tag
        return self.text


@dataclass
class ContentConfig:
    """Configuration for content view pipeline."""
    max_rows: int = 1024
    max_cols: int = 60
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    k_sim: int = TOP_K_DEFAULT
    n_total: int = CORPUS_SIZE_DEFAULT
    device: str = "auto"
    seed: int = 42


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    k_eval: int = 20
    k_sim: int = TOP_K_DEFAULT
    n_total: int = CORPUS_SIZE_DEFAULT
    seed: int = 42
