#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Walk Analysis Script

Analyzes random walk parameters, embeddings, and similarity matrices.
Generates statistics and visualizations for understanding the learned
representations.

Features:
- Parameter analysis and display
- Embedding statistics (mean, std, L2 norms)
- View comparison (Tag vs Text)
- Similarity matrix analysis
- Visualization generation

Usage:
    python scripts/analyze_walks.py --base_dir ./tmp --output_dir ./analysis_outputs
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class RandomWalkAnalyzer:
    """Analyzes random walk results and generates visualizations."""

    def __init__(self, base_dir: str = 'tmp', output_dir: str = 'analysis_outputs'):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_walk_params(self) -> pd.DataFrame:
        """Load and display random walk parameters."""
        params_path = self.base_dir / 'rw_params.parquet'
        if not params_path.exists():
            print(f"Warning: {params_path} not found")
            return None

        params_df = pd.read_parquet(params_path)
        print("=" * 70)
        print("RANDOM WALK PARAMETERS")
        print("=" * 70)
        for col in params_df.columns:
            value = params_df[col].iloc[0]
            print(f"{col:30s}: {value}")
        print("=" * 70)
        return params_df

    def load_embeddings(self, view: str = 'tag') -> pd.DataFrame:
        """Load embeddings for a specific view."""
        emb_path = self.base_dir / f'Z_{view}.parquet'
        if emb_path.exists():
            return pd.read_parquet(emb_path)
        return None

    def load_all_epoch_embeddings(self, view: str = 'tag') -> dict:
        """Load embeddings from all epochs."""
        embeddings = {}

        # Load final
        final_path = self.base_dir / f'Z_{view}.parquet'
        if final_path.exists():
            embeddings['final'] = pd.read_parquet(final_path)

        # Load individual epochs
        for epoch in range(1, 10):
            epoch_path = self.base_dir / f'Z_{view}_epoch{epoch}.parquet'
            if epoch_path.exists():
                embeddings[f'epoch{epoch}'] = pd.read_parquet(epoch_path)

        return embeddings

    def analyze_embedding_statistics(self, embeddings_dict: dict,
                                     view: str = 'tag') -> pd.DataFrame:
        """Analyze embedding statistics across epochs."""
        print(f"\n{'=' * 70}")
        print(f"EMBEDDING STATISTICS - {view.upper()} VIEW")
        print("=" * 70)

        stats = []
        for name, emb_df in embeddings_dict.items():
            if emb_df is None or len(emb_df) == 0:
                continue

            # Get embedding columns (exclude doc_idx)
            emb_cols = [c for c in emb_df.columns if c.startswith('f')]
            emb_array = emb_df[emb_cols].values

            # Compute statistics
            stat = {
                'name': name,
                'shape': emb_array.shape,
                'mean': np.mean(emb_array),
                'std': np.std(emb_array),
                'min': np.min(emb_array),
                'max': np.max(emb_array),
                'l2_norm_mean': np.mean(np.linalg.norm(emb_array, axis=1)),
                'l2_norm_std': np.std(np.linalg.norm(emb_array, axis=1))
            }
            stats.append(stat)

            print(f"\n{name.upper()}:")
            print(f"  Shape: {stat['shape']}")
            print(f"  Value Range: [{stat['min']:.4f}, {stat['max']:.4f}]")
            print(f"  Mean: {stat['mean']:.4f}, Std: {stat['std']:.4f}")
            print(f"  L2 Norm - Mean: {stat['l2_norm_mean']:.4f}, "
                  f"Std: {stat['l2_norm_std']:.4f}")

        return pd.DataFrame(stats)

    def visualize_embedding_distribution(self, embeddings_dict: dict,
                                         view: str = 'tag') -> None:
        """Visualize embedding value distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for name, emb_df in embeddings_dict.items():
            if emb_df is None or len(emb_df) == 0:
                continue

            emb_cols = [c for c in emb_df.columns if c.startswith('f')]
            emb_array = emb_df[emb_cols].values

            # Plot 1: Value distribution
            axes[0, 0].hist(emb_array.flatten(), bins=100, alpha=0.6, label=name)

            # Plot 2: L2 norms
            norms = np.linalg.norm(emb_array, axis=1)
            axes[0, 1].hist(norms, bins=50, alpha=0.6, label=name)

        axes[0, 0].set_xlabel('Embedding Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'{view.upper()}: Value Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].set_xlabel('L2 Norm')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'{view.upper()}: L2 Norm Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Plot 3 & 4: Dimension-wise statistics (final only)
        if 'final' in embeddings_dict:
            emb_df = embeddings_dict['final']
            emb_cols = [c for c in emb_df.columns if c.startswith('f')]
            emb_array = emb_df[emb_cols].values

            dim_means = np.mean(emb_array, axis=0)
            dim_stds = np.std(emb_array, axis=0)

            axes[1, 0].plot(dim_means, label='Mean', linewidth=2)
            axes[1, 0].plot(dim_stds, label='Std', linewidth=2)
            axes[1, 0].set_xlabel('Dimension')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title(f'{view.upper()}: Dimension Statistics (Final)')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

            # 2D projection of first 100 docs
            sample = emb_array[:100, :2]
            axes[1, 1].scatter(sample[:, 0], sample[:, 1], alpha=0.6, s=30)
            axes[1, 1].set_xlabel('Dimension 0')
            axes[1, 1].set_ylabel('Dimension 1')
            axes[1, 1].set_title(f'{view.upper()}: 2D Projection (First 100 Docs)')
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        out_path = self.output_dir / f'{view}_embedding_distribution.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {out_path}")
        plt.close()

    def compare_views(self, tag_emb: pd.DataFrame, text_emb: pd.DataFrame) -> None:
        """Compare tag and text view embeddings."""
        print(f"\n{'=' * 70}")
        print("VIEW COMPARISON")
        print("=" * 70)

        # Get embedding columns
        tag_cols = [c for c in tag_emb.columns if c.startswith('f')]
        text_cols = [c for c in text_emb.columns if c.startswith('f')]
        tag_array = tag_emb[tag_cols].values
        text_array = text_emb[text_cols].values

        print(f"\nTag View Shape: {tag_array.shape}")
        print(f"Text View Shape: {text_array.shape}")

        print(f"\nTag View - Mean: {np.mean(tag_array):.4f}, "
              f"Std: {np.std(tag_array):.4f}")
        print(f"Text View - Mean: {np.mean(text_array):.4f}, "
              f"Std: {np.std(text_array):.4f}")

        # L2 norms
        tag_norms = np.linalg.norm(tag_array, axis=1)
        text_norms = np.linalg.norm(text_array, axis=1)

        print(f"\nTag View L2 Norms - Mean: {np.mean(tag_norms):.4f}, "
              f"Std: {np.std(tag_norms):.4f}")
        print(f"Text View L2 Norms - Mean: {np.mean(text_norms):.4f}, "
              f"Std: {np.std(text_norms):.4f}")

        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].hist(tag_array.flatten(), bins=100, alpha=0.6,
                        label='Tag View', color='blue')
        axes[0, 0].hist(text_array.flatten(), bins=100, alpha=0.6,
                        label='Text View', color='red')
        axes[0, 0].set_xlabel('Embedding Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Value Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].hist(tag_norms, bins=50, alpha=0.6, label='Tag View', color='blue')
        axes[0, 1].hist(text_norms, bins=50, alpha=0.6, label='Text View', color='red')
        axes[0, 1].set_xlabel('L2 Norm')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('L2 Norm Distribution Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(np.mean(tag_array, axis=0), label='Tag View', linewidth=2)
        axes[1, 0].plot(np.mean(text_array, axis=0), label='Text View', linewidth=2)
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].set_title('Dimension-wise Mean Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(np.std(tag_array, axis=0), label='Tag View', linewidth=2)
        axes[1, 1].plot(np.std(text_array, axis=0), label='Text View', linewidth=2)
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Std Deviation')
        axes[1, 1].set_title('Dimension-wise Std Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        out_path = self.output_dir / 'view_comparison.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {out_path}")
        plt.close()

    def run_full_analysis(self) -> None:
        """Run complete random walk analysis."""
        print("\n" + "=" * 70)
        print("RANDOM WALK ANALYSIS - COMPREHENSIVE REPORT")
        print("=" * 70)

        # 1. Load and display parameters
        self.load_walk_params()

        # 2. Analyze Tag View
        print("\n\n>>> ANALYZING TAG VIEW <<<")
        tag_embeddings = self.load_all_epoch_embeddings('tag')
        if tag_embeddings:
            self.analyze_embedding_statistics(tag_embeddings, 'tag')
            self.visualize_embedding_distribution(tag_embeddings, 'tag')

        # 3. Analyze Text View
        print("\n\n>>> ANALYZING TEXT VIEW <<<")
        text_embeddings = self.load_all_epoch_embeddings('text')
        if text_embeddings:
            self.analyze_embedding_statistics(text_embeddings, 'text')
            self.visualize_embedding_distribution(text_embeddings, 'text')

        # 4. Compare views
        if tag_embeddings and text_embeddings:
            if 'final' in tag_embeddings and 'final' in text_embeddings:
                print("\n\n>>> COMPARING TAG AND TEXT VIEWS <<<")
                self.compare_views(tag_embeddings['final'], text_embeddings['final'])

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print(f"All outputs saved to: {self.output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Random Walk Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--base_dir", type=str, default="./tmp",
                        help="Directory containing embeddings and parameters")
    parser.add_argument("--output_dir", type=str, default="./analysis_outputs",
                        help="Directory for output visualizations")
    args = parser.parse_args()

    analyzer = RandomWalkAnalyzer(
        base_dir=args.base_dir,
        output_dir=args.output_dir
    )
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
