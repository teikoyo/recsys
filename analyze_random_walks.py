#!/usr/bin/env python3
"""
Random Walk Analysis and Visualization Script
Analyzes random walk parameters, embeddings, and resulting similarity matrices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class RandomWalkAnalyzer:
    """Analyzes random walk results and visualizations"""

    def __init__(self, base_dir='tmp'):
        self.base_dir = Path(base_dir)
        self.output_dir = Path('random_walk_analysis_outputs')
        self.output_dir.mkdir(exist_ok=True)

    def load_walk_params(self):
        """Load random walk parameters"""
        params_path = self.base_dir / 'rw_params.parquet'
        if params_path.exists():
            params_df = pd.read_parquet(params_path)
            print("=" * 70)
            print("RANDOM WALK PARAMETERS")
            print("=" * 70)
            for col in params_df.columns:
                value = params_df[col].iloc[0]
                print(f"{col:30s}: {value}")
            print("=" * 70)
            return params_df
        else:
            print(f"Warning: {params_path} not found")
            return None

    def load_embeddings(self, view='tag'):
        """Load embeddings for a specific view"""
        # Try final embedding first
        emb_path = self.base_dir / f'Z_{view}.parquet'
        if emb_path.exists():
            return pd.read_parquet(emb_path)
        return None

    def load_all_epoch_embeddings(self, view='tag'):
        """Load embeddings from all epochs"""
        embeddings = {}

        # Load final
        final_path = self.base_dir / f'Z_{view}.parquet'
        if final_path.exists():
            embeddings['final'] = pd.read_parquet(final_path)

        # Load individual epochs
        for epoch in range(1, 10):  # Try up to epoch 9
            epoch_path = self.base_dir / f'Z_{view}_epoch{epoch}.parquet'
            if epoch_path.exists():
                embeddings[f'epoch{epoch}'] = pd.read_parquet(epoch_path)

        return embeddings

    def analyze_embedding_statistics(self, embeddings_dict, view='tag'):
        """Analyze embedding statistics across epochs"""
        print(f"\n{'=' * 70}")
        print(f"EMBEDDING STATISTICS - {view.upper()} VIEW")
        print("=" * 70)

        stats = []
        for name, emb_df in embeddings_dict.items():
            if emb_df is None or len(emb_df) == 0:
                continue

            # Convert to numpy array
            if isinstance(emb_df, pd.DataFrame):
                emb_array = emb_df.values
            else:
                emb_array = emb_df

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
            print(f"  L2 Norm - Mean: {stat['l2_norm_mean']:.4f}, Std: {stat['l2_norm_std']:.4f}")

        return pd.DataFrame(stats)

    def visualize_embedding_distribution(self, embeddings_dict, view='tag'):
        """Visualize embedding value distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Flatten all embeddings for overall distribution
        for name, emb_df in embeddings_dict.items():
            if emb_df is None or len(emb_df) == 0:
                continue

            if isinstance(emb_df, pd.DataFrame):
                emb_array = emb_df.values
            else:
                emb_array = emb_df

            # Plot 1: Value distribution histogram
            axes[0, 0].hist(emb_array.flatten(), bins=100, alpha=0.6, label=name)

        axes[0, 0].set_xlabel('Embedding Value', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title(f'{view.upper()} View: Embedding Value Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Plot 2: L2 norms across epochs
        for name, emb_df in embeddings_dict.items():
            if emb_df is None or len(emb_df) == 0:
                continue

            if isinstance(emb_df, pd.DataFrame):
                emb_array = emb_df.values
            else:
                emb_array = emb_df

            norms = np.linalg.norm(emb_array, axis=1)
            axes[0, 1].hist(norms, bins=50, alpha=0.6, label=name)

        axes[0, 1].set_xlabel('L2 Norm', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title(f'{view.upper()} View: L2 Norm Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Plot 3: Dimension-wise statistics (using final embedding)
        if 'final' in embeddings_dict:
            emb_df = embeddings_dict['final']
            if isinstance(emb_df, pd.DataFrame):
                emb_array = emb_df.values
            else:
                emb_array = emb_df

            dim_means = np.mean(emb_array, axis=0)
            dim_stds = np.std(emb_array, axis=0)

            axes[1, 0].plot(dim_means, label='Mean', linewidth=2)
            axes[1, 0].plot(dim_stds, label='Std', linewidth=2)
            axes[1, 0].set_xlabel('Dimension', fontsize=11)
            axes[1, 0].set_ylabel('Value', fontsize=11)
            axes[1, 0].set_title(f'{view.upper()} View: Dimension Statistics (Final)', fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

        # Plot 4: Sample embeddings (first 100 documents, first 2 dimensions for visualization)
        if 'final' in embeddings_dict:
            emb_df = embeddings_dict['final']
            if isinstance(emb_df, pd.DataFrame):
                emb_array = emb_df.values
            else:
                emb_array = emb_df

            # Take first 100 documents and first 2 dimensions
            sample = emb_array[:100, :2]
            axes[1, 1].scatter(sample[:, 0], sample[:, 1], alpha=0.6, s=30)
            axes[1, 1].set_xlabel('Dimension 0', fontsize=11)
            axes[1, 1].set_ylabel('Dimension 1', fontsize=11)
            axes[1, 1].set_title(f'{view.upper()} View: 2D Projection (First 100 Docs)', fontsize=12, fontweight='bold')
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'embedding_distribution_{view}.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / f'embedding_distribution_{view}.png'}")
        plt.close()

    def load_similarity_matrix(self, matrix_name='S_tag_symrow_k50'):
        """Load similarity matrix from partitioned parquet files"""
        manifest_path = self.base_dir / f'{matrix_name}_manifest.json'

        if not manifest_path.exists():
            print(f"Warning: {manifest_path} not found")
            return None, None

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        print(f"\nLoading {matrix_name}...")
        print(f"  Shape: {manifest['shape']}")
        print(f"  Non-zeros: {manifest['nnz']:,}")
        print(f"  Sparsity: {(1 - manifest['nnz'] / (manifest['shape'][0] * manifest['shape'][1])) * 100:.4f}%")

        # Load all parts
        all_rows = []
        all_cols = []
        all_data = []

        for part_info in manifest['parts']:
            part_path = self.base_dir / part_info['path']
            if part_path.exists():
                df = pd.read_parquet(part_path)
                all_rows.extend(df['row'].values)
                all_cols.extend(df['col'].values)
                all_data.extend(df['val'].values)

        # Create sparse matrix
        shape = tuple(manifest['shape'])
        sparse_matrix = csr_matrix((all_data, (all_rows, all_cols)), shape=shape)

        return sparse_matrix, manifest

    def analyze_similarity_matrix(self, sparse_matrix, matrix_name):
        """Analyze similarity matrix statistics"""
        print(f"\n{'=' * 70}")
        print(f"SIMILARITY MATRIX ANALYSIS - {matrix_name}")
        print("=" * 70)

        # Basic statistics
        nnz = sparse_matrix.nnz
        shape = sparse_matrix.shape
        sparsity = (1 - nnz / (shape[0] * shape[1])) * 100

        print(f"Shape: {shape}")
        print(f"Non-zeros: {nnz:,}")
        print(f"Sparsity: {sparsity:.4f}%")

        # Degree distribution
        degrees = np.array(sparse_matrix.sum(axis=1)).flatten()
        nonzero_degrees = degrees[degrees > 0]

        print(f"\nDegree Statistics:")
        print(f"  Min degree: {np.min(nonzero_degrees):.4f}")
        print(f"  Max degree: {np.max(nonzero_degrees):.4f}")
        print(f"  Mean degree: {np.mean(nonzero_degrees):.4f}")
        print(f"  Median degree: {np.median(nonzero_degrees):.4f}")
        print(f"  Std degree: {np.std(nonzero_degrees):.4f}")

        # Value statistics
        values = sparse_matrix.data
        print(f"\nValue Statistics:")
        print(f"  Min value: {np.min(values):.6f}")
        print(f"  Max value: {np.max(values):.6f}")
        print(f"  Mean value: {np.mean(values):.6f}")
        print(f"  Median value: {np.median(values):.6f}")
        print(f"  Std value: {np.std(values):.6f}")

        # Edges per row
        edges_per_row = np.diff(sparse_matrix.indptr)
        nonzero_edges = edges_per_row[edges_per_row > 0]

        print(f"\nEdges Per Row:")
        print(f"  Min edges: {np.min(nonzero_edges)}")
        print(f"  Max edges: {np.max(nonzero_edges)}")
        print(f"  Mean edges: {np.mean(nonzero_edges):.2f}")
        print(f"  Median edges: {np.median(nonzero_edges):.0f}")

        return {
            'degrees': degrees,
            'values': values,
            'edges_per_row': edges_per_row
        }

    def visualize_similarity_matrix(self, stats, matrix_name):
        """Visualize similarity matrix statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Degree distribution
        degrees = stats['degrees']
        nonzero_degrees = degrees[degrees > 0]

        axes[0, 0].hist(nonzero_degrees, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Degree (Row Sum)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title(f'{matrix_name}: Degree Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)

        # Plot 2: Value distribution
        values = stats['values']
        axes[0, 1].hist(values, bins=100, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Similarity Value', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title(f'{matrix_name}: Value Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)

        # Plot 3: Edges per row distribution
        edges_per_row = stats['edges_per_row']
        nonzero_edges = edges_per_row[edges_per_row > 0]

        axes[1, 0].hist(nonzero_edges, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Number of Edges', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title(f'{matrix_name}: Edges Per Row Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        # Plot 4: Log-log degree distribution (power law check)
        degree_counts = np.bincount(nonzero_edges.astype(int))
        nonzero_counts = np.where(degree_counts > 0)[0]

        axes[1, 1].loglog(nonzero_counts, degree_counts[nonzero_counts], 'o-', alpha=0.7)
        axes[1, 1].set_xlabel('Edges Per Row (log scale)', fontsize=11)
        axes[1, 1].set_ylabel('Frequency (log scale)', fontsize=11)
        axes[1, 1].set_title(f'{matrix_name}: Log-Log Degree Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3, which='both')

        plt.tight_layout()
        output_path = self.output_dir / f'similarity_matrix_{matrix_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")
        plt.close()

    def compare_views(self, tag_emb, text_emb):
        """Compare tag and text view embeddings"""
        print(f"\n{'=' * 70}")
        print("VIEW COMPARISON")
        print("=" * 70)

        if isinstance(tag_emb, pd.DataFrame):
            tag_array = tag_emb.values
        else:
            tag_array = tag_emb

        if isinstance(text_emb, pd.DataFrame):
            text_array = text_emb.values
        else:
            text_array = text_emb

        # Compare shapes
        print(f"\nTag View Shape: {tag_array.shape}")
        print(f"Text View Shape: {text_array.shape}")

        # Compare statistics
        print(f"\nTag View - Mean: {np.mean(tag_array):.4f}, Std: {np.std(tag_array):.4f}")
        print(f"Text View - Mean: {np.mean(text_array):.4f}, Std: {np.std(text_array):.4f}")

        # L2 norms
        tag_norms = np.linalg.norm(tag_array, axis=1)
        text_norms = np.linalg.norm(text_array, axis=1)

        print(f"\nTag View L2 Norms - Mean: {np.mean(tag_norms):.4f}, Std: {np.std(tag_norms):.4f}")
        print(f"Text View L2 Norms - Mean: {np.mean(text_norms):.4f}, Std: {np.std(text_norms):.4f}")

        # Visualize comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Plot 1: Value distribution comparison
        axes[0, 0].hist(tag_array.flatten(), bins=100, alpha=0.6, label='Tag View', color='blue')
        axes[0, 0].hist(text_array.flatten(), bins=100, alpha=0.6, label='Text View', color='red')
        axes[0, 0].set_xlabel('Embedding Value', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Value Distribution Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Plot 2: L2 norm comparison
        axes[0, 1].hist(tag_norms, bins=50, alpha=0.6, label='Tag View', color='blue')
        axes[0, 1].hist(text_norms, bins=50, alpha=0.6, label='Text View', color='red')
        axes[0, 1].set_xlabel('L2 Norm', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('L2 Norm Distribution Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Plot 3: Dimension-wise mean comparison
        tag_dim_means = np.mean(tag_array, axis=0)
        text_dim_means = np.mean(text_array, axis=0)

        axes[1, 0].plot(tag_dim_means, label='Tag View', linewidth=2, alpha=0.7)
        axes[1, 0].plot(text_dim_means, label='Text View', linewidth=2, alpha=0.7)
        axes[1, 0].set_xlabel('Dimension', fontsize=11)
        axes[1, 0].set_ylabel('Mean Value', fontsize=11)
        axes[1, 0].set_title('Dimension-wise Mean Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Plot 4: Dimension-wise std comparison
        tag_dim_stds = np.std(tag_array, axis=0)
        text_dim_stds = np.std(text_array, axis=0)

        axes[1, 1].plot(tag_dim_stds, label='Tag View', linewidth=2, alpha=0.7)
        axes[1, 1].plot(text_dim_stds, label='Text View', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Dimension', fontsize=11)
        axes[1, 1].set_ylabel('Std Deviation', fontsize=11)
        axes[1, 1].set_title('Dimension-wise Std Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'view_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'view_comparison.png'}")
        plt.close()

    def run_full_analysis(self):
        """Run complete random walk analysis"""
        print("\n" + "=" * 70)
        print("RANDOM WALK ANALYSIS - COMPREHENSIVE REPORT")
        print("=" * 70)

        # 1. Load and display parameters
        params_df = self.load_walk_params()

        # 2. Analyze Tag View
        print("\n\n>>> ANALYZING TAG VIEW <<<")
        tag_embeddings = self.load_all_epoch_embeddings('tag')
        if tag_embeddings:
            tag_stats = self.analyze_embedding_statistics(tag_embeddings, 'tag')
            self.visualize_embedding_distribution(tag_embeddings, 'tag')

        # 3. Analyze Text View
        print("\n\n>>> ANALYZING TEXT VIEW <<<")
        text_embeddings = self.load_all_epoch_embeddings('text')
        if text_embeddings:
            text_stats = self.analyze_embedding_statistics(text_embeddings, 'text')
            self.visualize_embedding_distribution(text_embeddings, 'text')

        # 4. Compare views
        if tag_embeddings and text_embeddings:
            if 'final' in tag_embeddings and 'final' in text_embeddings:
                print("\n\n>>> COMPARING TAG AND TEXT VIEWS <<<")
                self.compare_views(tag_embeddings['final'], text_embeddings['final'])

        # 5. Analyze similarity matrices
        print("\n\n>>> ANALYZING SIMILARITY MATRICES <<<")

        # Tag similarity matrix
        tag_sim, tag_manifest = self.load_similarity_matrix('S_tag_symrow_k50')
        if tag_sim is not None:
            tag_sim_stats = self.analyze_similarity_matrix(tag_sim, 'S_tag_symrow_k50')
            self.visualize_similarity_matrix(tag_sim_stats, 'S_tag_symrow_k50')

        # Text similarity matrix
        text_sim, text_manifest = self.load_similarity_matrix('S_text_symrow_k50')
        if text_sim is not None:
            text_sim_stats = self.analyze_similarity_matrix(text_sim, 'S_text_symrow_k50')
            self.visualize_similarity_matrix(text_sim_stats, 'S_text_symrow_k50')

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print(f"All outputs saved to: {self.output_dir}")
        print("=" * 70)


def main():
    """Main execution function"""
    analyzer = RandomWalkAnalyzer(base_dir='tmp')
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
