#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SGNS (Skip-gram with Negative Sampling) Model

Implements the neural network model for learning document embeddings
using the Skip-gram objective with negative sampling loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGNS(nn.Module):
    """
    Skip-gram with Negative Sampling model.

    Learns document embeddings by predicting context documents from
    center documents in random walk sentences. Uses negative sampling
    for efficient training.

    The loss function is:
        L = -log σ(u_c · v_o) - Σ_{k} log σ(-u_c · v_k)

    where:
        - u_c is the input embedding of center document
        - v_o is the output embedding of positive context
        - v_k are output embeddings of negative samples

    Args:
        vocab_size: Number of documents (vocabulary size)
        dim: Embedding dimension
        sparse: Whether to use sparse gradients (for SparseAdam optimizer)

    Example:
        >>> model = SGNS(vocab_size=100000, dim=256)
        >>> loss = model(center_ids, context_ids, neg_ids)
    """

    def __init__(self, vocab_size: int, dim: int, sparse: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        # Input embeddings (center documents)
        self.in_emb = nn.Embedding(vocab_size, dim, sparse=sparse)
        # Output embeddings (context/negative documents)
        self.out_emb = nn.Embedding(vocab_size, dim, sparse=sparse)

        # Initialize with small uniform values
        nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.uniform_(self.out_emb.weight, -0.5 / dim, 0.5 / dim)

    def forward(
        self,
        center: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SGNS loss.

        Args:
            center: Center document indices [B]
            pos: Positive context indices [B]
            neg: Negative sample indices [B, K]

        Returns:
            Scalar loss tensor [1]
        """
        # Get embeddings
        v = self.in_emb(center)          # [B, d]
        u = self.out_emb(pos)            # [B, d]
        neg_u = self.out_emb(neg)        # [B, K, d]

        # Positive score: v · u for each pair
        pos_logit = torch.sum(v * u, dim=1)  # [B]

        # Negative scores: v · neg_u for each negative
        neg_logit = torch.einsum("bd,bkd->bk", v, neg_u)  # [B, K]

        # Loss using softplus for numerical stability
        # softplus(-x) = log(1 + exp(-x)) ≈ -log(sigmoid(x))
        pos_loss = F.softplus(-pos_logit)
        neg_loss = F.softplus(neg_logit).sum(dim=1)

        # Mean over batch, return as [1] tensor
        return (pos_loss + neg_loss).mean().unsqueeze(0)

    def get_embeddings(self, normalize: bool = True) -> torch.Tensor:
        """
        Get the learned document embeddings.

        Args:
            normalize: Whether to L2-normalize the embeddings

        Returns:
            Embedding matrix [vocab_size, dim]
        """
        emb = self.in_emb.weight.detach()
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb
