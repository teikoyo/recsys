import pytest
import numpy as np
import torch
from scipy import sparse

@pytest.fixture
def small_csr():
    """A small 5x5 CSR matrix for testing."""
    rows = [0,0,1,1,2,3,3,4]
    cols = [1,2,0,3,4,0,4,2]
    vals = [0.5,0.3,0.4,0.6,0.8,0.2,0.7,0.9]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(5, 5))

@pytest.fixture
def cpu_device():
    return torch.device("cpu")
