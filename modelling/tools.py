import numpy as np
from numba import njit

BIN_WIDTH = 0.5
MAX_RMSD_VAL = 8
N_RMSD_CLASSES = int(MAX_RMSD_VAL / BIN_WIDTH)
RMSD_BINS = np.arange(0, MAX_RMSD_VAL + 1e-10, BIN_WIDTH)
BIN_MIDPTS = (RMSD_BINS[0:-1] + RMSD_BINS[1:]) / 2
TOP_K_VALUES = [5,10,30]
TARGET_GRAPH_DIR = 'data/graphs'

def get_expected_rmsd(pdist: np.ndarray):
    expected_rmsd = np.sum(pdist * BIN_MIDPTS, axis = 1)
    return expected_rmsd


def convert_to_one_hot(arr: np.ndarray):
    max_indices = np.argmax(arr, axis=1)  # Find the indices of the maximal values in each row
    one_hot = np.zeros_like(arr)  # Create an array of zeros with the same shape as the input array
    one_hot[np.arange(arr.shape[0]), max_indices] = 1  # Set the maximal value indices to 1
    return one_hot

from spektral.data import Dataset, Graph
import os, time, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Callable, List, Dict, Any
from scipy.sparse import csc_matrix
from modelling.constants import TARGET_GRAPH_DIR, N_RMSD_CLASSES, RMSD_BINS


def adjacency_matrix_map(n_nodes_min: int, n_nodes_max: int, self_edge: Optional[bool] = False, n_neighbors: Optional[int] = 1, fully_connected: Optional[bool] = False) -> dict:
    """
    Generate adjacency matrices for a range of node numbers, representing the connectivity between nodes in a graph.

    Args:
        n_nodes_min (int): The minimum number of nodes.
        n_nodes_max (int): The maximum number of nodes.
        self_edge (bool, optional): Whether to connect each node to itself. Defaults to False.
        n_neighbors (int, optional): The number of neighbors each node should be connected to. Defaults to 1.
        fully_connected (bool, optional): Whether to generate adjacency matrices for fully connected graphs (except for self-connections). Defaults to False.

    Returns:
        dict: A dictionary mapping the number of nodes to their corresponding adjacency matrices.

    """

    A_map = {}  # Dictionary to store the adjacency matrices

    for n_nodes in range(n_nodes_min, n_nodes_max + 1):
        
        if fully_connected:
            A = np.ones(shape=(n_nodes, n_nodes)) - np.eye(n_nodes)  # Create a fully connected adjacency matrix
        else:
            A = np.zeros(shape=(n_nodes, n_nodes))  # Create an empty adjacency matrix
            try:
                for i in range(n_nodes):
                    for j in range(0, n_neighbors):
                        if i > 0:
                            A[i, max(0, i - j)] = 1  # Connect the current node to its left neighbor

                        if self_edge:
                            A[i, i] = 1  # Optionally connect the current node to itself

                        if i < n_nodes - 1:
                            A[i, min(n_nodes - 1, i + j)] = 1  # Connect the current node to its right neighbor
            except Exception as e:
                print(e)

        A_map[n_nodes] = csc_matrix(A)  # Store the adjacency matrix in the dictionary

    return A_map


def rmsd_to_y_onehot(rmsd: np.float32) -> int:
    """
    Convert the root mean square deviation (RMSD) value to a one-hot encoded label.

    Args:
        rmsd (np.float32): The root mean square deviation value to be converted.

    Returns:
        int: A one-hot encoded label representing the RMSD value.
    """
    y = np.zeros(shape = N_RMSD_CLASSES)  # Initialize an array of zeros with shape N_CLASSES
    for idx, ub in enumerate(RMSD_BINS[1:]):
        if rmsd < ub:
            break
    y[idx] = 1  # Set the element at the corresponding index to 1
    return y



class LRMSDScaler:

    def __init__(self) -> None:
        return
    
    def fit(self, X: np.ndarray) -> None:
        return

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.power(1 + np.power(X / 4, 2),-1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return 4 * np.power(1 / X - 1, 0.5)
    


class GraphWrapper(Graph):

    def __init__(self, x=None, a=None, e=None, y=None, w=None, **kwargs):
        self.w = w
        super().__init__(x, a, e, y, **kwargs)