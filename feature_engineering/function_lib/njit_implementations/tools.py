from typing import Union, Tuple, Any, List, Optional
import numpy as np
from numba import njit


def batcher(iterable: Union[Tuple, List[Any]], batch_size: Optional[int] = 10) -> List[List]:
    """
    Splits an iterable into smaller batches of size `batch_size`.

    Args:
        iterable: An iterable object (e.g. list, tuple) to be split into batches.
        batch_size: An optional integer specifying the maximum number of elements in each batch. Default is 10.

    Returns:
        A list of lists, where each sublist contains a batch of elements from the original iterable.

    Raises:
        ValueError: If `batch_size` is not a positive integer.

    Example:
        >>> batcher([1, 2, 3, 4, 5, 6], batch_size=2)
        [[1, 2], [3, 4], [5, 6]]
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    batches = []
    max_idx = len(iterable)
    for start_idx in range(0, max_idx, batch_size):
        end_idx = min(start_idx + batch_size, max_idx)
        batches.append(iterable[start_idx:end_idx])

    return batches


@njit()
def flattened_upper_triangle(matrix: np.ndarray, k: Optional[int] = 0) -> np.ndarray:
    """
    Returns the flattened upper triangle of a square symmetric matrix.

    Parameters:
    -----------
    matrix: np.ndarray
        A square symmetric matrix.
    k: Optional[int]
        Offset from central diagonal. Default is 0 (no offset).

    Returns:
    --------
    np.ndarray:
        The flattened upper triangle of the input matrix.
    """
    matrix = matrix.astype('float64')  # Convert matrix to float64 to ensure compatibility
    n = matrix.shape[0]
    assert n == matrix.shape[1], "requires square matrix"
    assert n > k

    upper_triangle = []
    for i in range(n):
        for j in range(i + k, n):
            upper_triangle.append(matrix[i, j])

    return np.array(upper_triangle)