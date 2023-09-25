import numpy as np
from typing import Optional

def freedman_diaconis_rule(data: np.ndarray) -> float:
    """
    Determines the bin width using Freedman-Diaconis Rule.

    Parameters:
        data (np.ndarray): Input data array.

    Returns:
        float: Bin width.

    """
    n = len(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr * (n ** (-1/3))
    return bin_width

def scotts_rule(data: np.ndarray) -> float:
    """
    Determines the bin width using Scott's Rule.

    Parameters:
        data (np.ndarray): Input data array.

    Returns:
        float: Bin width.

    """
    n = len(data)
    sigma = np.std(data)
    bin_width = 3.5 * sigma * (n ** (-1/3))
    return bin_width

def sturges_rule(data: np.ndarray) -> float:
    """
    Determines the bin width using Sturges' Rule.

    Parameters:
        data (np.ndarray): Input data array.

    Returns:
        float: Bin width.

    """
    n = len(data)
    num_bins = int(np.ceil(1 + np.log2(n)))
    bin_width = (np.max(data) - np.min(data)) / num_bins
    return bin_width

def rices_rule(data: np.ndarray) -> float:
    """
    Determines the bin width using Rice's Rule.

    Parameters:
        data (np.ndarray): Input data array.

    Returns:
        float: Bin width.

    """
    n = len(data)
    num_bins = int(np.ceil(2 * (n ** (1/3))))
    bin_width = (np.max(data) - np.min(data)) / num_bins
    return bin_width

def square_root_choice(data: np.ndarray) -> float:
    """
    Determines the bin width using Square Root Choice.

    Parameters:
        data (np.ndarray): Input data array.

    Returns:
        float: Bin width.

    """
    n = len(data)
    num_bins = int(np.ceil(np.sqrt(n)))
    bin_width = (np.max(data) - np.min(data)) / num_bins
    return bin_width

def doanes_formula(data: np.ndarray) -> float:
    """
    Determines the bin width using Doane's Formula.

    Parameters:
        data (np.ndarray): Input data array.

    Returns:
        float: Bin width.

    """
    n = len(data)
    skewness = ((np.mean(data) - np.median(data)) / np.std(data)) * (6 / (n ** 0.5))
    num_bins = int(np.ceil(1 + np.log2(n) + np.log2(1 + abs(skewness) / 0.34)))
    bin_width = (np.max(data) - np.min(data)) / num_bins
    return bin_width

def get_bin_size(data: np.ndarray) -> float:
    """
    Determines the bin width using different rules for differing numbers of observations.

    Parameters:
        data (np.ndarray): Input data array.

    Returns:
        float: Bin width.

    """
    if data.shape[0] <= 10:
        bin_size = sturges_rule(data = data)
    elif data.shape[0] <= 30:
        bin_size = scotts_rule(data = data)
    else:
        bin_size = freedman_diaconis_rule(data = data)
    return bin_size

def get_discrete_labels(arr: np.ndarray, same_dist: Optional[bool] = False) -> np.ndarray:
    """
    Assign discrete labels to the values in the array based on the given bin sizes.

    Args:
        arr (np.ndarray): Array of shape (n, m) containing observations.
        bin_size (np.ndarray): Array of bin sizes for each column.
        same_dist (bool): Flag indicating whether all columns follow the same distribution.

    Returns:
        np.ndarray: Array of discrete labels.

    """
    if same_dist:
        bin_size = get_bin_size(data = arr.flatten())
        arr_min = np.min(arr)
        labels = np.floor((arr - arr_min) / bin_size).astype('int32')
        return labels
    
    n, m = arr.shape
    labels = np.zeros(shape = arr.shape).astype('int32')
    for i in range(m):       
        col = arr[:,i]
        bin_size = get_bin_size(data = col)
        col_min = np.min(col)
        labels[:,i] = np.floor((col - col_min) / bin_size).astype('int32')

    return labels

def get_label_probability(arr: np.ndarray, same_dist: Optional[bool] = False):
    """
    Compute the probabilities of labels for each value in the array based on the given bin sizes.

    Args:
        arr (np.ndarray): Array of shape (n, m) containing observations.
        same_dist (bool, optional): Flag indicating whether all columns follow the same distribution.
            Defaults to False.

    Returns:
        np.ndarray: Array of label probabilities.

    """
    n, m = arr.shape
    
    if same_dist:
        N = n * m
        arr_max = np.max(arr)
        bin_counts = np.bincount(arr.flatten(), minlength = arr_max + 1)
        bin_probs = bin_counts / N
        p_dist = {idx: bin_probs[idx] for idx in range(0, arr_max + 1)}
        probs = np.zeros(shape = arr.shape).astype('float64')
        for i in range(n):
            for j in range(m):
                probs[i,j] = p_dist[arr[i, j]]
        return probs
    
    probs = np.zeros(shape = arr.shape).astype('float64')
    for i in range(m):       
        col = arr[:,i]
        col_max = np.max(col)
        bin_counts = np.bincount(col, minlength = col_max + 1)
        bin_probs = bin_counts / n
        p_dist = {idx: bin_probs[idx] for idx in range(0, col_max + 1)}
        probs[:,i] = np.array([p_dist[col[j]] for j in range(n)]).astype('float64')
    return probs

def compute_entropy(arr: np.ndarray, same_dist: Optional[bool] = False, labelled_arr: Optional[bool] = False) -> np.ndarray:
    """
    Compute the row-wise entropy of the input array based on the given distribution assumptions.

    Args:
        arr (np.ndarray): Array of shape (n, m) containing observations.
        same_dist (bool, optional): Flag indicating whether all columns follow the same distribution.
            Defaults to False.
        labelled_arr (bool, optional): Flag indicating whether the input array is already labelled.
            If True, it is assumed that the values in the array are discrete labels.
            Defaults to False.

    Returns:
        np.ndarray: Array of row-wise entropy values.

    """
    if labelled_arr:
        labels = arr.astype('int32')
    else:
        labels = get_discrete_labels(arr = arr, same_dist = same_dist)
    
    probs = get_label_probability(arr = labels, same_dist = same_dist)
    entropy = -np.sum(probs * np.log2(probs), axis=1)
    
    return entropy

def convert_one_hot_to_label(arr: np.ndarray) -> np.ndarray:
    """
    Convert a one-hot encoded array to label-encoded representation.

    Parameters:
    -----------
    arr : np.ndarray
        The one-hot encoded array to be converted.

    Returns:
    --------
    np.ndarray:
        The label-encoded representation of the input array.

    Notes:
    ------
    This function converts a one-hot encoded array to a label-encoded representation, where each unique row in the
    input array is assigned a unique label.

    Examples:
    ---------
    >>> arr = np.array([[1, 0, 0],
    ...                 [0, 1, 0],
    ...                 [0, 0, 1],
    ...                 [1, 0, 0]])
    >>> labels = convert_one_hot_to_label(arr)
    >>> print(labels)
    [[0]
     [1]
     [2]
     [0]]
    """
    unique_labels = {}
    labels = np.empty(arr.shape[0], dtype=np.int64)
    current_label = 0

    for i in range(arr.shape[0]):
        is_unique = True

        for label, unique_row in unique_labels.items():
            if np.array_equal(arr[i], unique_row):
                labels[i] = label
                is_unique = False
                break

        if is_unique:
            unique_labels[current_label] = arr[i]
            labels[i] = current_label
            current_label += 1

    return labels.reshape(-1, 1)


def compute_one_hot_entropy(arr: np.ndarray, same_dist: Optional[bool] = False) -> np.array:
    arr = convert_one_hot_to_label(arr = arr)
    entropy = compute_entropy(arr = arr, same_dist = same_dist, labelled_arr = True)
    return entropy



if __name__ == '__main__':
    x1 = np.random.normal(size = (1000, 30))
    x1[0,] = 5 * np.ones(shape = 30)
    
    
    x = compute_entropy(x1)
    print(x)