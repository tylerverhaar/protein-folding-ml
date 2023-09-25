import numpy as np


BIN_WIDTH = 0.5
MAX_RMSD_VAL = 8
N_RMSD_CLASSES = int(MAX_RMSD_VAL / BIN_WIDTH)


RMSD_BINS = np.arange(0, MAX_RMSD_VAL + 1e-10, BIN_WIDTH)
BIN_MIDPTS = (RMSD_BINS[0:-1] + RMSD_BINS[1:]) / 2
TOP_K_VALUES = [5,10,30]
TARGET_GRAPH_DIR = 'data/graphs'

def get_expected_rmsd(pdist):
    expected_rmsd = np.sum(pdist * BIN_MIDPTS, axis = 1)
    return expected_rmsd