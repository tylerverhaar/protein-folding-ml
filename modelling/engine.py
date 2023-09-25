import numpy as np
from modelling.decoys import Decoys

BIN_WIDTH = 0.5
MAX_RMSD_VAL = 8
N_CLASSES = int(MAX_RMSD_VAL / BIN_WIDTH)
BINS = np.arange(0, MAX_RMSD_VAL + 1e-10, BIN_WIDTH)
BIN_MIDPTS = (BINS[0:-1] + BINS[1:]) / 2
TOP_K_VALUES = [5,10,30]

from modelling.train_test import TrainTestModule


from modelling.model_structures.baseline import GraphNet

def main():
    tt = TrainTestModule('modelling')
    results = tt.run(model = GraphNet(), train_size = 0.8, epochs_tr = 25)
    


    
    


if __name__ == '__main__':
    decoys = Decoys()
    decoys.read()
    pass
