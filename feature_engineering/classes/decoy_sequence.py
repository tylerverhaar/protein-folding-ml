import Bio
from feature_engineering.classes.base_sequence import BaseSequence
from feature_engineering.classes.decoy_loop import DecoyLoop
import numpy as np
from numba import njit
from typing import Dict, Any, Optional, Tuple
import feature_engineering.function_lib.njit_implementations.geometry as geometry


class DecoySequence(BaseSequence):

    def __init__(self, protein_id: str, structure: Bio.PDB.Structure.Structure, start_residue: int, end_residue: int, decoy_loop: DecoyLoop) -> None:
        """
        Initialize a DecoySequence object.

        Parameters:
        -----------
        protein_id : str
            ID of the protein.
        structure : Bio.PDB.Structure.Structure
            Bio.PDB structure object representing the protein structure.
        start_residue : int
            Start residue index of the decoy loop.
        end_residue : int
            End residue index of the decoy loop.
        decoy_loop : DecoyLoop
            DecoyLoop object representing the decoy loop.

        """
        self.start_residue = start_residue
        self.end_residue = end_residue
        self.decoy_loop = decoy_loop
        self.energy_evaluation = decoy_loop.energy_evaluation
        self.decoy_id = decoy_loop.decoy_id
        self.invalid_sequence = False
        try:
            super().__init__(protein_id, structure, start_residue = start_residue, end_residue = end_residue)
        except Exception as e:
            print(f'DecoySequence.__init__ - error, trace below\n{e}')
            self.invalid_sequence = True

    def set_decoy_loop(self) -> None:
        """
        Set the decoy loop for the decoy sequence by replacing the residues and calculating RMSD.

        The function replaces the residues in the decoy loop with the corresponding residues in the sequence.
        It calculates the root-mean-square deviation (RMSD) between the backbone coordinates of the base structure and the decoy loop.

        Returns:
        --------
        None

        """
        
        self.rmsd_dict = {}  # Dictionary to store the RMSD values for each residue
        for residue in self.decoy_loop.loop:
            res_idx = self.index[residue.id]  # Get the index of the residue in the sequence
            bb_coords_base = self.residues[res_idx].bb_coords  # Backbone coordinates of the base structure residue
            bb_coords_decoy = residue.bb_coords  # Backbone coordinates of the decoy loop residue
            
            bb_coords_base = bb_coords_base.astype('float64')
            bb_coords_decoy = bb_coords_decoy.astype('float64')
            
            residue_rmsd = geometry.rmsd(coords1 = bb_coords_base, coords2 = bb_coords_decoy)  # Calculate RMSD
            self.rmsd_dict[residue.id] = residue_rmsd  # Store RMSD value for the residue
            
            assert self.residues[res_idx].id == residue.id, 'mismatched residue IDs, cannot replace existing residue'
            self.residues[res_idx] = residue  # Replace the residue in the sequence with the decoy loop residue

        # Calculate the overall RMSD for the decoy sequence by taking the square root of the average of the squared RMSD values
        self.decoy_rmsd = np.sqrt(np.sum([rmsd**2 for rmsd in self.rmsd_dict.values()]) / self.decoy_loop.n_residues)
        
    def set_residue_features(self, dist_k_neighbors: Optional[int] = 30, seq_k_neighbors: Optional[int] = 2) -> None:
        """
        Set the residue features for the decoy sequence.

        Parameters:
        -----------
        dist_k_neighbors : Optional[int], default=30
            Number of nearest neighbors to consider for distance-based features.
        seq_k_neighbors : Optional[int], default=2
            Number of nearest neighbors to consider for sequence-based features.

        """
        residue_ids = np.arange(self.start_residue, self.end_residue + 1)
        super().set_residue_features(residue_ids = residue_ids, dist_k_neighbors = dist_k_neighbors, seq_k_neighbors = seq_k_neighbors)
        
    def get_write_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the data to be written for the decoy sequence.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the node features, edge features, and additional information.

        """
        N, E = self.get_graph_representation()
        K = self.energy_evaluation * np.ones(shape = (N.shape[0], 1))
        N = np.append(N, K, axis = 1)
        I = np.append(np.array([self.decoy_loop.decoy_id, self.energy_evaluation, self.decoy_rmsd, self.start_residue, self.end_residue]),
                      np.array([value for key, value in sorted(self.rmsd_dict.items())]), 
                      axis = 0)
        return (N, E, I)



    