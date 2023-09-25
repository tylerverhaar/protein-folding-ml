import numpy as np
import pandas as pd
import Bio
import itertools
import os
from collections import OrderedDict
from typing import List, Dict, Callable, Union, Optional, Tuple, Any

import feature_engineering.function_lib.njit_implementations.geometry as geometry
import feature_engineering.function_lib.njit_implementations.entropy as entropy
from .residue import Residue



class BaseSequence:

    def __init__(self, protein_id: str, structure: Bio.PDB.Structure.Structure, **kwargs: Dict[str,Any]) -> None:
        """
        Intializes the BaseSequence class used for computing node and edges features 
        and generateing their graph representations.

        Args:
            protein_id (str): PDB code for protein structure
            structure (Bio.PDB.Structure.Structure): protein structure
        """        
        self.protein_id = protein_id
        self.structure = structure

        self.start_residue = kwargs.get('start_residue', 0)
        self.end_residue = kwargs.get('end_residue', np.inf)

        self.residues = []
        self.res_id_index = {} # maps: list_index -> residue index
        self.index = {}        # maps: residue index -> list index
        
        self._set_base_sequence(**kwargs)

        BB_combos = set(itertools.permutations(['N', 'O', 'C', 'CA'], 2)) | set(itertools.product(['N', 'O', 'C', 'CA'], repeat=2))
        OTHER_combos = set(itertools.permutations(['bb_com', 'sc_com'], 2))| set(itertools.product(['bb_com', 'sc_com'], repeat=2))
        self.valid_combos = BB_combos.union(OTHER_combos)

        
    def _set_base_sequence(self, **kwargs: Dict[str,Any]) -> None:
        """
        Set the base sequence information by iterating over the structure and creating Residue objects.

        This function populates the `residues` list with Residue objects based on the structure information.
        It also sets the index and res_id_index dictionaries for easy access to residues by ID.

        Note: This function assumes that the structure attribute is already set.

        Returns:
            None
        """

        # Initialize variables
        self.residues = []  # List to store Residue objects
        self.index = {}  # Dictionary to map residue ID to index
        self.res_id_index = {}  # Dictionary to map index to residue ID
        idx = 0

        # Iterate over the structure to extract residue information
        for model in self.structure:
            for chain in model:
                for res in chain:
                    # Extract residue information
                    id = res.get_id()[1]  # Residue ID
                    in_loop = (self.start_residue <= id) and (id <= self.end_residue)
                    residue_type = res.get_resname()  # Residue type
                    chain_id = chain.get_id()  # Chain ID

                    raw_atoms = []  # List to store raw atom information
                    for atom in res:
                        # Extract atom information
                        atom_name = atom.get_name()  # Atom name
                        atom_number = atom.get_serial_number()  # Atom number
                        x, y, z = atom.get_coord()  # Atom coordinates

                        raw_atom = {
                            'atom_name': atom_name,
                            'atom_number': atom_number,
                            'x_coord': x,
                            'y_coord': y,
                            'z_coord': z
                        }
                        raw_atoms.append(raw_atom)
                    # Create a Residue object
                    try:
                        residue = Residue(id = id, residue_type = residue_type, chain_id = chain_id, raw_atoms = raw_atoms, in_loop = in_loop)
                    except Exception as e:
                        print(f'BaseSequence._set_base_sequence error for {self.protein_id}, raising exception')
                        raise Exception(e)

                    # Update index dictionaries
                    self.index[id] = idx
                    self.res_id_index[idx] = id

                    # Add the Residue object to the list
                    self.residues.append(residue)

                    idx += 1  # Increment the index

        self.n_residues = len(self.residues)  # Set the number of residues

    '''
    Node Features SET Methods
    '''

    def _set_bb_coords_arrays(self, hard_set: Optional[bool] = False) -> None:
        """
        Set the backbone coordinates array.

        Args:
            hard_set (bool, optional): If True, the array will be set even if it already exists.
                                    Defaults to False.
        """
        if not hasattr(self, "bb_coords") or hard_set:
            self.bb_coords = np.array([res.bb_coords for res in self.residues]).astype('float64')

    def _set_bb_com_coords_arrays(self, hard_set: Optional[bool] = False) -> None:
        """
        Set the backbone center of mass coordinates array.

        Args:
            hard_set (bool, optional): If True, the array will be set even if it already exists.
                                    Defaults to False.
        """
        if not hasattr(self, "bb_com_coords") or hard_set:
            self.bb_com_coords = np.array([res._get_bb_com_coords(weight = 'mass') for res in self.residues]).astype('float64')
    
    def _set_sc_com_coords_arrays(self, hard_set: Optional[bool] = False) -> None:
        """
        Set the side-chain center of mass coordinates array.

        Args:
            hard_set (bool, optional): If True, the array will be set even if it already exists.
                                    Defaults to False.
        """
        if not hasattr(self, "sc_com_coords") or hard_set:
            self.sc_com_coords = np.array([res._get_sc_com_coords(weight = 'mass') for res in self.residues]).astype('float64')

    def _set_coordinate_dictionary(self, hard_set: Optional[bool] = False) -> None:
        """
        Set the coordinate values for bacbone atoms, and side-chain.

        Args:
            hard_set (bool, optional): If True, the array will be set even if it already exists.
                                    Defaults to False.
        """
        if not hasattr(self, 'coordinate_dict') or hard_set:
            self._set_bb_coords_arrays(hard_set = hard_set)
            self._set_bb_com_coords_arrays(hard_set = hard_set)
            self._set_sc_com_coords_arrays(hard_set = hard_set)
            self.cb_coords = np.array([res._get_cb_coords(bb_com_weight = 'mass') for res in self.residues]).astype('float64')
            self.coordinate_dict = {
                "bb_com": self.bb_com_coords,
                "sc_com": self.sc_com_coords,
                "N": self.bb_coords[:, 0, :],
                "CA": self.bb_coords[:, 1, :],
                "C": self.bb_coords[:, 2, :],
                "O": self.bb_coords[:, 3, :],
                "CB": self.cb_coords
            }
    
    def _set_nn_distance_indexing(self, nn_axis: np.array, k_neighbors: Optional[int] = 30, hard_set: Optional[bool] = False) -> None:
        """
        Set the nearest neighbor distance indices and pairwise distance matrix.

        Args:
            nn_axis (np.array): The indices of the points to consider for nearest neighbor calculations.
            k_neighbors (int, optional): The number of nearest neighbors to consider.
                                        Defaults to 30.
            hard_set (bool, optional): If True, the attributes will be set even if they already exist.
                                    Defaults to False.
        """
        if not hasattr(self, "bb_com_pw_distance") or hard_set:
            self.bb_com_pw_distance = geometry.pairwise_distance_matrix(pts = self.bb_com_coords, axis = nn_axis)[nn_axis, ]
            
        if not hasattr(self, "nn_distance_indices") or hard_set:
            self.nn_distance_indices = np.argsort(self.bb_com_pw_distance, axis=1)[:, 1:1+k_neighbors]

    def _set_dihedrals_angles(self, residue_ids: np.array, hard_set: Optional[bool] = False) -> None:
        """
        Calculate and set the dihedral angles (omega, phi, psi) for the given residue IDs.

        Args:
            residue_ids (np.array): Array of residue IDs for which to calculate the dihedral angles.
            hard_set (bool, optional): If True, the coordinate dictionary will be set even if it already exists.
                                    Defaults to False.
        """

        if not hasattr(self, 'dihedrals') or hard_set:
            
            # Set the coordinate dictionary
            self._set_coordinate_dictionary(hard_set = hard_set)

            # Retrieve the necessary coordinate arrays
            residue_index = [self.index[res_id] for res_id in residue_ids]
            CA = self.coordinate_dict['CA']
            N = self.coordinate_dict['N']
            C = self.coordinate_dict['C']
            O = self.coordinate_dict['O']

            # Initialize the dihedral angles array
            self.dihedrals = np.zeros(shape = (residue_ids.shape[0], 3))

            # Calculate dihedral angles for each residue
            for i, res_idx in enumerate(residue_index):
                omega = geometry.dihedrals(CA[res_idx], C[res_idx], N[res_idx + 1], CA[res_idx + 1])
                phi = geometry.dihedrals(C[res_idx - 1], N[res_idx], CA[res_idx], C[res_idx])
                psi = geometry.dihedrals(N[res_idx], CA[res_idx], C[res_idx], N[res_idx + 1])
                self.dihedrals[i,] = omega, phi, psi

    def _set_nn_sequence_indexing(self, nn_axis: np.array, k_neighbors: Optional[int] = 30, hard_set: Optional[bool] = False) -> None:
        """
        Set the nearest neighbor sequence indices.

        Args:
            nn_axis (np.array): The indices of the points to consider for nearest neighbor calculations.
            k_neighbors (int, optional): The number of nearest neighbors to consider.
                                        Defaults to 30.
            hard_set (bool, optional): If True, the attribute will be set even if it already exists.
                                    Defaults to False.
        """
        if not hasattr(self, "nn_sequence_indices") or hard_set:
            indices = []
            for idx in nn_axis:
                neighbor_indices = list(range(idx - k_neighbors // 2, idx)) + list(range(idx + 1, idx + k_neighbors // 2 + 1))
                indices.append(neighbor_indices)
            self.nn_sequence_indices = np.array(indices)

    def _set_loop_com(self, residue_ids: np.array, hard_set: Optional[bool] = False) -> None:
        if hasattr(self, 'loop_com') and not hard_set:
            return
        
        self._set_coordinate_dictionary(hard_set = hard_set)
        residue_index = [self.index[res_id] for res_id in residue_ids]
        self.loop_com = self.coordinate_dict['bb_com'][residue_index,].mean(axis = 0)

    def _set_bb_com_distances(self, residue_ids: np.array, hard_set: Optional[bool] = False) -> None:
        if hasattr(self,'bb_com_distances') and not hard_set:
            return
        
        self._set_loop_com(residue_ids = residue_ids, hard_set = hard_set)
        residue_index = [self.index[res_id] for res_id in residue_ids]
        self.bb_com_distances = {}
        
        for key in self.coordinate_dict.keys():
            self.bb_com_distances[key] = np.array([geometry.p_norm(p1 = p, p2 = self.loop_com) for p in self.coordinate_dict[key][residue_index]]).reshape(-1,1)

    def _set_bb_com_angles(self, residue_ids: np.array, hard_set: Optional[bool] = False) -> None:
        if hasattr(self,'bb_com_angles') and not hard_set:
            return
        
        self._set_loop_com(residue_ids = residue_ids, hard_set = hard_set)
        residue_index = [self.index[res_id] for res_id in residue_ids]
        self.bb_com_angles = {}
        for key in self.coordinate_dict.keys():
            shifted = self.coordinate_dict[key][residue_index] - self.loop_com
            spherical_repr = geometry.cartesian_to_spherical_conversion(x = shifted)[:,1:]
            angle_repr = geometry.point_array_angle(pt = self.loop_com, arr = self.coordinate_dict[key][residue_index]).reshape(-1,1)
            self.bb_com_angles[key] = np.hstack([spherical_repr, angle_repr])

    '''
    Edge Features SET Methods
    '''

    def _set_edge_ids(self, residue_ids: np.array, hard_set: Optional[bool] = False) -> None:
        """
        Set the edge IDs between residue IDs, we set edges for a fully connected graph representation here
        because we can always omit edges during training later and change the adjacency matrix structure.

        Args:
            residue_ids (np.array): An array of residue IDs.
            hard_set (bool, optional): If True, the attribute will be set even if it already exists.
                                    Defaults to False.
        """
        if hasattr(self, 'edge_ids') and not hard_set:
            return
        
        self.edge_ids = []
        for i in range(residue_ids.shape[0]):
            for j in range(residue_ids.shape[0]):
                if i != j:
                    self.edge_ids.append((residue_ids[i], residue_ids[j]))

    '''
    Node Features GET Methods
    '''

    def _get_nn_distance_pairwise_dists(self, nn_axis: np.array, k_neighbors: Optional[int] = 30) -> dict:
        """
        Compute pairwise nearest neighbor distances between different coordinate arrays.

        Args:
            nn_axis (np.array): The indices of the points to consider for nearest neighbor calculations.
            k_neighbors (int, optional): The number of nearest neighbors to consider. Defaults to 30.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing pairwise nearest neighbor distances for different coordinate arrays.
                The keys of the dictionary follow the format 'nn_distance_{array1}_{array2}'.

        """
        # Set the coordinate dictionary
        self._set_coordinate_dictionary()

        # Compute nearest neighbor indexing
        self._set_nn_distance_indexing(nn_axis=nn_axis, k_neighbors=k_neighbors)

        # Initialize dictionary to store pairwise distances
        nn_distances = {}
        
        # Iterate over coordinate array combinations
        for key_i, coord_array_i in self.coordinate_dict.items():
            for key_j, coord_array_j in self.coordinate_dict.items():

                if (key_i, key_j) not in self.valid_combos:
                    continue

                # Create a unique key for the dictionary
                key = f"nn(dist)_{key_i}_{key_j}_dists"

                # Initialize an array to store distances
                if key not in nn_distances:
                    nn_distances[key] = np.empty(shape=(nn_axis.shape[0], k_neighbors))

                # Compute nearest neighbor distances
                for i, nn_idx in enumerate(nn_axis):
                    nn_idx_neighbor_index = self.nn_distance_indices[i,]
                    nn_com_coords = coord_array_j[nn_idx_neighbor_index,]
                    d = geometry.point_array_distance(pt=coord_array_i[nn_idx], arr=nn_com_coords)
                    nn_distances[key][i, ] = d

        return nn_distances

    def _get_nn_distance_pairwise_angles(self, nn_axis: np.array, k_neighbors: Optional[int] = 30) -> dict:
        """
        Compute pairwise nearest neighbor angles between different coordinate arrays.

        Args:
            nn_axis (np.array): The indices of the points to consider for nearest neighbor calculations.
            k_neighbors (int, optional): The number of nearest neighbors to consider. Defaults to 30.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing pairwise nearest neighbor angles for different coordinate arrays.
                The keys of the dictionary follow the format 'nn_distance_{array1}_{array2}'.

        """
        # Set the coordinate dictionary
        self._set_coordinate_dictionary()

        # Compute nearest neighbor indexing
        self._set_nn_distance_indexing(nn_axis=nn_axis, k_neighbors=k_neighbors)

        # Initialize dictionary to store pairwise angles
        nn_angles = {}

        # Iterate over coordinate array combinations
        for key_i, coord_array_i in self.coordinate_dict.items():
            for key_j, coord_array_j in self.coordinate_dict.items():

                if (key_i, key_j) not in self.valid_combos:
                    continue

                # Create a unique key for the dictionary
                key = f"nn(dist)_{key_i}_{key_j}_angle"

                # Initialize an array to store angles
                if key not in nn_angles:
                    nn_angles[key] = np.empty(shape=(nn_axis.shape[0], k_neighbors))

                # Compute nearest neighbor angles
                for i, nn_idx in enumerate(nn_axis):
                    nn_idx_neighbor_index = self.nn_distance_indices[i,]
                    nn_com_coords = coord_array_j[nn_idx_neighbor_index,]
                    d = geometry.point_array_angle(pt=coord_array_i[nn_idx], arr=nn_com_coords)
                    nn_angles[key][i, ] = d

        return nn_angles
    
    def _get_nn_distance_pairwise_angles_spherical(self, nn_axis: np.array, k_neighbors: Optional[int] = 30) -> dict:
        """
        Compute pairwise nearest neighbor angles between different coordinate arrays.

        Args:
            nn_axis (np.array): The indices of the points to consider for nearest neighbor calculations.
            k_neighbors (int, optional): The number of nearest neighbors to consider. Defaults to 30.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing pairwise nearest neighbor angles for different coordinate arrays.
                The keys of the dictionary follow the format 'nn_distance_{array1}_{array2}'.

        """
        # Set the coordinate dictionary
        self._set_coordinate_dictionary()

        # Compute nearest neighbor indexing
        self._set_nn_distance_indexing(nn_axis=nn_axis, k_neighbors=k_neighbors)

        # Initialize dictionary to store pairwise angles
        nn_angles_spherical = {}
        
        # Iterate over coordinate array combinations
        for key_i, coord_array_i in self.coordinate_dict.items():
            for key_j, coord_array_j in self.coordinate_dict.items():

                if (key_i, key_j) not in [('bb_com', 'bb_com'),('CA', 'CA'), ('N', 'N'),('O', 'O'),('C', 'C')]:
                    continue

                # Create a unique key for the dictionary
                key_theta = f"nn(dist)_{key_i}_{key_j}_angle_spherical_theta"
                key_psi = f"nn(dist)_{key_i}_{key_j}_angle_spherical_psi"

                # Initialize an array to store angles
                if key_theta not in nn_angles_spherical:
                    nn_angles_spherical[key_theta] = np.zeros(shape = (nn_axis.shape[0], k_neighbors))

                if key_psi not in nn_angles_spherical:
                    nn_angles_spherical[key_psi] = np.zeros(shape = (nn_axis.shape[0], k_neighbors))

                # Compute nearest neighbor angles
                for i, nn_idx in enumerate(nn_axis):
                    nn_idx_neighbor_index = self.nn_distance_indices[i,]
                    nn_com_coords = coord_array_j[nn_idx_neighbor_index,]
                    shifted = nn_com_coords - coord_array_i[nn_idx]
                    coords_spherical = geometry.cartesian_to_spherical_conversion(x = shifted)
                    nn_angles_spherical[key_theta][i, ] = coords_spherical[:,1]
                    nn_angles_spherical[key_psi][i, ] = coords_spherical[:,2]
        
        return nn_angles_spherical
    

    def _get_nn_distance_residue_types(self, nn_axis: np.array, k_neighbors: Optional[int] = 30) -> dict:
        """
        Compute pairwise nearest neighbor residue types.

        Args:
            nn_axis (np.array): The indices of the points to consider for nearest neighbor calculations.
            k_neighbors (int, optional): The number of nearest neighbors to consider. Defaults to 30.

        Returns:
            Dict[str, np.ndarray]: A np.array containing the sum of the residue type one-hot encodings of the nearest neighbor residues.

        """
        # Set the coordinate dictionary
        self._set_coordinate_dictionary()

        # Compute nearest neighbor indexing
        self._set_nn_distance_indexing(nn_axis = nn_axis, k_neighbors = k_neighbors)

        # Initialize dictionary to store pairwise angles
        nn_angles_spherical = {}
        
        # Iterate over coordinate array combinations
        for key_i, coord_array_i in self.coordinate_dict.items():
            for key_j, coord_array_j in self.coordinate_dict.items():

                if (key_i, key_j) not in [('bb_com', 'bb_com'),('CA', 'CA'), ('N', 'N'),('O', 'O'),('C', 'C')]:
                    continue

                # Create a unique key for the dictionary
                key_theta = f"nn(dist)_{key_i}_{key_j}_angle_spherical_theta"
                key_psi = f"nn(dist)_{key_i}_{key_j}_angle_spherical_psi"

                # Initialize an array to store angles
                if key_theta not in nn_angles_spherical:
                    nn_angles_spherical[key_theta] = np.zeros(shape = (nn_axis.shape[0], k_neighbors))

                if key_psi not in nn_angles_spherical:
                    nn_angles_spherical[key_psi] = np.zeros(shape = (nn_axis.shape[0], k_neighbors))

                # Compute nearest neighbor angles
                for i, nn_idx in enumerate(nn_axis):
                    nn_idx_neighbor_index = self.nn_distance_indices[i,]
                    
                    
                    
                    nn_com_coords = coord_array_j[nn_idx_neighbor_index,]
                    shifted = nn_com_coords - coord_array_i[nn_idx]
                    coords_spherical = geometry.cartesian_to_spherical_conversion(x = shifted)
                    nn_angles_spherical[key_theta][i, ] = coords_spherical[:,1]
                    nn_angles_spherical[key_psi][i, ] = coords_spherical[:,2]
        
        return nn_angles_spherical


    def _get_nn_sequence_pairwise_dists(self, nn_axis: np.array, k_neighbors: Optional[int] = 2) -> dict:
        """
        Compute pairwise nearest neighbor distances between different coordinate arrays.

        Args:
            nn_axis (np.array): The indices of the points to consider for nearest neighbor calculations.
            k_neighbors (int, optional): The number of nearest neighbors to consider. Defaults to 30.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing pairwise nearest neighbor distances for different coordinate arrays.
                The keys of the dictionary follow the format 'nn_distance_{array1}_{array2}'.

        """
        # Set the coordinate dictionary
        self._set_coordinate_dictionary()

        # Compute nearest neighbor indexing
        self._set_nn_sequence_indexing(nn_axis=nn_axis, k_neighbors=k_neighbors)

        # Initialize dictionary to store pairwise distances
        nn_distances = {}

        # combinations to compute sequence pairs for
        sequence_combos = set([('bb_com', 'bb_com')])

        # Iterate over coordinate array combinations
        for key_i, coord_array_i in self.coordinate_dict.items():
            for key_j, coord_array_j in self.coordinate_dict.items():

                if (key_i, key_j) not in sequence_combos:
                    continue

                # Create a unique key for the dictionary
                key = f"nn(seq)_{key_i}_{key_j}_dists"

                # Initialize an array to store distances
                if key not in nn_distances:
                    nn_distances[key] = np.empty(shape=(nn_axis.shape[0], k_neighbors))

                # Compute nearest neighbor distances
                for i, nn_idx in enumerate(nn_axis):
                    nn_idx_neighbor_index = self.nn_sequence_indices[i,]
                    nn_com_coords = coord_array_j[nn_idx_neighbor_index,]
                    d = geometry.point_array_distance(pt=coord_array_i[nn_idx], arr=nn_com_coords)
                    nn_distances[key][i, ] = d

        return nn_distances

    def _get_nn_sequence_pairwise_angles(self, nn_axis: np.array, k_neighbors: Optional[int] = 2) -> dict:
        """
        Compute pairwise nearest neighbor angles between different coordinate arrays.

        Args:
            nn_axis (np.array): The indices of the points to consider for nearest neighbor calculations.
            k_neighbors (int, optional): The number of nearest neighbors to consider. Defaults to 30.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing pairwise nearest neighbor angles for different coordinate arrays.
                The keys of the dictionary follow the format 'nn_distance_{array1}_{array2}'.

        """
        # Set the coordinate dictionary
        self._set_coordinate_dictionary()

        # Compute nearest neighbor indexing
        self._set_nn_sequence_indexing(nn_axis=nn_axis, k_neighbors=k_neighbors)

        # Initialize dictionary to store pairwise angles
        nn_angles = {}

        # combinations to compute sequence pairs for
        sequence_combos = set([('bb_com', 'bb_com')])

        # Iterate over coordinate array combinations
        for key_i, coord_array_i in self.coordinate_dict.items():
            for key_j, coord_array_j in self.coordinate_dict.items():

                if (key_i, key_j) not in sequence_combos:
                    continue

                # Create a unique key for the dictionary
                key = f"nn(seq)_{key_i}_{key_j}_angles"

                # Initialize an array to store angles
                if key not in nn_angles:
                    nn_angles[key] = np.empty(shape=(nn_axis.shape[0], k_neighbors))

                # Compute nearest neighbor angles
                for i, nn_idx in enumerate(nn_axis):
                    nn_idx_neighbor_index = self.nn_sequence_indices[i,]
                    nn_com_coords = coord_array_j[nn_idx_neighbor_index,]
                    d = geometry.point_array_angle(pt=coord_array_i[nn_idx], arr=nn_com_coords)
                    nn_angles[key][i, ] = d

        return nn_angles
        
    def _get_nn_sequence_pairwise_angles_spherical(self, nn_axis: np.array, k_neighbors: Optional[int] = 2) -> dict:
        """
        Compute pairwise nearest neighbor angles between different coordinate arrays.

        Args:
            nn_axis (np.array): The indices of the points to consider for nearest neighbor calculations.
            k_neighbors (int, optional): The number of nearest neighbors to consider. Defaults to 30.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing pairwise nearest neighbor angles for different coordinate arrays.
                The keys of the dictionary follow the format 'nn_distance_{array1}_{array2}'.

        """
        # Set the coordinate dictionary
        self._set_coordinate_dictionary()

        # Compute nearest neighbor indexing
        self._set_nn_sequence_indexing(nn_axis=nn_axis, k_neighbors=k_neighbors)

        # Initialize dictionary to store pairwise angles
        nn_angles_spherical = {}

        # combinations to compute sequence pairs for
        sequence_combos = set([('bb_com', 'bb_com')])

        # Iterate over coordinate array combinations
        for key_i, coord_array_i in self.coordinate_dict.items():
            for key_j, coord_array_j in self.coordinate_dict.items():

                if (key_i, key_j) not in sequence_combos:
                    continue

                # Create a unique key for the dictionary
                key_theta = f"nn(seq)_{key_i}_{key_j}_angle_spherical_theta"
                key_psi = f"nn(seq)_{key_i}_{key_j}_angle_spherical_psi"

                # Initialize an array to store angles
                if key_theta not in nn_angles_spherical:
                    nn_angles_spherical[key_theta] = np.zeros(shape = (nn_axis.shape[0], k_neighbors))

                if key_psi not in nn_angles_spherical:
                    nn_angles_spherical[key_psi] = np.zeros(shape = (nn_axis.shape[0], k_neighbors))

                # Compute nearest neighbor angles
                for i, nn_idx in enumerate(nn_axis):
                    nn_idx_neighbor_index = self.nn_sequence_indices[i,]
                    nn_com_coords = coord_array_j[nn_idx_neighbor_index,]
                    shifted = nn_com_coords - coord_array_i[nn_idx]
                    coords_spherical = geometry.cartesian_to_spherical_conversion(x = shifted)
                    nn_angles_spherical[key_theta][i, ] = coords_spherical[:,1]
                    nn_angles_spherical[key_psi][i, ] = coords_spherical[:,2]
        return nn_angles_spherical
    
    def _get_dihedral_angles(self, residue_ids: np.array) -> Dict[str, np.ndarray]:
        """
        Get the dihedral angles (omega, phi, psi) for the given residue IDs.

        Args:
            residue_ids (np.array): Array of residue IDs for which to calculate the dihedral angles.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the dihedral angles (omega, phi, psi).

        """
        # Calculate the dihedral angles for the given residue IDs
        self._set_dihedrals_angles(residue_ids=residue_ids)

        # Return a dictionary containing the dihedral angles
        return {
            'omega': self.dihedrals[:, 0].reshape(-1, 1),  # Extract the omega dihedral angles
            'phi': self.dihedrals[:, 1].reshape(-1, 1),    # Extract the phi dihedral angles
            'psi': self.dihedrals[:, 2].reshape(-1, 1)     # Extract the psi dihedral angles
        }   
    
    def _get_bb_com_distances(self, residue_ids: np.array) -> Dict[str, np.ndarray]:
        self._set_bb_com_distances(residue_ids = residue_ids)
        return {
            'CA_COM_dist': self.bb_com_distances['CA'],
            'C_COM_dist': self.bb_com_distances['C'],
            'O_COM_dist': self.bb_com_distances['O'],
            'N_COM_dist': self.bb_com_distances['N'],
            'bb_com_COM_dist': self.bb_com_distances['bb_com']
        }

    def _get_bb_com_angles(self, residue_ids: np.array) -> Dict[str, np.ndarray]:
        self._set_bb_com_angles(residue_ids = residue_ids)
        return {
            'CA_COM_angle': self.bb_com_angles['CA'],
            'C_COM_angle': self.bb_com_angles['C'],
            'O_COM_angle': self.bb_com_angles['O'],
            'N_COM_angle': self.bb_com_angles['N'],
            'bb_com_COM_angle': self.bb_com_angles['bb_com']
        }

    def _get_contact_maps(self, residue_ids, map_params: List[Dict[str,Any]], n_groups: Optional[int] = 3, **kwargs: Dict[str,Any]) -> None:
        """
        Set contact maps based on given parameters.

        Args:
            residue_ids (np.array): Array of residue IDs.
            map_params (List[Dict[str,Any]]): List of dictionaries containing map parameters.
            n_groups (Optional[int], optional): Number of contact map groups to consider. Defaults to 3.
            **kwargs (Dict[str,Any]): Additional keyword arguments.

        Returns:
            None
        """
        
        # setting maps
        contact_map_features = {}
        for map_param in map_params:
            coord_type = map_param['coord_type']
            DM = geometry.pairwise_distance_matrix(pts = self.coordinate_dict[coord_type])
            CM = 1 * (DM <= map_param['threshold'])
            contact_groups = self._get_contact_map_groups(DM = DM, CM = CM, residue_ids = residue_ids, coord_type = coord_type)
            for key in ['residue_contacts', 'size', 'shift', 'com_spherical_repr', 'angle', 'com']:
                for i in range(n_groups):
                    key_i = f'{key}_g{i+1}'
                    contact_map_features[f'{coord_type}_{key_i}'] = np.vstack([g[i][key_i] for g in contact_groups])
        return contact_map_features
    
    def _get_contact_map_groups(self, DM: np.ndarray, CM: np.ndarray, residue_ids: np.array, coord_type: str, n_groups: Optional[int] =  3) -> np.array:
        """
        Generate contact map groups based on given distance matrix and contact matrix.

        Args:
            DM (np.ndarray): Distance matrix.
            CM (np.ndarray): Contact matrix.
            residue_ids (np.array): Array of residue IDs.
            coord_type (str): Coordinate type.
            n_groups (Optional[int], optional): Number of contact map groups to consider. Defaults to 3.

        Returns:
            np.array: Array of contact map groups.
        """
        # construct groups
        groups = []
        for res_id in residue_ids:
            res_idx = self.index[res_id]
            row = CM[res_idx]
            g = []; s = []
            for i, value in enumerate(row):
                if value == 1:
                    s.append(i)
                elif s:
                    g.append({'res_id': res_id, 'res_idx': res_idx, 'group_index': s.copy(), 'average_group_distances': np.mean(DM[res_idx, s])})
                    s.clear()
                else:
                    continue
            if s:
                g.append({'res_id': res_id, 'res_idx': res_idx, 'group_index': s.copy(), 'average_group_distances': np.mean(DM[res_idx, s])})
            
            # sort and take the top n_groups segment groups
            g.sort(key = lambda s: s['average_group_distances'])
            g = g if len(g) <= n_groups else g[0:n_groups]

            # derive features from segment groups
            n_derived_groups = len(g)
            for i in range(n_groups):
                if n_derived_groups <= i:
                    g.append({
                        f'residue_contacts_g{i+1}': np.zeros(shape = (22,)), 
                        f'size_g{i+1}': np.array([0]), 
                        f'shift_g{i+1}': np.array([0]),
                        f'com_spherical_repr_g{i+1}': np.array([0,0,0]),
                        f'angle_g{i+1}': np.array([0]),
                        f'com_g{i+1}': np.array([0,0,0])
                        })
                else:
                    s = g[i]
                    g_residue_types = np.sum(np.vstack([self.residues[res_idx].residue_type_one_hot for res_idx in s['group_index']]), axis = 0)
                    g_coords = self.coordinate_dict[coord_type][s['group_index']]
                    g_com = geometry.centre_of_mass(p = g_coords, m = np.ones(shape = (g_coords.shape[0],1)))
                    res_coords = self.coordinate_dict[coord_type][res_idx]
                    g_com_shifted = (g_com - res_coords).reshape(-1,1).T
                    s[f'residue_contacts_g{i+1}'] = g_residue_types
                    s[f'size_g{i+1}'] = np.array([g_coords.shape[0]])
                    s[f'shift_g{i+1}'] = np.array([geometry.p_norm(p1 = res_coords, p2 = np.array([0,0,0]), p = 2)])
                    s[f'com_spherical_repr_g{i+1}'] = geometry.cartesian_to_spherical_conversion(x = g_com_shifted)
                    s[f'angle_g{i+1}'] = np.array([geometry.vector_angle(v1 = res_coords, v2 = g_com)])
                    s[f'com_g{i+1}'] = g_com

            groups.append(g)
        return groups

    '''
    Edge Feature GET Methods
    '''

    def _get_edge_distances(
            self, 
            residue_ids: np.array, 
            coord_keys: Optional[Union[List[str], str]] = 'bb_com_coords_wm'
            ) -> Dict[str, np.ndarray]:
        """
        Get edge distances between consecutive residues.

        Args:
            residue_ids (np.array): An array of residue IDs.
            coord_keys (Union[List[str], str], optional): Keys indicating the coordinate features to retrieve the edge distances from.
                                                        Defaults to 'nn(seq)_bb_com_bb_com_dists'.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the edge distances for each specified coordinate key.
        """
        if hasattr(self, 'edge_distances'):
            return self.edge_distances
        
        if isinstance(coord_keys, str):
            coord_keys = [coord_keys]
        
        self.edge_distances = {}
        
        for key in coord_keys:
            coords = self.residue_level_node_features[key]
            key_edge_dists = []
            cache = {}
            for i in range(residue_ids.shape[0]):
                for j in range(residue_ids.shape[0]):
                    if i == j:
                        continue
                    if (j,i) in cache:
                        d = cache[(j,i)]
                    elif (i,j) in cache:
                        d = cache[(i,j)]
                    else:
                        d = geometry.p_norm(p1 = coords[i,], p2 = coords[j,])
                        cache[(i,j)] = d
                    key_edge_dists.append(d)    
            self.edge_distances[f'{key}_dist'] = np.array(key_edge_dists).reshape(-1,1)
        return self.edge_distances

    def _get_edge_angles(
            self, 
            residue_ids: np.array, 
            coord_keys: Optional[Union[List[str], str]] = 'bb_com_coords_wm'
            ) -> Dict[str, np.ndarray]:
        """
        Get edge angles between consecutive residues.

        Args:
            residue_ids (np.array): An array of residue IDs.
            coord_keys (Union[List[str], str], optional): Keys indicating the coordinate features to retrieve the edge angles from.
                                                        Defaults to 'nn(seq)_bb_com_bb_com_angles'.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the edge angles for each specified coordinate key.
        """
        if hasattr(self, 'edge_angles'):
            return self.edge_angles
        
        if isinstance(coord_keys, str):
            coord_keys = [coord_keys]
        
        self.edge_angles = {}
        for key in coord_keys:
            coords = self.residue_level_node_features[key]
            key_edge_angles = []
            cache = {}

            for i in range(residue_ids.shape[0]):
                for j in range(residue_ids.shape[0]):
                    if i == j:
                        continue
                    if (j,i) in cache:
                        d = cache[(j,i)]
                    elif (i,j) in cache:
                        d = cache[(i,j)]
                    else:
                        d = geometry.vector_angle(v1 = coords[i,], v2 = coords[j,])
                        cache[(i,j)] = d
                    key_edge_angles.append(d)
            self.edge_angles[f'{key}_angle'] = np.array(key_edge_angles).reshape(-1,1)
        return self.edge_angles

    def _get_edge_spherical_angles(
        self, 
        residue_ids: np.array, 
        coord_keys: Optional[Union[List[str], str]] = ['bb_com_coords_wm', 'CA_coords']
        ) -> Dict[str, np.ndarray]:
        """
        Get edge angles between consecutive residues.

        Args:
            residue_ids (np.array): An array of residue IDs.
            coord_keys (Union[List[str], str], optional): Keys indicating the coordinate features to retrieve the edge angles from.
                                                        Defaults to READ IT

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the edge angles for each specified coordinate key.
        """
        if hasattr(self, 'edge_spherical_angles'):
            return self.edge_spherical_angles
        
        if isinstance(coord_keys, str):
            coord_keys = [coord_keys]
        
        self.edge_spherical_angles = {}
        for key in coord_keys:
            coords = self.residue_level_node_features[key]
            s_coords = geometry.pairwise_spherical_coordinates(points = coords)
            s_coords_theta = s_coords[:,:,1]
            s_coords_psi = s_coords[:,:,2]

            key_edge_theta = []
            key_edge_psi = []
            for i in range(residue_ids.shape[0]):
                for j in range(residue_ids.shape[0]):
                    if i != j:
                        key_edge_theta.append(s_coords_theta[i,j])
                        key_edge_psi.append(s_coords_psi[i,j])
            self.edge_spherical_angles[f'{key}_angle_theta'] = np.array(key_edge_theta).reshape(-1,1)
            self.edge_spherical_angles[f'{key}_angle_psi'] = np.array(key_edge_psi).reshape(-1,1)

        return self.edge_spherical_angles

    def _get_edge_types(self, residue_ids: np.array) -> Dict[str, np.ndarray]:
        """
        Get edge types between consecutive residues.

        Args:
            residue_ids (np.array): An array of residue IDs.

        Returns:
            np.ndarray: An array containing the edge types between consecutive residues.
        """
        self._set_edge_ids(residue_ids=residue_ids)

        def slide_insert(v1: np.array, v2: np.array) -> np.array:
            """
            Slide and insert elements from two arrays into a new array.

            Args:
                v1 (np.array): First input array.
                v2 (np.array): Second input array.

            Returns:
                np.array: New array with elements interleaved from v1 and v2.
            """
            v1v2 = np.array([x for pair in zip(v1, v2) for x in pair])
            return v1v2

        self.edge_types = []
        residue_types = self.residue_level_node_features['residue_type']
        for i in range(residue_ids.shape[0]):
            for j in range(residue_ids.shape[0]):
                if i == j:
                    continue
                self.edge_types.append(slide_insert(v1 = residue_types[i,], v2 = residue_types[j,]))
        self.edge_types = {'edge_types' : np.array(self.edge_types)}
        return self.edge_types

    def _get_edge_sequence_indicator(self, residue_ids: np.array) -> Dict[str, np.ndarray]:
        if hasattr(self, 'edge_sequence_indicator'):
            return self.edge_sequence_indicator

        edge_sequence_indicator =  []
        for i in range(residue_ids.shape[0]):
            for j in range(residue_ids.shape[0]):
                if i == j:
                    continue
                # if i - 1 = j (i is one to the left, or j is one to the right) or i + 1 = j (i is one to the right, or j right one to the right)
                if (i - 1 == j) or (i + 1 == j):
                    edge_sequence_indicator.append(1)
                # otherwise, no label
                else:
                    edge_sequence_indicator.append(0)
        self.edge_sequence_indicator = {'edge_sequence_indicator': np.array(edge_sequence_indicator)}
        return self.edge_sequence_indicator
    
    '''
    Residue Level: Node & Edge Feature SET Methods
    '''

    def _set_residue_level_node_features(self, residue_ids: np.array, dist_k_neighbors: Optional[int] = 30, seq_k_neighbors: Optional[int] = 10) -> None:

        if hasattr(self, 'residue_level_node_features'):
            return self.residue_level_node_features

        # convert residue_ids to indexes
        residue_indexes = np.array([self.index[res_id] for res_id in residue_ids])

        # nearest neighbor pairwise distances and angles ("nearest neighbor" based on Euclidean distances)
        nn_distance_pairwise_dists = self._get_nn_distance_pairwise_dists(nn_axis = residue_indexes, k_neighbors = dist_k_neighbors)
        nn_distance_pairwise_angles = self._get_nn_distance_pairwise_angles(nn_axis = residue_indexes, k_neighbors = dist_k_neighbors)
        nn_distance_pairwise_angles_spherical = self._get_nn_distance_pairwise_angles_spherical(nn_axis = residue_indexes, k_neighbors = dist_k_neighbors)
        
        # nearest neighbor pairwise distances and angles ("nearest neighbor" based on residue's position in sequence)
        nn_sequence_pairwise_dists = self._get_nn_sequence_pairwise_dists(nn_axis = residue_indexes, k_neighbors = seq_k_neighbors)
        nn_sequence_pairwise_angles = self._get_nn_sequence_pairwise_angles(nn_axis = residue_indexes, k_neighbors = seq_k_neighbors)
        nn_sequence_pairwise_angles_spherical = self._get_nn_sequence_pairwise_angles_spherical(nn_axis = residue_indexes, k_neighbors = seq_k_neighbors)
        
        # dihedral angles
        dihedrals = self._get_dihedral_angles(residue_ids = residue_ids)
        
        # COM distances & angles
        com_distances = self._get_bb_com_distances(residue_ids = residue_ids)
        com_angles = self._get_bb_com_angles(residue_ids = residue_ids)
        
        # contact map features
        contact_map_features = self._get_contact_maps(residue_ids = residue_ids, map_params = [{'coord_type': 'CA', 'threshold': 8}, {'coord_type': 'CB', 'threshold': 8}, {'coord_type': 'sc_com', 'threshold': 10}], n_groups = 3)
        
        features = [
            # distances & angles (NN w.r.t. Euclidean distance)
            nn_distance_pairwise_dists,
            nn_distance_pairwise_angles,
            nn_distance_pairwise_angles_spherical,
            # entropy on distances & angles
            {f'{key}_entropy(residue)' : entropy.compute_entropy(arr = arr, same_dist = False) for key, arr in nn_distance_pairwise_dists.items()},
            {f'{key}_entropy(residue)' : entropy.compute_entropy(arr = arr, same_dist = False) for key, arr in nn_distance_pairwise_angles.items()},
            # distances & angles (NN w.r.t. sequence position)
            nn_sequence_pairwise_dists,
            nn_sequence_pairwise_angles,
            nn_sequence_pairwise_angles_spherical,
            # entropy on distances & angles
            {f'{key}_entropy(residue)' : entropy.compute_entropy(arr = arr, same_dist = False) for key, arr in nn_sequence_pairwise_dists.items()},
            {f'{key}_entropy(residue)' : entropy.compute_entropy(arr = arr, same_dist = False) for key, arr in nn_sequence_pairwise_angles.items()},
            # dihedral angles
            dihedrals,
            # entropy on dihedrals
            {f'{key}_entropy(residues)' : entropy.compute_entropy(arr = arr, same_dist = False) for key, arr in dihedrals.items()},
            # COM distances
            com_distances,
            {f'{key}_entropy(residues)' : entropy.compute_entropy(arr = arr, same_dist = False) for key, arr in com_distances.items()},
            com_angles,
            {f'{key}_entropy(residues)' : entropy.compute_entropy(arr = arr, same_dist = False) for key, arr in com_angles.items()},
            # contact map features
            contact_map_features,
            {f'{key}_coords': self.coordinate_dict[key][residue_indexes,] for key in ['CA', 'C', 'O', 'N']}
        ]
        
        self.residue_level_node_features = {k: v for d in features for k, v in d.items()}

        residue_features = []
        for res_idx in residue_indexes:
            residue_features.append(self.residues[res_idx].get_features())

        residue_features_dict = dict(
            zip(
                residue_features[0].keys(), 
                [np.vstack([d[key] for d in residue_features]) for key in residue_features[0].keys()]
                )
            )
        self.residue_level_node_features.update(residue_features_dict)

    def _set_residue_level_edge_features(self, residue_ids: np.array) -> None:
        edge_distances = self._get_edge_distances(residue_ids = residue_ids)
        edge_angles = self._get_edge_angles(residue_ids = residue_ids)
        edge_spherical_angles = self._get_edge_spherical_angles(residue_ids = residue_ids)
        edge_types = self._get_edge_types(residue_ids = residue_ids)
        edge_sequence_indicator = self._get_edge_sequence_indicator(residue_ids = residue_ids)

        features = [
            edge_distances,
            {f'{key}_entropy(edge)' : entropy.compute_entropy(arr = arr, same_dist = False) for key, arr in edge_distances.items()},
            edge_angles,
            {f'{key}_entropy(edge)' : entropy.compute_entropy(arr = arr, same_dist = False) for key, arr in edge_angles.items()},
            edge_spherical_angles,
            {f'{key}_entropy(edge)' : entropy.compute_entropy(arr = arr, same_dist = False) for key, arr in edge_spherical_angles.items()},
            edge_types,
            edge_sequence_indicator
        ]
        self.residue_level_edge_features = {k: v for d in features for k, v in d.items()}
        
    '''
    Decoy Level: Node & Edge Feature SET Methods
    '''

    def set_decoy_level_node_features(self, features: Dict[str, np.ndarray]) -> None:
        self.decoy_level_node_features = features

    def set_decoy_level_edge_features(self, features: Dict[str, np.ndarray]) -> None:
        self.decoy_level_edge_features = features

    '''
    General Usage Functions
    '''

    def set_residue_features(self, residue_ids: np.array, dist_k_neighbors: Optional[int] = 30, seq_k_neighbors: Optional[int] = 10) -> None:

        # setting node features
        self._set_residue_level_node_features(residue_ids = residue_ids, dist_k_neighbors = dist_k_neighbors, seq_k_neighbors = seq_k_neighbors)

        # setting edge features
        self._set_residue_level_edge_features(residue_ids = residue_ids)
        
    def _set_feature_summary(self, feature_type: str, write_path: Optional[str] = None) -> None:
        features = self.node_features if feature_type == 'node' else self.edge_features
        dataframe = pd.DataFrame([
            {
                'name' : key,
                'n_obs' : val.shape[0],
                'n_cols' : val.shape[1]
            } for key, val in features.items()
        ])
        if write_path is not None:
            dataframe.to_csv(f'{write_path}/{feature_type}_feature_sumamry.csv', index = False)

    def _set_node_features(self, write_summary: Optional[bool] = False) -> None:
        if not hasattr(self, 'node_features'):
            assert hasattr(self, 'residue_level_node_features'), 'requires _set_residue_level_node_features() call'
            assert hasattr(self, 'decoy_level_node_features'), 'requires set_decoy_level_node_features() call'
            self.node_features = {**self.residue_level_node_features, **self.decoy_level_node_features}
            
            for key, value in self.node_features.items():
                if np.squeeze(value).ndim == 1:
                    value = np.reshape(value, (-1, 1))
                self.node_features[key] = value
            if write_summary:
                self._set_feature_summary('node', 'feature_summaries')
    
    def _set_edge_features(self, write_summary: Optional[bool] = False) -> None:
        if not hasattr(self, 'edge_features'):
            assert hasattr(self, 'residue_level_edge_features'), 'requires _set_residue_level_edge_features() call'
            assert hasattr(self, 'decoy_level_edge_features'), 'requires set_decoy_level_edge_features() call'
            self.edge_features = {**self.residue_level_edge_features, **self.decoy_level_edge_features}
            
            for key, value in self.edge_features.items():
                if np.squeeze(value).ndim == 1:
                    value = np.reshape(value, (-1, 1))
                self.edge_features[key] = value
            if write_summary:
                self._set_feature_summary('edge', 'feature_summaries')

    def get_node_features(self) -> None:
        """
        Get the node features for the graph.

        Returns:
        --------
        None

        Updates the attribute:
        - self.N: The concatenated node features as a numpy array.
        """
        self._set_node_features()
        if hasattr(self, 'N'):
            return self.N
        node_keys_sorted = sorted(list(self.node_features.keys()))
        hstack = []; node_feature_shapes = []
        for key in node_keys_sorted:
            v = self.node_features[key]
            hstack.append(v)
            node_feature_shapes.append({'feature_name': key, 'feature_dim': v.shape[1]})
        self.N = np.hstack(hstack)
        self.node_feature_shapes = node_feature_shapes
        return self.N

    def get_edge_features(self) -> None:
        """
        Get the edge features for the graph.

        Returns:
        --------
        None

        Updates the attribute:
        - self.E: The concatenated edge features as a numpy array.
        """
        self._set_edge_features()
        if hasattr(self, 'E'):
            return self.E
        edge_keys_sorted = sorted(list(self.edge_features.keys()))
        hstack = []; edge_features_shape = []
        for key in edge_keys_sorted:
            v = self.edge_features[key]
            hstack.append(v)
            edge_features_shape.append({'feature_name': key, 'feature_dim': v.shape[1]})

        self.E = np.hstack(hstack)
        self.edge_features_shape = edge_features_shape
        return self.E

    def get_graph_representation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the graph representation.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing the node features and edge features as numpy arrays.

        Calls the following methods:
        - get_node_features()
        - get_edge_features()
        """
        N = self.get_node_features()
        E = self.get_edge_features()
        return (N, E)


if __name__ == '__main__':
    pass

