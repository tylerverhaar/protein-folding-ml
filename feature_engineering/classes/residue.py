import numpy as np
import time

from collections import Counter
from typing import List, Dict, Any, Callable, Optional, Tuple

import feature_engineering.function_lib.njit_implementations.geometry as geometry
import feature_engineering.function_lib.njit_implementations.tools as arr_ops
from feature_engineering.classes.atom import Atom
from feature_engineering.constants import RESIDUE_TYPE_MAP


class Residue:

    def __init__(self, id: int, residue_type: str, chain_id: str, raw_atoms: List[Dict], in_loop: bool) -> None:
        """
        Initialize a residue object.
        
        Parameters:
            id (int): The identifier of the residue.
            residue_type (str): The type of the residue.
            chain_id (str): The chain identifier of the residue.
            raw_atoms (List[Dict]): The list of dictionaries containing raw atom information.
            in_loop (bool): Indicates if the residue is inside the decoy loop.
        
        Returns:
            None
        """
        self.id = id
        self.residue_type = residue_type
        self.chain_id = chain_id
        self.raw_atoms = raw_atoms
        self.in_loop = in_loop

        self.bb_atom_names = ['N', 'CA', 'C', 'O']
        self.no_sc = False
        self.__post_init__()

    def __post_init__(self) -> None:
        """
        Post-initialization method called after object creation.
        Performs the necessary setup steps.

        Returns:
            None
        """
        try:
            self._set_residue_type_one_hot()
            self._set_atoms()
            self._set_bb_coords()
            self._set_sc_coords()
            self._set_bb_coords_weighting_schemes()
            self._set_sc_coords_weighting_schemes()
            if self.in_loop:
                self._set_distance_adjustment_matrices()
        except Exception as e:
            print('Residue.__post_init__', e)
            raise Exception(e)

    def get_coord(self, key: str, weight: str) -> np.array:
        if key in self.bb_atoms_dict:
            return self.bb_atoms_dict[key].p,
        elif key == 'sc_com':
            return self._get_sc_com_coords(weight = weight)
        elif key == 'CB':
            return self._get_cb_coords(bb_com_weight = weight)
        else:
            raise NotImplemented(f'Residue.get_coord not implemented for key = {key}, weight = {weight}')

    def _set_atoms(self) -> None:
        """
        Set the atoms based on the raw atom data.

        Returns:
            None
        """
        self.atoms = []
        for d in self.raw_atoms:
            a = Atom(name = d['atom_name'], number = d['atom_number'], x_coord = d['x_coord'], y_coord = d['y_coord'], z_coord = d['z_coord'], residue_type = self.residue_type)
            a.set_atomic_attributes()

            for attr in ['atomic_radius', 'atomic_mass', 'covalent_radius', 'vdw_radius']:
                assert hasattr(a, attr)

            self.atoms.append(a)
        
        self.bb_atoms_dict = {}
        self.sc_atoms_dict = {}
        for atom in self.atoms:
            if atom.name in self.bb_atom_names:
                self.bb_atoms_dict[atom.name] = atom
            else:
                self.sc_atoms_dict[atom.name] = atom

    def _set_bb_coords(self) -> None:
        """
        Set the backbone and sidechain coordinates based on the atom dictionary.

        Returns:
            None
        """
        atom_keys = ['N', 'CA', 'C', 'O']
        present_keys = np.zeros(shape = 4)
        bb_coords = np.zeros((4, 3))

        for i, key in enumerate(atom_keys):
            if key in self.bb_atoms_dict:
                present_keys[i] = 1
                bb_coords[i] = self.bb_atoms_dict[key].p
            else:
                present_keys[i] = 0
        present_idx = (present_keys == 1)

        if present_keys.sum() < 4:
            centroid = np.mean(bb_coords[present_idx, ], axis=0)
            bb_coords[~present_idx,] = centroid

        self.bb_coords = bb_coords.astype('float64')

    def _set_sc_coords(self) -> None:
        """
        Set the side chain coordinates based on the atom dictionary.

        Returns:
            None
        """
        self.sc_coords = np.array([atom.p for atom in self.sc_atoms_dict.values()]).astype('float64')
        if self.sc_coords.shape[0] == 0:
            self.sc_coords = self.bb_coords
            self.no_sc = True

    def _set_bb_coords_weighting_schemes(self, weight_methods: List[Any] = ['uniform', 'mass']) -> None:
        atom_keys = ['N', 'CA', 'C', 'O']
        weighting_schemes = dict()
        for method in weight_methods:
            present_keys = np.zeros(shape = 4)
            bb_coords_weights = np.zeros(shape = 4)

            # extract weights
            for i, key in enumerate(atom_keys):
                if key in self.bb_atoms_dict:
                    present_keys[i] = 1
                    weight = 1 if method == 'uniform' else self.bb_atoms_dict[key].atomic_mass
                    bb_coords_weights[i] = weight
                else:
                    present_keys[i] = 0
            present_idx = (present_keys == 1)

            if present_keys.sum() < 4:
                centroid = np.mean(bb_coords_weights[present_idx, ], axis=0)
                bb_coords_weights[~present_idx,] = centroid

            weighting_schemes[method] = bb_coords_weights.astype('float64').reshape(-1,1)

        self.bb_weighting_schemes = weighting_schemes

    def _set_sc_coords_weighting_schemes(self, weight_methods: List[Any] = ['uniform', 'mass']) -> None:
        """
        Set the side chain coordinates weights based on the atom dictionary.

        Returns:
            None
        """
        if self.no_sc:
            self.sc_coords_weights = self.bb_weighting_schemes
        
        weighting_schemes = dict()
        for method in weight_methods:
            w = []
            for atom in self.sc_atoms_dict.values():
                weight = 1 if method == 'uniform' else atom.atomic_mass
                w.append(weight)
            
            weighting_schemes[method] = np.array(w).astype('float64').reshape(-1,1)
        self.sc_weighting_schemes = weighting_schemes

    def _set_residue_type_one_hot(self) -> None:
        """
        Set the one-hot encoding for the residue type.

        Returns:
            None
        """
        
        if self.residue_type not in RESIDUE_TYPE_MAP:
            self.residue_type = 'LEU'
        
        self.residue_type_one_hot = np.zeros(shape = len(RESIDUE_TYPE_MAP))
        self.residue_type_one_hot[RESIDUE_TYPE_MAP[self.residue_type] - 1] = 1

    def _set_distance_adjustment_matrices(self, radius_types: List[str] = ['atomic_radius','vdw_radius','covalent_radius']) -> None:
        self.adjustment_matrices = {}
        atom_keys = ['N', 'CA', 'C', 'O']
        for radius_type in radius_types:
            dA = np.zeros((4, 4))
            for i in range(0,len(atom_keys)):
                for j in range(i+1,len(atom_keys)):
                    ri = getattr(self.bb_atoms_dict[atom_keys[i]], radius_type)
                    rj = getattr(self.bb_atoms_dict[atom_keys[j]], radius_type)
                    dA[i,j] = ri + rj
            self.adjustment_matrices[radius_type] = dA    

    def _get_potential_well_depth(self) -> np.ndarray:
        if not self.in_loop:
            raise Exception('cannot compute well-defined potential well depth matrix')
        
        epsilon = np.zeros((4,4))
        atom_keys = ['N', 'CA', 'C', 'O']
        for i in range(0,len(atom_keys)):
            for j in range(i+1,len(atom_keys)):
                ei = getattr(self.bb_atoms_dict[atom_keys[i]], 'potential')
                ej = getattr(self.bb_atoms_dict[atom_keys[j]], 'potential')
                epsilon[i,j] = np.sqrt(ei * ej)
        return epsilon

    def _get_bb_com_coords(self, weight: str) -> np.ndarray:
        bb_coords = self.bb_coords
        m = self.bb_weighting_schemes['uniform'] if weight == 'uniform' else self.bb_weighting_schemes['mass']
        return geometry.centre_of_mass(p = bb_coords, m = m)

    def _get_sc_com_coords(self, weight: str) -> np.ndarray:
        if self.no_sc:
            return self._get_bb_com_coords(weight = weight)
        sc_coords = self.sc_coords
        m = self.sc_weighting_schemes['uniform'] if weight == 'uniform' else self.sc_weighting_schemes['mass']
        return geometry.centre_of_mass(p = sc_coords, m = m)

    def _get_com_coords(self, weight: str) -> np.ndarray:
        # backbone atoms
        bb_coords = self.bb_coords
        bb_w = self.bb_weighting_schemes[weight]
        
        # side-chain atoms
        sc_coords = self.sc_coords
        sc_w = self.sc_weighting_schemes[weight]

        # all atoms
        coords = np.vstack([bb_coords, sc_coords])
        w = np.vstack([bb_w, sc_w])
        return geometry.centre_of_mass(p = coords, m = w)

    def _get_cb_coords(self, bb_com_weight: str) -> np.array:
        if 'CB' in self.sc_atoms_dict:
            return self.sc_atoms_dict['CB'].p
        elif 'CA' in self.bb_atoms_dict:
            return self.bb_atoms_dict['CA'].p
        else:
            return self._get_bb_com_coords(weight = bb_com_weight)

    def get_features(self, **kwargs: Dict[str,Any]) -> Dict[str, np.ndarray]:
        t0 = time.time()
        features = {'residue_type': self.residue_type_one_hot}
        # backbone atoms
        bb_coords = self.bb_coords
        bb_wu = self.bb_weighting_schemes['uniform']
        bb_wm = self.bb_weighting_schemes['mass']

        # side-chain atoms
        sc_coords = self.sc_coords
        sc_wu = self.sc_weighting_schemes['uniform']
        sc_wm = self.sc_weighting_schemes['mass']

        # all atoms
        coords = np.vstack([bb_coords, sc_coords])
        wu = np.vstack([bb_wu, sc_wu])
        wm = np.vstack([bb_wm, sc_wm])
        
        features.update({'bb_coords': bb_coords.flatten(), 'n_sc_atoms': np.array([sc_coords.shape[0]]), 'n_atoms': np.array([coords.shape[0]])})

        # backbone centre of mass & backbone atoms distances-angle orientation w.r.t. COM under uniform weigthing
        bb_com_coords_wu = geometry.centre_of_mass(p = bb_coords, m = bb_wu)
        bb_coords_bb_COM_distance_wu = geometry.point_array_distance(pt = bb_com_coords_wu, arr = self.bb_coords)
        bb_coords_bb_COM_angle_wu = geometry.point_array_angle(pt = bb_com_coords_wu, arr = bb_coords)
        features.update({
            'bb_com_coords_wu': bb_com_coords_wu, 
            'bb_coords_bb_COM_distance_wu': bb_coords_bb_COM_distance_wu, 
            'bb_coords_bb_COM_angle_wu': bb_coords_bb_COM_angle_wu
            })

        # backbone centre of mass & backbone atoms distance-angle orientation w.r.t. COM under atomic mass weigthing
        bb_com_coords_wm = geometry.centre_of_mass(p = bb_coords, m = bb_wm)
        bb_coords_bb_COM_distance_wm = geometry.point_array_distance(pt = bb_com_coords_wm, arr = bb_coords)
        bb_coords_bb_COM_angle_wm = geometry.point_array_angle(pt = bb_com_coords_wm, arr = bb_coords)
        features.update({
            'bb_com_coords_wm': bb_com_coords_wm, 
            'bb_coords_bb_COM_distance_wm': bb_coords_bb_COM_distance_wm, 
            'bb_coords_bb_COM_angle_wm': bb_coords_bb_COM_angle_wm
            })
        
        # centre of mass of residue & backbone atoms distance-angle orientation w.r.t. COM under uniform weighting
        com_coords_wu = bb_com_coords_wu if self.no_sc else geometry.centre_of_mass(p = coords, m = wu)
        bb_coords_COM_distance_wu = geometry.point_array_distance(pt = com_coords_wu, arr = bb_coords)
        bb_coords_COM_angle_wu = geometry.point_array_angle(pt = com_coords_wu, arr = bb_coords)
        features.update({
            'com_coords_wu': com_coords_wu, 
            'bb_coords_COM_distance_wu': bb_coords_COM_distance_wu, 
            'bb_coords_COM_angle_wu': bb_coords_COM_angle_wu
            })
        
        # backbone centre of mass & backbone atoms distance-angle orientation w.r.t. COM under atomic mass weigthing
        com_coords_wm = bb_com_coords_wm if self.no_sc else geometry.centre_of_mass(p = coords, m = wm)
        bb_coords_COM_distance_wm = geometry.point_array_distance(pt = com_coords_wm, arr = bb_coords)
        bb_coords_COM_angle_wm = geometry.point_array_angle(pt = com_coords_wm, arr = bb_coords)
        features.update({
            'com_coords_wm': com_coords_wm, 
            'bb_coords_COM_distance_wm': bb_coords_COM_distance_wm, 
            'bb_coords_COM_angle_wm': bb_coords_COM_angle_wm
            })

        # centre of mass for side-chain structure & backbone COM distance-angle w.r.t. COM under uniform weighting
        sc_com_coords_wu = geometry.centre_of_mass(p = sc_coords, m = sc_wu) if not self.no_sc else bb_com_coords_wu
        bb_com_sc_com_distance_wu = geometry.p_norm(p1 = bb_com_coords_wu, p2 = sc_com_coords_wu, p = 2)
        bb_com_sc_com_angle_wu = geometry.vector_angle(v1 = bb_com_coords_wu, v2 = sc_com_coords_wu)
        features.update({
            'sc_com_coords_wu': sc_com_coords_wu, 
            'bb_com_sc_com_distance_wu': np.array([bb_com_sc_com_distance_wu]), 
            'bb_com_sc_com_angle_wu': np.array([bb_com_sc_com_angle_wu])
            })

        # centre of mass for side-chain structure & backbone COM distance-angle w.r.t. COM under atomic weighting
        sc_com_coords_wm = geometry.centre_of_mass(p = sc_coords, m = sc_wm) if not self.no_sc else bb_com_coords_wm
        bb_com_sc_com_distance_wm = geometry.p_norm(p1 = bb_com_coords_wm, p2 = sc_com_coords_wm, p = 2)
        bb_com_sc_com_angle_wm = geometry.vector_angle(v1 = bb_com_coords_wm, v2 = sc_com_coords_wm)
        features.update({
            'sc_com_coords_wm': sc_com_coords_wu, 
            'bb_com_sc_com_distance_wm': np.array([bb_com_sc_com_distance_wm]), 
            'bb_com_sc_com_angle_wm': np.array([bb_com_sc_com_angle_wm])
            })


        # pairwise distance matrix
        bb_pairwise_distance_matrix = geometry.pairwise_distance_matrix(pts = bb_coords)
        features['bb_pairwise_distance_matrix'] = arr_ops.flattened_upper_triangle(bb_pairwise_distance_matrix, k = 1)

        # pairwise distance matrix adjusted for radius around each bacbbone atoms
        #bb_pairwise_distance_matrix_atomic_radius_adjustment = bb_pairwise_distance_matrix - self.adjustment_matrices['atomic_radius']
        bb_pairwise_distance_matrix_vdw_radius_adjustment = bb_pairwise_distance_matrix - self.adjustment_matrices['vdw_radius']
        #bb_pairwise_distance_matrix_covalent_radius_adjustment = bb_pairwise_distance_matrix - self.adjustment_matrices['covalent_radius']
        features.update({
            #'bb_pairwise_distance_matrix_atomic_radius_adjustment': arr_ops.flattened_upper_triangle(bb_pairwise_distance_matrix_atomic_radius_adjustment, k = 1),
            'bb_pairwise_distance_matrix_vdw_radius_adjustment': arr_ops.flattened_upper_triangle(bb_pairwise_distance_matrix_vdw_radius_adjustment, k = 1),
            #'bb_pairwise_distance_matrix_covalent_radius_adjustment': arr_ops.flattened_upper_triangle(bb_pairwise_distance_matrix_covalent_radius_adjustment, k = 1),
        })

        # pairwise Van de Walls forces
        epsilon = self._get_potential_well_depth()
        sigma = self.adjustment_matrices['vdw_radius'] / 2
        bb_lennard_jones_potential = geometry.lennard_jones_potential(DM = bb_pairwise_distance_matrix, epsilon = epsilon, sigma = sigma)
        features['bb_lennard_jones_potential'] = arr_ops.flattened_upper_triangle(bb_lennard_jones_potential, k = 1)
        features['bb_lennard_jones_potential_total'] = np.array([np.sum(features['bb_lennard_jones_potential'])])

        # pairwise angle matrix for backbone atoms
        bb_pairwise_angle_matrix = geometry.pairwise_angle_matrix(pts = bb_coords)
        features['bb_pairwise_angle_matrix'] = arr_ops.flattened_upper_triangle(bb_pairwise_angle_matrix, k = 1)

        
        # radius of gyration of backbone, side-chain, and all atoms w.r.t. uniform weighting
        bb_radius_of_gyration_wu = geometry.radius_of_gyration(p = bb_coords, m = bb_wu)
        sc_radius_of_gyration_wu = 0 if self.no_sc else geometry.radius_of_gyration(p = sc_coords, m = sc_wu) if not self.no_sc else bb_radius_of_gyration_wu
        radius_of_gyration_wu = bb_radius_of_gyration_wu if self.no_sc else geometry.radius_of_gyration(p = coords, m = wu)
        features.update({
            'bb_radius_of_gyration_wu': np.array([bb_radius_of_gyration_wu]), 
            'sc_radius_of_gyration_wu': np.array([sc_radius_of_gyration_wu]), 
            'radius_of_gyration_wu': np.array([radius_of_gyration_wu])
            })

        # radius of gyration of backbone, side-chain, and all atoms w.r.t. atomic weighting
        bb_radius_of_gyration_wm = geometry.radius_of_gyration(p = bb_coords, m = bb_wm)
        sc_radius_of_gyration_wm = 0 if self.no_sc else geometry.radius_of_gyration(p = sc_coords, m = sc_wm) if not self.no_sc else bb_radius_of_gyration_wm
        radius_of_gyration_wm = bb_radius_of_gyration_wm if self.no_sc else geometry.radius_of_gyration(p = coords, m = wu)
        features.update({
            'bb_radius_of_gyration_wm': np.array([bb_radius_of_gyration_wm]), 
            'sc_radius_of_gyration_wm': np.array([sc_radius_of_gyration_wm]), 
            'radius_of_gyration_wm': np.array([radius_of_gyration_wm])
            })
        
        # backbone gyration tensor w.r.t. uniform and atomic mass weighting
        bb_gyration_tensor_wu = geometry.gyration_tensor(p = bb_coords, m = bb_wu)
        bb_gyration_tensor_wm = geometry.gyration_tensor(p = bb_coords, m = bb_wm)
        features.update({
            'bb_gyration_tensor_wu': arr_ops.flattened_upper_triangle(bb_gyration_tensor_wu, k = 0), 
            'bb_gyration_tensor_wm': arr_ops.flattened_upper_triangle(bb_gyration_tensor_wm, k = 0)
            })

        # residue gyration tensor w.r.t. uniform and atomic mass weighting
        gyration_tensor_wu = bb_gyration_tensor_wu if self.no_sc else geometry.gyration_tensor(p = coords, m = wu)
        gyration_tensor_wm = bb_gyration_tensor_wm if self.no_sc else geometry.gyration_tensor(p = coords, m = wm)
        features.update({
            'gyration_tensor_wu': arr_ops.flattened_upper_triangle(gyration_tensor_wu, k = 0), 
            'gyration_tensor_wm': arr_ops.flattened_upper_triangle(gyration_tensor_wm, k = 0)
            })

        features['no_sc'] = np.array([self.no_sc * 1])
        t1 = time.time()
        
        return features

    def set_residue_level_features(self, weighting: Optional[bool] = False) -> None:
        
        self.residue_level_features = {
            'bb_coords' : self.bb_coords,
            'bb_com_coords': self._get_bb_com_coords(),
            'sc_com_coords': self._get_sc_com_coords(),
            'bb_pairwise_distance_matrix': self._get_bb_pairwise_distance_matrix(),
            'bb_pairwise_angle_matrix': self._get_bb_pairwise_angle_matrix(),
            'bb_sc_com_distance': self._get_bb_sc_com_distance(),
            'bb_sc_com_angle': self._get_bb_sc_com_angle(),
            'bb_radius_of_gyration': self._get_bb_radius_of_gyration(),
            'sc_radius_of_gyration': self._get_sc_radius_of_gyration(),
            'bb_gyration_tensor': self._get_bb_gyration_tensor(),
            'bb_inertia_tensor': self._get_bb_inertia_tensor(),
            'no_sc' : self.no_sc * 1
        }
        
    
if __name__ == '__main__':
    pass


            
            
        
        
        


        

