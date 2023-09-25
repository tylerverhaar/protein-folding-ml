import numpy as np
import pandas as pd
import time, sys
from typing import Dict, Any, List
import feature_engineering.function_lib.njit_implementations.entropy as entropy
from feature_engineering.classes.decoy_sequence import DecoySequence



class DecoySet:

    def __init__(self, decoy_dict: Dict[int, DecoySequence]) -> None:
        """
        Initialize a DecoySet object.

        Parameters:
        -----------
        decoy_dict : Dict[int, DecoySequence]
            A dictionary mapping decoy IDs to DecoySequence objects.
        """
        self.decoy_dict = decoy_dict

    def set_decoy_entropy_values(self, **kwargs: Dict[str, Any]) -> None:
        """
        Set the decoy entropy values.

        Parameters:
        -----------
        **kwargs : Dict[str, Any]
            Additional keyword arguments.

        Calls the following methods:
        - set_decoy_node_entropy_values()
        - set_decoy_edge_entropy_values()
        """
        self.set_decoy_node_entropy_values()
        self.set_decoy_edge_entropy_values()

    def set_decoy_node_entropy_values(self, additional_keys: List[str] = None, **kwargs: Dict[str, Any]) -> None:
        """
        Calculate entropy values for selected features in each decoy sequence's residue_level_features dictionary
        and assign them to the corresponding decoy level features.

        Args:
            additional_keys (List[str], optional): Additional feature keys to include in entropy calculation.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            None
        """
        
        # Initialize the dictionary to store decoy entropy values
        decoy_entropy_values = {decoy_id: {} for decoy_id in self.decoy_dict.keys()}

        def filter_func(key: str) -> bool:
            if 'entropy' in key:
                return False
            if '_com_g' in key:
                return False
            if '_shift_g' in key:
                return False
            if '_angle_g' in key:
                return False
            if 'com_spherical_repr_g' in key:
                return False
            if key in ['CA_coords','C_coords','N_coords','O_coords','bb_coords', 'n_sc_atoms', 'n_atoms', 'bb_com_coords_wu', 'bb_com_coords_wm', 'com_coords_wu', 'com_coords_wm', 'sc_com_coords_wu', 'sc_com_coords_wm', 'bb_gyration_tensor_wu', 'bb_gyration_tensor_wm', 'gyration_tensor_wu', 'gyration_tensor_wm', 'no_sc']:
                return False
            return True
        # Get the selected feature keys
        selected_feature_keys = [key for key in self.decoy_dict[0].residue_level_node_features.keys() if filter_func(key = key)]
        if additional_keys is not None and isinstance(additional_keys, list):
            selected_feature_keys = selected_feature_keys + additional_keys

        # Iterate over each selected feature key
        for key in selected_feature_keys:
            # mapping between indices and decoy IDs
            index_decoy_map = {}

            # Create an empty list to store the feature values for each decoy
            values = []

            # Initialize the index counter
            i = 0

            # Iterate over each decoy ID and decoy sequence
            for decoy_id, decoy_sequence in self.decoy_dict.items():
                # Get the feature values for the current key, update index tracker
                vals = decoy_sequence.residue_level_node_features[key]
                n_vals = vals.shape[0]
                values.append(vals)
                index_decoy_map[decoy_id] = np.arange(i, i + n_vals)
                i += n_vals

            # Stack the values vertically
            values = np.vstack(values)

            zero_check = np.allclose(values, 0, atol = 1e-5)

            if zero_check:
                entropy_values = values

            else:
                try:
                    if 'residue_contacts' in key:
                        entropy_values = entropy.compute_entropy(arr = values, labelled_arr = True, same_dist = True)
                    elif 'size' in key:
                        entropy_values = entropy.compute_entropy(arr = values, labelled_arr = True, same_dist = True)
                    elif 'residue_type' in key:
                        entropy_values = entropy.compute_entropy(arr = values, labelled_arr = True, same_dist = True)
                    elif '_com_g' in key or '_com_spherical_repr' in key or '_shift_g' in key:
                        continue
                    else:
                        entropy_values = entropy.compute_entropy(arr = values)
                except:
                    print(f'problem with this key: {key}')
                    #import pdb; pdb.set_trace()
                    #x = 1
                    raise Exception(f'problem with this key: {key}')


            # Assign the entropy values to the corresponding feature key in decoy_entropy_values
            for decoy_id in decoy_entropy_values.keys():
                decoy_entropy_values[decoy_id][f'{key}_entropy(decoy)'] = entropy_values[index_decoy_map[decoy_id],]

        # Update the decoy-level features in each decoy sequence
        for decoy_id, decoy_sequence in self.decoy_dict.items():
            decoy_sequence.set_decoy_level_node_features(features=decoy_entropy_values[decoy_id])

    def set_decoy_edge_entropy_values(self, additional_keys: List[str] = None, **kwargs: Dict[str, Any]) -> None:
        def filter_func(key: str) -> bool:
            if 'entropy' in key:
                return False
            if key in ['bb_com_coords', 'sc_com_coords', 'bb_pairwise_distance_matrix', 'bb_pairwise_angle_matrix', 'bb_sc_com_distance', 'bb_sc_com_angle', 'bb_radius_of_gyration', 'sc_radius_of_gyration', 'bb_gyration_tensor', 'bb_inertia_tensor', 'no_sc', 'edge_sequence_indicator']:
                return False
            return True
        
        # Initialize the dictionary to store decoy entropy values
        decoy_entropy_values = {decoy_id: {} for decoy_id in self.decoy_dict.keys()}

        # Get the selected feature keys that don't contain 'entropy'
        selected_feature_keys = [key for key in self.decoy_dict[0].residue_level_edge_features.keys() if filter_func(key = key)]
        
        if additional_keys is not None and isinstance(additional_keys, list):
            selected_feature_keys = selected_feature_keys + additional_keys

        # Iterate over each selected feature key
        for key in selected_feature_keys:

            # mapping between indices and decoy IDs
            index_decoy_map = {}

            # Create an empty list to store the feature values for each decoy
            values = []

            # Initialize the index counter
            i = 0

            # Iterate over each decoy ID and decoy sequence
            for decoy_id, decoy_sequence in self.decoy_dict.items():
                # Get the feature values for the current key, update index tracker
                
                vals = decoy_sequence.residue_level_edge_features[key]
                try:
                    n_vals = vals.shape[0]
                except:
                    raise AttributeError('n_vals has no attribute "shape", likely not a numpy.ndarray')
                values.append(vals)
                index_decoy_map[decoy_id] = np.arange(i, i + n_vals)
                i += n_vals

            # Stack the values vertically
            values = np.vstack(values)
            try:
                # Calculate the entropy values
                entropy_value = entropy.compute_entropy(arr = values) if key != 'edge_types' else entropy.compute_one_hot_entropy(arr = values)
            except Exception as e:
                raise Exception(e)

            # Assign the entropy values to the corresponding feature key in decoy_entropy_values
            for decoy_id in decoy_entropy_values.keys():
                decoy_entropy_values[decoy_id][f'{key}_entropy(decoy)'] = entropy_value[index_decoy_map[decoy_id],]

        # Update the decoy-level features in each decoy sequence
        for decoy_id, decoy_sequence in self.decoy_dict.items():
            decoy_sequence.set_decoy_level_edge_features(features=decoy_entropy_values[decoy_id])
      
    def write_graph_representations(self, decoy_file_id: str, write_path: str) -> None:
        """
        Write the graph representations of the decoy sequences to a file.

        Parameters:
        -----------
        decoy_file_id : str
            The identifier of the decoy file.
        write_path : str
            The path where the graph representations will be written.

        Saves the graph representations as compressed NumPy arrays in the specified write path.

        The graph representations include:
        - E: Edge features
        - N: Node features
        - I: Information features

        Each feature array corresponds to a decoy sequence and is stacked along the first axis.

        Note: The graph representations are obtained by calling the 'get_write_data()' method of each decoy sequence.

        Example usage:
            decoy_set.write_graph_representations(decoy_file_id='decoy123', write_path='/path/to/save')

        """
        
        save_path = f'{write_path}/{decoy_file_id}'
        E_arrays = []
        N_arrays = []
        I_arrays = []

        # Collect the edge, node, and information features from each decoy sequence
        for i, decoy_sequence in enumerate(self.decoy_dict.values()):
            N, E, I = decoy_sequence.get_write_data()
            if i == 0:
                node_feature_data = pd.DataFrame(decoy_sequence.node_feature_shapes)
                node_feature_data['tag'] = 'node'
                edge_feature_data = pd.DataFrame(decoy_sequence.edge_features_shape)
                edge_feature_data['tag'] = 'edge'
                feature_data = pd.concat([node_feature_data, edge_feature_data])
                feature_data.to_csv(f'{write_path}/feature_shapes.csv', index = False)

            E_arrays.append(E)
            N_arrays.append(N)
            I_arrays.append(I)

        # Stack the feature arrays along the first axis
        E_stack = np.stack(E_arrays, axis=0).astype('float32')
        N_stack = np.stack(N_arrays, axis=0).astype('float32')
        I_stack = np.stack(I_arrays, axis=0).astype('float32')
        
        # Save the stacked feature arrays as compressed NumPy arrays
        np.savez_compressed(f'{save_path}.npz', E = E_stack, N = N_stack, I = I_stack)


        
        



