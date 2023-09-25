import random, time
import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, MinMaxScaler, Normalizer, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from spektral.data import Dataset, Graph
from tqdm import tqdm
from typing import Optional, Callable, List, Dict, Any, Tuple, Union
from multiprocessing import Pool, cpu_count
from modelling.tools import adjacency_matrix_map, rmsd_to_y_onehot, LRMSDScaler, GraphWrapper

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
import concurrent.futures

class Decoys(Dataset):
    

    def __init__(self, graph_path: str, transforms = None, **kwargs: Dict[str, Any]):
        """
        Initializes a new instance of the Decoys class.

        Args:
            graph_path (str): The path to the directory containing the graph data.
            transforms (optional): The data transformation pipeline from spektral. Default is None.
            **kwargs: Additional keyword arguments.
        """

        # path management
        assert os.path.exists(graph_path)
        self.graph_path = graph_path
        self.name = kwargs.get('name', graph_path.split('/')[-2])
        self.data_dir = '/'.join(graph_path.split('/')[:-1])
        self.read_saved_indexing = kwargs.get('read_saved_indexing', False)
        self.graph_indexing_path = kwargs.get('graph_indexing_path', f'{self.graph_path}/decoys_n1181000.csv')
        self.energy_evals = kwargs.get('energy_evals', False)

        # dataset filtering and train-test generation
        self.n_decoys = kwargs.get('n_decoys', -1)
        self.invalid_proteins = kwargs.get('invalid_proteins', [])
        self.shuffle = kwargs.get('shuffle', False)
        self.seed = kwargs.get('seed', 42)
        self.train_size = kwargs.get('train_size', 0.8)
        self.sampling = kwargs.get('sampling', False)
        self.sample_size = kwargs.get('sample_size', 0.8)
        self.test_set = kwargs.get('test_set', False)
        
        # labelling of response variable
        self.labelling = kwargs.get('labelling', 'continuous')
        self.output_dim = 1
        self.activation = 'relu'
        self.bin_params = kwargs.get('bin_params', {'n_bins': 4})
        self.scaling = kwargs.get('scaling', None)

        # adjacency and edge matrix construction
        self.edge_features = kwargs.get('edge_features', True)
        self.n_neighbors = kwargs.get('n_neighbors', 6)
        assert self.n_neighbors % 2 == 0, 'requires even number for n_neighbors'
        self.fully_connected = kwargs.get('fully_connected', False)
        self.adjacency_matrix_format = kwargs.get('adjacency_matrix_format', 'binary')
        self.n_nodes_range = range(4,40)
        
        # feature transforms

        self.node_level_rmsd_regression = kwargs.get('node_level_rmsd_regression', False)
        self.node_regr_fpath = kwargs.get('node_regr_fpath', None)
        self._load_node_level_regr_model()
        

        self.transform_node_features = kwargs.get('transform_node_features', False)
        self.node_transform_methods = kwargs.get('node_transform_methods', 'standard')
        self.node_feature_indices = kwargs.get('node_feature_indices', None)
        self.transform_edge_features = kwargs.get('transform_edge_features', False)
        self.edge_transform_methods = kwargs.get('edge_transform_methods', 'standard')
        self.edge_feature_indices = kwargs.get('edge_feature_indices', None)
        
        self.transforms = {
            'function': FunctionTransformer(func = kwargs.get('func', np.log)),
            'log': FunctionTransformer(func = np.log),
            'max_abs': MaxAbsScaler(),
            'min_max': MinMaxScaler(),
            'normalizer': Normalizer(norm = kwargs.get('norm', 'l2')),
            'box_cox': PowerTransformer(method = 'box-cox'),
            'yeo_johnson': PowerTransformer(method = 'yeo-johnson'),
            'quantile': QuantileTransformer(n_quantiles = kwargs.get('n_quantiles', 20), output_distribution = kwargs.get('output_distribution', 'uniform')),
            'robust': RobustScaler(),
            'standard': StandardScaler()
        }

        # set transforms to None
        self.ev_transform = None
        self.scaler = None

        # clustering
        self.node_level_clustering = kwargs.get('node_level_clustering', False)
        self.graph_level_clustering = kwargs.get('graph_level_clustering', False)
        self.n_processes = kwargs.get('n_processes', None)
        
        # set the seed and precomputed indexing for adjacency and edge matrix construction
        random.seed(self.seed)
        self._set_adjacency_matrices(n_neighbors = self.n_neighbors, format_type = self.adjacency_matrix_format)
        self._set_edge_indexing(n_neighbors = self.n_neighbors)
        self._set_feature_indexing()

        #node_group_patterns = [('nn(dist)_dists', r"^nn\(dist\)_(.*?)_(.*?)_dists$"), ('dihederals', r"\b(omega|phi|psi)\b")]
        #unions = [['nn(dist)_dists', 'dihederals']]
        #self._set_node_feature_groups(feature_patterns = node_group_patterns, feature_unions = unions)
        #self.col_index = self.node_feature_groups['nn(dist)_dists__dihederals']

        super().__init__(transforms, **kwargs)


    def _load_node_level_regr_model(self) -> None:
        if not self.node_level_rmsd_regression:
            return
        assert os.path.exists(self.node_regr_fpath), FileNotFoundError(f'model at {self.node_regr_fpath} does not exist')
        self.model = xgb.Booster()
        self.model.load_model(self.node_regr_fpath)


    def _get_decoy_filenames(self, suffix: Optional[str] = '.npz') -> List[str]:
        """
        Retrieves the filenames of the decoy files in the graph directory.

        Args:
            suffix (str, optional): The file suffix to filter the filenames. Default is '.npz'.

        Returns:
            List[str]: A list of decoy filenames.

        """
        decoy_filenames = list(); protein_ids = set()
        for fname in os.listdir(self.graph_path):
            if not fname.endswith(suffix):
                continue
            protein_id = fname[0:4].upper(); protein_ids.add(protein_id)
            if protein_id in self.invalid_proteins:
                continue
            decoy_filenames.append(fname)
            if self.n_decoys > 0 and len(decoy_filenames) == self.n_decoys:
                break
        
        protein_ids = list(protein_ids)
        if self.shuffle:
            random.shuffle(protein_ids)
        key_func = lambda x: protein_ids.index(x[0:4].upper())
        decoy_filenames = sorted(decoy_filenames, key = key_func)
        return decoy_filenames
    
    def _set_feature_indexing(self) -> None:
        """
        Set the feature indexing for node and edge features.

        Reads the feature shapes data from a CSV file and computes the start and end indexes for each feature.
        The computed feature indexing is stored in the 'feature_indexing' attribute as a dictionary.

        Parameters:
            None

        Returns:
            None
        """
        feature_data = pd.read_csv(f'{self.graph_path}/feature_shapes.csv')
        self.feature_indexing = {}
        for key in ['node', 'edge']:
            self.feature_indexing[key] = {}
            data = feature_data[feature_data['tag'] == key].copy().reset_index()
            data['end_index'] = np.cumsum(data['feature_dim'])
            data['start_index'] = data['end_index'] - data['feature_dim']
            self.feature_indexing[key]['n_features'] = data.iloc[-1]['end_index']
            self.feature_indexing[key]['feature_index_map'] = data.set_index("feature_name").apply(lambda x: np.arange(x["start_index"], x["end_index"]), axis=1).to_dict()
            self.feature_indexing[key]['index_feature_map'] = {v: key for key, value in self.feature_indexing[key]['feature_index_map'].items() for v in value}
        
    def _set_node_feature_groups(self, feature_patterns: List[Tuple[str, str]], feature_unions: Optional[List[List[str]]] = None) -> None:
        """
        Group features based on provided patterns and create unions of feature groups if specified.

        Parameters:
            feature_patterns (List[Tuple[str, str]]): A list of tuples containing a tag and a regex pattern
                to match feature names and group them accordingly.
            feature_unions (Optional[List[List[str]]]): A list of lists, where each inner list contains tags of previously
                created feature groups that need to be combined into a new feature group (union).

        Returns:
            None
        """
        groups = {tag: [] for tag, _ in feature_patterns}
        for feature in self.feature_indexing['node']['feature_index_map']:
            for tag, tag_pattern in feature_patterns:  # Renamed pattern to tag_pattern
                if re.match(tag_pattern, feature):
                    groups[tag].append(self.feature_indexing['node']['feature_index_map'][feature])
        self.node_feature_groups = {tag: np.sort(np.concatenate(indexes)) for tag, indexes in groups.items()}

        if feature_unions is None or (isinstance(feature_unions, list) and len(feature_unions) == 0):
            return
        
        for union_tags in feature_unions:  # Renamed feature_union to union_tags
            self.node_feature_groups['__'.join(union_tags)] = np.sort(np.unique(np.concatenate([self.node_feature_groups[tag] for tag in union_tags])))

    def _get_load_jobs(self, data: pd.DataFrame, **kwargs: Dict[str,Any]) -> Tuple[List[Dict[str,Any]], pd.DataFrame]:
        jobs = []

        data = data[data['sampled']].copy()
        data['start_residue'] = data['start_residue'].astype(int)
        data['end_residue'] = data['end_residue'].astype(int)
        data['decoy_loop_ids'] = data.apply(lambda row: '_'.join([str(row['protein_id']), str(row['start_residue']), str(row['end_residue'])]), axis=1)

        for decoy_loop_id in tqdm(data['decoy_loop_ids'].unique(), desc = 'loading graphs'):
            loc = data[data['decoy_loop_ids'] == decoy_loop_id]
            index = loc['decoy_id'].values - 1
            jobs.append({'npz_file': f'{decoy_loop_id}.npz', 'index': index})

        return jobs, data

    def _get_derived_indexing(self, feature_index_map: Dict[str, np.array], features: List[List[str]]) -> Dict[Tuple[str], np.array]:
        """
        Compute the derived feature indexing for the given list of features.

        Parameters:
            feature_index_map (Dict[str, np.array]): A dictionary mapping feature names to their corresponding indices.
            features (List[List[str]]): List of features to compute the derived indexing for.

        Returns:
            Dict[Tuple[str], np.array]: A dictionary mapping tuples of feature names to their corresponding derived indices.
        """
        indexing = {}
        for fc in features:
            indexing[tuple(fc)] = np.sort(np.hstack([feature_index_map[f] for f in fc]))
        return indexing
    
    def _get_feature_collection_indexes(self, node_features: List[Union[str, List[str]]] = None, edge_features: List[Union[str, List[str]]] = None) -> Dict[str, Dict[Tuple[str], np.array]]:
        """
        Compute the feature collection indexing for node and edge features.

        Parameters:
            node_features (List[Union[str, List[str]]], optional): List of node features. Default is None.
            edge_features (List[Union[str, List[str]]], optional): List of edge features. Default is None.

        Returns:
            Dict[str, Dict[Tuple[str], np.array]]: A dictionary containing feature collection indexing for node and edge features.
        """
        feature_collection_indexing = {}

        def process_features(feature_type, features):
            if features is not None:
                for i in range(len(features)):
                    if isinstance(features[i], str):
                        features[i] = [features[i]]
                feature_indexing = self._get_derived_indexing(feature_index_map=self.feature_indexing[feature_type], features=features)
                feature_collection_indexing[feature_type] = feature_indexing

        if node_features is not None:
            process_features('node', node_features)
        if edge_features is not None:
            process_features('edge', edge_features)
            
        return feature_collection_indexing
        
    def _set_edge_indexing(self, n_neighbors: int) -> None:
        """
        Sets the edge indexing for the dataset.

        Args:
            n_neighbors (int): The number of neighbors to consider for each node.

        Returns:
            None

        """
        self.edge_indexing = {}
        # IMPLEMENT THE LOGIC HERE

    def _set_binary_adjacency_matrices(self, n_neighbors: int) -> None:
        """
        Sets the binary adjacency matrices for the dataset.

        Args:
            n_neighbors (int): The number of neighbors to consider for each node.

        Returns:
            None

        """
        if self.fully_connected:
            self.adjacency_matrices = adjacency_matrix_map(4, 40, fully_connected = True)
        else:
            self.adjacency_matrices = adjacency_matrix_map(4, 40, n_neighbors = int(n_neighbors / 2))

    def _set_COOrdinate_adjacency_matrices(self, n_neighbors: int) -> None:
        """
        Sets the COOrdinate adjacency matrices for the dataset.

        Args:
            n_neighbors (int): The number of neighbors to consider for each node.

        Returns:
            None

        """
        self.adjacency_matrices = None
    
    def _set_adjacency_matrices(self, n_neighbors: int, format_type: int) -> None:
        """
        Sets the adjacency matrices based on the format type.

        Args:
            n_neighbors (int): The number of neighbors to consider for each node.
            format_type (str): The format of the adjacency matrices.

        Returns:
            None

        Raises:
            Exception: If the format_type is not supported.

        """
        if format_type == 'binary':
            self._set_binary_adjacency_matrices(n_neighbors = n_neighbors)
        elif format_type == 'COOrdinate':
            self._set_COOrdinate_adjacency_matrices(n_neighbors = n_neighbors)
        else:
            raise Exception(f'no _set_{format_type}_adjacency_matrices function')
         
    def _get_train_test_split(self, protein_arr: np.array) -> int:
        """
        Get the index for splitting the protein array into train and test sets.

        Args:
            protein_arr (np.array): Array of protein data.

        Returns:
            int: Index for splitting the protein array.

        Note:
            The method finds the index that separates the training and testing data.
            The train_size attribute is used to determine the percentage of data to use for training.
            The method ensures that the split occurs at a protein boundary, where the protein changes.

        """
        idx = int(protein_arr.shape[0] * self.train_size)
        while True:
            if protein_arr[idx-1] != protein_arr[idx]:
                break
            idx += 1
        return idx

    def _get_train_test_tagging(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'tag' column to the data indicating train or test split.

        Args:
            data (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with 'tag' column.
        """
        data['tag'] = 'train'
        if self.train_size < 1:
            split_index = self._get_train_test_split(protein_arr = data['protein_id'].values)
            data.loc[split_index:, 'tag'] = 'test'
        else:
            split_index = data.shape[0]

        self.split_index = split_index
        self.test_size = (data['tag'] == 'test').sum()
        self.train_size = (data['tag'] == 'train').sum()
        print(f'presampling - train size: {self.train_size}, test size: {self.test_size}, split proportion: {self.train_size}')
        return data
    
    def _get_train_sample_size(self, n_train_samples: int) -> int:
        """
        Calculate the sample size for training observations.

        Args:
            n_train_samples (int): The total number of training samples.

        Returns:
            int: The sample size for training observations.

        """
        size = int(n_train_samples * self.sample_size)
        return size

    def _get_sampled_train_observations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get sampled train observations based on specified conditions.

        Args:
            data (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with added 'sampled' and 'weight' columns.
        """
        data['sampled'] = True
        if self.sampling:
            data.loc[data['tag'] == 'train', 'sampled'] = False
            data['weight'] = 1 / (data['rmsd']**2)
            x = data.loc[data['tag'] == 'train', 'index'].values
            w = data.loc[data['tag'] == 'train', 'weight'].values
            p = w / np.sum(w)
            n_train_samples = (data['tag'] == 'train').sum()
            size = self._get_train_sample_size(n_train_samples = n_train_samples)
            assert size < n_train_samples, f'cannot sample {size} from {n_train_samples} training samples'
            train_samples_idx = np.random.choice(x, size = size, p = p, replace = False)
            data.loc[train_samples_idx, 'sampled'] = True
        return data

    def _get_transformed_graphs(self, graphs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply transformations to the graphs.

        Args:
            graphs (np.ndarray): Input array of graphs.

        Returns:
            np.ndarray: Transformed array of graphs.
        """
        
        if self.scaling:
            y = np.array([g.y for g in graphs]).reshape(-1, 1).astype('float32')
            y_train = y[:-self.test_size]
            
            if self.scaling == 'min_max':
                self.scaler = MinMaxScaler()
            elif self.scaling == 'robust':
                self.scaler = RobustScaler()
            elif self.scaling == 'LRMSD':
                self.scaler = LRMSDScaler()
            else:
                raise Exception(f'unimplemented scaler: {self.scaling}')
            
            self.scaler.fit(y_train)
            y_transformed = self.scaler.transform(y)
            
            for i, g in enumerate(graphs):
                g.y = y_transformed[i,].item()
        else:
            y_transformed = np.array([g.y for g in graphs]).reshape(-1, 1).astype('float32')
        
        if self.labelling == 'discrete':
            # currently only implements one labelling method
            # similar logic as above can be used to employ different labelling schemes
            y = np.array([g.y for g in graphs]).reshape(-1, 1).astype('float32')
            y_train = y[:-self.test_size]

            kbd = KBinsDiscretizer(n_bins = self.bin_params['n_bins'], encode = 'ordinal', strategy = 'quantile')
            kbd.fit(X = y_train)
            y_transformed = kbd.transform(y)

            encoder = OneHotEncoder()
            y_encoded = encoder.fit_transform(X = y_transformed).toarray()

            for i, g in enumerate(graphs):
                g.y = y_encoded[i]
            
            self.bin_edges = kbd.bin_edges_[0,]
            self.bin_midpoints = (self.bin_edges[1:] + self.bin_edges[0:-1]) / 2
            
            def expected_value_transform(x: np.array) -> float:
                return np.dot(a = x, b = self.bin_midpoints)
            self.ev_transform = expected_value_transform

            self.output_dim = self.bin_midpoints.shape[0]
            self.activation = 'softmax'
        else:
            y_transformed = np.array([g.y for g in graphs]).reshape(-1, 1).astype('float32')

        return graphs, y_transformed

    def _get_transformed_tensor(self, T: np.ndarray, methods: Optional[Union[str, List[str]]] = "normalizer", feature_indices: Optional[Union[np.array, List[np.array]]] = None, **kwargs: Dict[str, Any]) -> np.ndarray:

        # transform methods
        if isinstance(methods, str):
            methods = [methods]

        # validate indexes
        if isinstance(methods, list):

            # set method-index correspondence
            if feature_indices is None:
                feature_indices = [list(range(T.shape[2]))] * len(methods)
            else:
                assert len(methods) == len(feature_indices), "number of methods must match the number of feature index lists."

            # assert no overlaps in indexes and all indices are accounted for
            unique_indices = set()
            for indices in feature_indices:
                assert isinstance(indices, (list, np.ndarray)), "feature indices must be provided as a list or NumPy array."
                assert len(set(indices)) == len(indices), "feature indices must not contain duplicates."
                unique_indices.update(indices)
            assert len(unique_indices) == T.shape[2], "all features must be covered by the provided indices."

            '''
            t0 = time.time()
            # for each method-index pair, set transformed values
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:

                # Function to apply transformation for each feature index
                def apply_transform(idx, method):
                    transformer = self.transforms[method].copy()  # Create a copy of the transformer
                    T[:, :, idx] = transformer.fit_transform(T[:, :, idx])

                # Execute the transformations in parallel using ThreadPoolExecutor
                futures = [executor.submit(apply_transform, idx, method) for idx in unique_indices for method in methods]

                # Wait for all transformations to complete
                concurrent.futures.wait(futures)

            t1 = time.time(); print(t1-t0)
            ''' 
            # for each method-index pair, set transformed values
            for method, indices in zip(methods, feature_indices):
                assert method in self.transforms, f"invalid method '{method}' not implemented in Decoys._get_transformed_tensor."
                transformer = self.transforms[method]
                
                for idx in indices:
                    T[:, :, idx] = transformer.fit_transform(T[:, :, idx])

            #t2 = time.time(); print(t2-t1)

        else:
            raise ValueError("methods must be provided as a string or a list of strings.")

        return T

    def _add_node_level_rmsd_regression(self, graphs: List[Graph], indexes: Optional[Union[List[np.array], np.array]] = None, **kwargs: Dict[str, Any]) -> List[Graph]:
        import pdb; pdb.set_trace()
        indexes = None
        if indexes is None:
            indexes = np.arange(0, graphs[0].x.shape[1])

        # If a single index array is provided, convert it to a list for consistency
        if not isinstance(indexes, list):
            indexes = [indexes]

        n_indexes = len(indexes)
        for i, index_group in enumerate(indexes):
            print(f'completed adding node-level regression on {i}/{n_indexes} index groups')

            # Combine node features from all graphs for the current index group into a single matrix X
            X_train = np.vstack([g.x[:, index_group] for g in graphs[0:self.split_index]])
            y_train = np.vstack([g.w for g in graphs[0:self.split_index]])
            X_test = np.vstack([g.x[:, index_group] for g in graphs[self.split_index:]])
            y_test = np.vstack([g.w for g in graphs[self.split_index:]])
            
            # Perform KMeans clustering on the node features
            model = Pipeline(steps = [('scaler', StandardScaler(), 'regressor', XGBRegressor())])
            model.fit(X_train, y_train)
            y = model.predict(X)

            # Unpack one-hot encodings into graph node features for the current index group
            idx0 = 0
            for g in graphs:
                x = g.x
                n_nodes = x.shape[0]
                idx1 = idx0 + n_nodes
                g.x = np.hstack((x, y[idx0:idx1, :].toarray()))
                idx0 = idx1

        return graphs

    def _add_node_level_clustering(self, graphs: List[Graph], indexes: Optional[Union[List[np.array], np.array]] = None, **kwargs: Dict[str, Any]) -> List[Graph]:
        """
        Add node-level clustering information to the given graphs.

        Parameters:
            graphs (List[Graph]): List of Graph objects to which node-level clustering will be added.
            indexes (Union[List[np.array], np.array]): A list of index arrays or a single index array
                indicating which node features to use for clustering. If a list is provided, each index array
                corresponds to a separate group of node features for clustering.
            **kwargs (Dict[str,Any]): Additional keyword arguments.
                
        Returns:
            List[Graph]: List of Graph objects with node-level clustering information added as additional node features.
        """
        
        # If no indexes passed we use all node features
        if indexes is None:
            indexes = np.arange(0, graphs[0].x.shape[1])

        # If a single index array is provided, convert it to a list for consistency
        if not isinstance(indexes, list):
            indexes = [indexes]

        n_clusters = kwargs.get('n_clusters', 8)  # Number of clusters for KMeans
        n_indexes = len(indexes)
        for i, index_group in enumerate(indexes):
            print(f'completed clustering on {i}/{n_indexes} index groups')
            # Combine node features from all graphs for the current index group into a single matrix X
            X = np.vstack([g.x[:, index_group] for g in graphs])

            # Perform KMeans clustering on the node features
            model = KMeans(n_clusters = n_clusters, random_state = 0)
            model.fit(X)
            labels = model.labels_.reshape(-1, 1)

            # One-hot encode the clustering labels
            enc = OneHotEncoder()
            y = enc.fit_transform(labels)

            # Unpack one-hot encodings into graph node features for the current index group
            idx0 = 0
            for g in graphs:
                x = g.x
                n_nodes = x.shape[0]
                idx1 = idx0 + n_nodes
                g.x = np.hstack((x, y[idx0:idx1, :].toarray()))
                idx0 = idx1

        return graphs


    def _get_node_level_rmsd_regression_values(self, X: np.ndarray, **kwargs: Dict[str,Any]) -> np.ndarray:
        X_shape = X.shape
        X = X.reshape(-1, X.shape[2])
        DM = xgb.DMatrix(data = X)
        y = self.model.predict(DM).reshape(-1,1)
        X = np.append(X,y,axis = 1)
        X = X.reshape((X_shape[0], X_shape[1], X_shape[2] + 1))
        return X
    

    def _get_energy_evaluations(self, X: np.ndarray, K: np.ndarray, **kwargs: Dict[str,Any]) -> np.ndarray:
        K_min = K.min(); K_max = K.max()
        K_min_max_transformed = (K - K_min) / (K_max  - K_min)
        K = np.append(K, K_min_max_transformed, axis = 1)
        K = np.repeat(K, X.shape[1], axis = 0)
        K = K.reshape((X.shape[0], X.shape[1], 2))
        X = np.append(X, K, axis = 2)
        return X




    def _process_single_read_job(self, job_args: Dict[str, Any]) -> Tuple[List[Graph], List[Dict[str, Any]]]:
        npz_file = job_args['npz_file']; index = job_args['index']
        decoy_id = npz_file.strip('.npz'); protein_id = decoy_id.split('_')[0]

        graphs = []; data = []
        
        try:
            npz_path = f'{self.graph_path}/{npz_file}'
            np_data = np.load(npz_path)
        except:
            print(f'cannot read: {npz_file}')
            return
        
        # RMSD & energy evaluation
        I = np_data['I']
        index = np.arange(0, I.shape[0]) if index is None else index
        I = I[index,]
        
        # node features
        N = np_data['N']
        N = N[:,:,self.col_index] if hasattr(self, 'col_index') and isinstance(self.col_index, np.ndarray) else N
        N_min = N.min(axis = 0)
        N_max = N.max(axis = 0)
        norm_vals = N_max - N_min
        norm_vals[norm_vals == 0] = 1
        N = (N - N_min) / norm_vals
        
        
        N = self._get_energy_evaluations(X = N, K = I[index,1].reshape(-1,1)) if self.energy_evals else N
        N = self._get_node_level_rmsd_regression_values(X = N) if self.node_level_rmsd_regression else N

        

        
        if self.transform_node_features:
            N = self._get_transformed_tensor(T = N, methods = self.node_transform_methods, feature_indices = self.feature_indexing['node'])
        
        N = N[index,]
        
        # adjacency matrix information
        n_nodes = N.shape[1]
        A = self.adjacency_matrices[n_nodes]
        
        # edge feature matrix
        if self.edge_features:
            # e_idx = self.edge_indexing[n_nodes]
            E = np_data['E'][index,]
            if self.transform_edge_features:
                E = self._get_transformed_tensor(T = E, methods = self.edge_transform_methods, feature_indices = self.edge_feature_indices)
            E = E[index,]

        # define graph collection
        for idx in range(0, N.shape[0]):
            # Set the target value y based on labelling type
            y = I[idx, 2] if self.labelling else rmsd_to_y_onehot(I[idx, 2])

            graph = GraphWrapper(
                    x = N[idx,].astype('float32'),
                    a = A.astype('float32'),
                    y = y.astype('float32'),
                    w = I[idx,5:].astype('float32').reshape(-1,1)
                )
            
            if self.edge_features:
                graph.e = E[idx,].astype('float32')
        
            # Create Graph object and append to graphs list
            graphs.append(graph)
            
            # Create a dictionary for decoy information and append to data list
            start_residue = int(I[idx, 3])
            end_residue = int(I[idx, 4])
            data.append(
                {
                    'decoy_collection_id': f'{protein_id}_{start_residue}_{end_residue}',
                    'protein_id': protein_id,
                    'decoy_id': int(I[idx, 0]),
                    'energy_evaluation': I[idx, 1],
                    'rmsd': I[idx, 2],
                    'start_residue': start_residue,
                    'end_residue': end_residue
                }
            )
        return graphs, data

    def _read(self, n_processes: Optional[int] = 1, debug: Optional[bool] = False) -> List[Graph]:
        """
        Read and process the decoy data.

        Returns:
            List[Graph]: List of processed graphs.
        """

        if self.n_processes is not None:
            n_processes = self.n_processes
        
        decoy_filenames = self._get_decoy_filenames()
        jobs = [{'npz_file': npz_file, 'index': None} for npz_file in decoy_filenames]
        n_jobs = len(jobs)
        print(f'processing {n_jobs} read jobs with {n_processes} processes')
        
        t0 = time.time()
        if debug or n_processes == 1:
            results = []
            for job in tqdm(jobs, desc = f'{n_jobs} jobs processing sequentially'):
                results.append(self._process_single_read_job(job))
        else:        
            with Pool(processes = n_processes) as pool:
                results = list(pool.imap(self._process_single_read_job, jobs))
        t1 = time.time(); dur = (t1 - t0) / 60; print(f'completed reading graphs in {dur:.4}m')
        

        master_graphs = []; master_data = []
        for graphs, data in results:
            master_graphs.extend(graphs)
            master_data.extend(data)
        del results
        
        
        # set graph array and dataframe tracking decoys, ordering is equal here
        graphs = np.array(master_graphs)
        data = pd.DataFrame(master_data)
        del master_graphs; del master_data
        
        # save original index
        data['index'] = data.index

        if not self.test_set:

            # Perform train-test tagging, adds column "tag" one of "train" or "test"
            data = self._get_train_test_tagging(data = data)
            
            # Get sampled train observations
            data = self._get_sampled_train_observations(data = data)
            
            # extract only sampled graphs using the same index
            sidx = data['sampled']
            data = data[sidx].reset_index().drop(columns = ['level_0'])
            graphs = graphs[sidx]

            # compute new split index as n_graphs - n_test_graphs
            self.split_index = len(graphs) - self.test_size
        else:
            data['tag'] = 'test'
            self.split_index = len(graphs)
        
        # Apply transformations to the graphs, does not effect ordering
        graphs, y_transformed = self._get_transformed_graphs(graphs = graphs)
        if y_transformed.shape[1] == 1:
            data['transformed_labels'] = y_transformed
        
        # define load path and save to it
        n_obs = data.shape[0]
        self.decoy_path = f'{self.graph_path}/decoys_n{n_obs}.csv'
        print(f'writing decoys to: {self.decoy_path}')
        data.to_csv(self.decoy_path, index = False)
        print(f'completed Decoys._read(), wrote decoys to {self.decoy_path}')
        return graphs
    
    def _read_saved_indexing(self, n_processes: Optional[int] = 8, debug: Optional[bool] = False) -> List[Graph]:
        """
        Read and process the decoy data from a pre-saved index.

        Returns:
            List[Graph]: List of processed graphs.
        """
        
        if self.n_processes is not None:
            n_processes = self.n_processes

        assert os.path.exists(self.graph_indexing_path)
        jobs, saved_data = self._get_load_jobs(data = pd.read_csv(self.graph_indexing_path))
        n_jobs = len(jobs)


        print(f'processing {n_jobs} read jobs')
        t0 = time.time()
        if debug or n_processes == 1:
            results = []
            for job in jobs:
                results.append(self._process_single_read_job(job))
        else:        
            with Pool(processes = n_processes) as pool:
                results = list(pool.imap(self._process_single_read_job, jobs))
        dur = time.time() - t0; print(f'completed processing {n_jobs} jobs in {dur:.4}s')

        # aggregating data
        master_graphs = []; master_data = []
        for graphs, data in results:
            master_graphs.extend(graphs)
            master_data.extend(data)
        
        # set graph array and dataframe tracking decoys
        graphs = np.array(master_graphs)
        data = pd.DataFrame(master_data)
        data['index'] = data.index
        data['tag'] = saved_data['tag']
        data['sampled'] = saved_data['sampled']
        self.test_size = (data['tag'] == 'test').sum()
        
        # apply transformations to the graphs
        graphs, y_transformed = self._get_transformed_graphs(graphs = graphs)
        if y_transformed.shape[1] == 1:
            data['transformed_labels'] = y_transformed
        
        self.split_index = len(graphs) - self.test_size
        self.decoy_path = self.graph_indexing_path

        return graphs
    
    

    def read(self) -> List[Graph]:
        """
        Read and process the decoy data.

        Returns:
            List[Graph]: List of processed graphs.
        """
        if self.read_saved_indexing:
            print('calling Decoys._read_saved_indexing')
            return self._read_saved_indexing()
        else:
            print('calling Decoys._read()')
            return self._read()
        
