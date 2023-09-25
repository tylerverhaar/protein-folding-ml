import os
import re
import time
import itertools
import os, gc
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Union, Tuple, Generator


class NodeLevelModel:

    def __init__(self, graph_path: str, **kwargs: Dict[str, Any]) -> None:

        # graph loading
        assert os.path.exists(graph_path)
        self.graph_path = graph_path
        self.name = kwargs.get('name', graph_path.split('/')[-2])
        self.data_dir = '/'.join(graph_path.split('/')[:-1])
        
        self.read_saved_indexing = kwargs.get('read_saved_indexing', False)
        self.graph_indexing_path = kwargs.get('graph_indexing_path', f'{self.data_dir}/new_decoys.csv')

        self.downsampling = kwargs.get('downsampling', False)
        self.sampling_size = kwargs.get('sampling_size', 0.8)

        self.labelling = kwargs.get('labelling', 'continuous')
        self.n_bins = kwargs.get('n_bins', 8)

        self.energy_evals = kwargs.get('energy_evals', False)

        self._set_feature_indexing()
        self._set_model_directory()

    def _set_feature_indexing(self, **kwargs: Dict[str,Any]) -> None:
        """
        Set up feature indexing based on feature information in the CSV file.

        Parameters:
            **kwargs (Dict[str, Any]): Additional keyword arguments (not used in this method).

        Returns:
            None
        """
        data = pd.read_csv(f'{self.graph_path}/feature_shapes.csv')
        data = data[data['tag'] == 'node']
        data['end_index'] = np.cumsum(data['feature_dim'])
        data['start_index'] = data['end_index'] - data['feature_dim']
        self.n_features = data.iloc[-1]['end_index']
        self.feature_indexing = data.set_index("feature_name").apply(lambda x: np.arange(x["start_index"], x["end_index"]), axis=1).to_dict()
        self.index_feature_map = {v: key for key, value in self.feature_indexing.items() for v in value}
        
    def _set_model_directory(self) -> None:
        """
        Set the model directory for storing XGBoost models.

        This method constructs the directory path for storing XGBoost models within the
        specified data directory. If the directory does not already exist, it will be created.

        Attributes:
            self.data_dir (str): The base directory where data and models are stored.

        Sets:
            self.model_dir (str): The full path to the directory for storing XGBoost models.

        Usage:
            Call this method within a class instance to set up the model directory.
            The method ensures that the directory exists or creates it if needed.

        Returns:
            None
        """
        self.model_dir = f'{self.data_dir}/xgb_models'
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def _set_feature_groups(self, feature_patterns: List[Tuple[str, str]], feature_unions: Optional[List[List[str]]] = None) -> None:
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
        for feature in self.feature_indexing:
            for tag, tag_pattern in feature_patterns:  # Renamed pattern to tag_pattern
                if re.match(tag_pattern, feature):
                    groups[tag].append(self.feature_indexing[feature])
        self.feature_groups = {tag: np.sort(np.concatenate(indexes)) for tag, indexes in groups.items()}

        if feature_unions is None or (isinstance(feature_unions, list) and len(feature_unions) == 0):
            return

        for union_tags in feature_unions:  # Renamed feature_union to union_tags
            self.feature_groups['__'.join(union_tags)] = np.sort(np.unique(np.concatenate([self.feature_groups[tag] for tag in union_tags])))

    def _set_discrete_response_transforms(self, y: np.ndarray, **kwargs: Dict[str,Any]) -> None:
        self.kbd = KBinsDiscretizer(n_bins = self.n_bins, encode = 'ordinal', strategy = 'quantile')
        self.kbd.fit(y)
        self.bin_edges = self.kbd.bin_edges_[0,]
        self.bin_midpoints = (self.bin_edges[1:] + self.bin_edges[0:-1]) / 2
        self.encoder = OneHotEncoder()
        self.encoder.fit(y)
        def expected_value_transform(x: np.array) -> float:
            return np.dot(a = x, b = self.bin_midpoints)
        self.ev_transform = expected_value_transform

    def _set_continuous_response_transforms(self, y: np.ndarray, **kwargs: Dict[str,Any]) -> None:
        return
    
    def _get_response_vector(self, jobs: List[Dict[str,Any]], **kwargs: Dict[str,Any]) -> np.ndarray:
        Y = []
        for i in tqdm(range(len(jobs)), desc = "loading y's for train transforms"):
            fname = jobs[i]['filepath']
            np_load_path = f'{self.graph_path}/{fname}'
            data = np.load(np_load_path)
            index = jobs[i]['index']
            y = data['I'][index,5:].reshape(-1,1)
            Y.append(y)
        y = np.vstack(Y)
        return y

    def _set_train_transforms(self, jobs: List[Dict[str,Any]], **kwargs: Dict[str,Any]) -> None:
        y = self._get_response_vector(jobs = jobs, **kwargs)
        
        if self.labelling == 'continuous':
            y = None
            self._set_continuous_response_transforms(y = y, **kwargs)
        
        if self.labelling == 'discrete':
            self._set_discrete_response_transforms(y = y, **kwargs)

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

    def _get_downsampled_train_observations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get sampled train observations based on specified conditions.

        Args:
            data (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with added 'sampled' and 'weight' columns.
        """
        data['sampled'] = True
        if self.downsampling:
            data.loc[data['tag'] == 'train', 'sampled'] = False
            data['weight'] = 1 / (data['rmsd']) ** 2
            x = data.loc[data['tag'] == 'train', 'index'].values
            w = data.loc[data['tag'] == 'train', 'weight'].values
            p = w / np.sum(w)
            n_train_samples = (data['tag'] == 'train').sum()
            size = int(self.sampling_size * n_train_samples)
            assert size < n_train_samples, f'cannot sample {size} from {n_train_samples} training samples'
            train_samples_idx = np.random.choice(x, size = size, p = p, replace = False)
            data.loc[train_samples_idx, 'sampled'] = True
        return data
    
    def _get_jobs(self, data: pd.DataFrame, **kwargs: Dict[str, Any]) -> List[Dict[str,Any]]:
        n_index_splits = kwargs.get('n_index_splits', 0)
        shuffle = kwargs.get('shuffle', False)
        n_train_batches = kwargs.get('n_train_batches', -1)

        data = data[data['sampled']].copy()
        data['start_residue'] = data['start_residue'].astype(int)
        data['end_residue'] = data['end_residue'].astype(int)
        data['decoy_loop_ids'] = data.apply(lambda row: '_'.join([str(row['protein_id']), str(row['start_residue']), str(row['end_residue'])]), axis=1)
        
        train_jobs = []; test_jobs = []
        n_decoys = data['decoy_loop_ids'].unique().shape[0]
        for decoy_loop_id in tqdm(data['decoy_loop_ids'].unique(), desc = f'loading {n_decoys} graphs'):
            loc = data[data['decoy_loop_ids'] == decoy_loop_id]
            tag = loc['tag'].values[0]
            index = loc['decoy_id'].values - 1

            if tag == 'test':
                test_jobs.append({'filepath': f'{decoy_loop_id}.npz', 'index': index, 'tag': tag})
                continue

            indexes = np.array_split(index, n_index_splits) if n_index_splits > 1 and n_train_batches > 1 else [index]
            for index in indexes:
                train_jobs.append({'filepath': f'{decoy_loop_id}.npz', 'index': index, 'tag': tag})

        if shuffle:
            np.random.shuffle(train_jobs)

        return train_jobs, test_jobs

    def _load_file(self, filepath, index, col_index, **kwargs: Dict[str,Any]):
        data = np.load(filepath)
        X = data['N'][index,]
        y = data['I'][index,5:]
        
        E = data['I'][index,1]
        E = np.repeat(E, X.shape[1], axis = 0).reshape(-1,1)
        X = X.reshape(-1, X.shape[2])
        
        if col_index is not None:
            X = X[:,col_index]
            self.n_features = X.shape[1]
        
        if self.energy_evals:
            self.n_features = X.shape[1]
            self.feature_indexing['energy_evaluation'] = np.array([self.n_features])
            self.feature_indexing['energy_evaluation_MMS'] = np.array([self.n_features + 1])
            self.index_feature_map[self.n_features] = 'energy_evaluation'
            self.index_feature_map[self.n_features+1] = 'energy_evaluation_MMS'
            E_min = E.min(); E_max = E.max()
            E_scaled = (E - E_min) / (E_max - E_min)
            X = np.concatenate([X, E, E_scaled], axis = 1)
        
        y = y.reshape(-1,1)
        return X, y

    def _get_Xy_generator_multi_threaded(self, jobs: List[Dict[str, Any]], col_index: Optional[np.array] = None, n_threads: Optional[int] = 4) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate X, y pairs from a list of jobs using multiple threads.

        Parameters:
            jobs (List[Dict[str, Any]]): List of jobs, each containing 'filepath' and 'index'.
            col_index (Optional[np.array]): Optional column index.
            n_threads (Optional[int]): Number of threads to use.

        Yields:
            Tuple[np.ndarray, np.ndarray]: A tuple containing X and y data.

        Usage:
            Use this method to generate X, y pairs from the provided list of jobs.
            If n_threads is greater than 1, data loading will be multi-threaded.
        """
        with ThreadPoolExecutor(max_workers = n_threads) as executor:
            futures = [executor.submit(self._load_file, os.path.join(self.graph_path, job['filepath']), job['index'], col_index) for job in jobs]
            for future in futures:
                X, y = future.result()
                yield X, y

    def _get_Xy_generator_single_thread(self, jobs: List[Dict[str, Any]], col_index: Optional[np.array] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate X, y pairs from a list of jobs using a single thread.

        Parameters:
            jobs (List[Dict[str, Any]]): List of jobs, each containing 'filepath', 'index', and 'col_index'.
            col_index (Optional[np.array]): Optional column index.

        Yields:
            Tuple[np.ndarray, np.ndarray]: A tuple containing X and y data.

        Usage:
            Use this method to generate X, y pairs from the provided list of jobs using a single thread.
        """
        for job in jobs:
            X, y = self._load_file(filepath = os.path.join(self.graph_path, job['filepath']), index = job['index'], col_index = job['index'])
            yield X, y
        
    def _get_Xy_generator(self, jobs: List[Dict[str, Any]], col_index: Optional[np.array] = None, n_threads: Optional[int] = 4) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate X, y pairs from a list of jobs.

        Parameters:
            jobs (List[Dict[str, Any]]): List of jobs, each containing 'filepath', 'index', and 'col_index'.
            col_index (Optional[np.array]): Optional column index.
            n_threads (Optional[int]): Number of threads to use.

        Yields:
            Tuple[Any, Any]: A tuple containing X and y data.

        Usage:
            Use this method to generate X, y pairs from the provided list of jobs.
            The number of threads used for data loading is determined by n_threads.
        """
        if n_threads > 1:
            return self._get_Xy_generator_multi_threaded(jobs = jobs, col_index = col_index, n_threads = n_threads)
        else:
            return self._get_Xy_generator_single_thread(jobs = jobs, col_index = col_index)

    def _get_xgb_DMatrix(self, jobs: List[Dict[str, Any]], col_index: Optional[np.array] = None, n_threads: Optional[int] = 4, **kwargs: Dict[str,Any]) -> xgb.DMatrix:
        """
        Generate an XGBoost DMatrix for training or prediction.

        This method generates an XGBoost DMatrix using the data generated from a list of jobs.
        It leverages the `_get_Xy_generator` method to obtain X and y data pairs.

        Parameters:
            jobs (List[Dict[str, Any]]): List of jobs, each containing 'filepath', 'index', and 'col_index'.
            col_index (Optional[np.array]): Optional column index.
            n_threads (int): Number of threads to use when loading (X,y) from generators. Default is 4.

        Returns:
            xgb.DMatrix: An XGBoost DMatrix containing the training data and labels.

        Usage:
            Call this method to obtain an XGBoost DMatrix for training or prediction.
            The generated DMatrix contains the feature matrix and labels derived from the provided jobs.
            If the labels are discrete, they are transformed using the provided label encoder.
        """
        Xs = []; ys = []
        for X,y in self._get_Xy_generator(jobs = jobs, col_index = col_index, n_threads = n_threads):
            Xs.append(X); ys.append(y)
        X = np.vstack(Xs); y = np.vstack(ys)
        if self.labelling == 'discrete':
            y = self.kbd.transform(y)
        
        X_cols = kwargs.get('X_col', None)
        if X_cols:
            assert X.shape[0] == X_cols.shape[0], AssertionError('misaligned shapes, X, X_col')
            X = np.append(X, X_cols, axis = 1)
        
        DM = xgb.DMatrix(X, label = y)
        return DM

    def _get_regressor_eval(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Dict[str, Any]) -> Dict[str, Union[np.float32, np.float64]]:
        """
        Evaluate the regression model's performance using various metrics.

        Parameters:
            y_true (np.ndarray): The true target variable.
            y_pred (np.ndarray): The predicted target variable.
            **kwargs (Dict[str, Any]): Additional keyword arguments.
                prefix (str, optional): A prefix to be added to the metric names in the returned dictionary.

        Returns:
            Dict[str, np.float]: A dictionary containing regression evaluation metrics.
        """
        assert y_true.shape == y_pred.shape

        if hasattr(self, 'scaler'):
            y_true = self.scaler.inverse_transform(y_true)
            y_pred = self.scaler.inverse_transform(y_pred)

        metrics = {
            "mae": mean_absolute_error(y_true, y_pred), 
            "mse": mean_squared_error(y_true, y_pred), 
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)), 
            "r2": r2_score(y_true, y_pred)
            }

        return metrics

    def _get_classifier_eval(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, **kwargs: Dict[str, Any]) -> Dict[str, Union[np.float32, np.float64]]:
        """
        Evaluate a classifier model's performance using various metrics.

        Parameters:
            y_true (np.ndarray): The true labels of the data (1-dimensional array).
            y_pred (np.ndarray): The predicted labels of the data (1-dimensional array).
            y_prob (np.ndarray): Predicted probabilities for positive class (1-dimensional array).

        Returns:
            Dict[str, np.float]: A dictionary containing classification evaluation metrics.
        """
        assert y_true.shape == y_pred.shape and y_true.shape == y_prob.shape

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred), 
            "precision(mirco)": precision_score(y_true, y_pred, average = 'micro'), 
            "recall(micro)": recall_score(y_true, y_pred, average = 'micro'), 
            "f1_score(micro)": f1_score(y_true, y_pred, average = 'micro'), 
            "log_loss": log_loss(y_true, y_prob)
            }
        
        return metrics
    
    def _get_model_eval(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, **kwargs: Dict[str,Any]) -> None:
        """
        Compute model evaluation metrics based on predicted and true labels.

        This method calculates evaluation metrics based on predicted and true labels,
        adapting to both classification and regression tasks.

        Parameters:
            y_true (np.ndarray): True labels as a numpy array.
            y_pred (np.ndarray): Predicted labels as a numpy array.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics and additional arguments.

        Usage:
            Call this method to obtain model evaluation metrics for predicted and true labels.
            The method adapts to both classification and regression tasks, depending on the label type.
            For classification tasks, provide 'y_prob' along with 'y_true' and 'y_pred' when output_type is 'clf'.
            Additional keyword arguments can be provided to customize the evaluation process.
            Metrics are returned in a dictionary, with an optional prefix to label the metrics.
        """
        assert y_true.shape == y_pred.shape, AssertionError('NodeLevelModelling._get_model_eval requires y_true.shape == y_pred.shape')
        
        if self.labelling == 'discrete':
            #y_prob = kwargs.get('y_prob', None)
            assert y_prob is not None, AssertionError('NodeLevelModelling._get_model_eval requires arguement "y_prob" when output_type = "clf"')
            eval =  self._get_classifier_eval(y_true = y_true, y_pred = y_pred, y_prob = y_prob, **kwargs)
        else:
            eval = self._get_regressor_eval(y_true = y_true, y_pred = y_pred, **kwargs)

        for key, value in kwargs.items():
            if key != 'prefix' and key != 'y_prob':
                eval[key] = value

        prefix = kwargs.get('prefix', None)
        if prefix is not None:
            eval = {f"{prefix}_{key}": value for key, value in eval.items()}
        
        return eval
        
    def _get_y_classifier(self, DM: xgb.DMatrix, model: xgb.Booster) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predicted labels, true labels, and predicted probabilities for classification tasks.

        This method generates predicted labels, true labels, and predicted probabilities
        using an XGBoost DMatrix and a trained classification model.

        Parameters:
            DM (xgb.DMatrix): XGBoost DMatrix containing feature data.
            model (xgb.Booster): Trained XGBoost classification model.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing y_pred (predicted labels),
            y_true (true labels), and y_prob (predicted probabilities).

        Usage:
            Call this method to obtain predicted labels, true labels, and predicted probabilities
            for a classification model. The method adapts to the 'discrete' labelling mode.
        """
        y_prob = model.predict(DM)
        y_pred = np.zeros(shape = y_prob.shape)
        y_pred[np.arange(y_pred.shape[0]), y_prob.argmax(axis=1)] = 1
        y_true = DM.get_label().reshape(-1,1).astype('int32')
        y_true = OneHotEncoder().fit_transform(y_true).toarray()
        return y_pred, y_true, y_prob
    
    def _get_y_regressor(self, DM: xgb.DMatrix, model: xgb.Booster) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predicted labels and true labels for regression tasks.

        This method generates predicted labels and true labels using an XGBoost DMatrix
        and a trained regression model.

        Parameters:
            DM (xgb.DMatrix): XGBoost DMatrix containing feature data.
            model (xgb.Booster): Trained XGBoost regression model.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing y_pred (predicted labels) and y_true (true labels).

        Usage:
            Call this method to obtain predicted labels and true labels for a regression model.
            The method is suitable for regression tasks.
        """
        y_pred = model.predict(DM).reshape(-1,1)
        y_true = DM.get_label().reshape(-1,1)
        return y_pred, y_true

    def _get_y(self, DM: xgb.DMatrix, model: xgb.Booster) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate predicted labels, true labels, and predicted probabilities for classification tasks or
        predicted labels and true labels for regression tasks.

        This method generates predicted labels, true labels, and predicted probabilities (for classification)
        or predicted labels and true labels (for regression) using an XGBoost DMatrix and a trained model.

        Parameters:
            DM (xgb.DMatrix): XGBoost DMatrix containing feature data.
            model (xgb.Booster): Trained XGBoost model.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            A tuple containing y_pred (predicted labels), y_true (true labels), and y_prob (predicted probabilities)
            for classification tasks. For regression tasks, the tuple contains y_pred and y_true only.

        Usage:
            Call this method to obtain predicted labels, true labels, and predicted probabilities (classification)
            or predicted labels and true labels (regression) for a trained XGBoost model.
        """
        if self.labelling == 'discrete':
            y_pred, y_true, y_prob = self._get_y_classifier(DM = DM, model = model)
        else:
            y_pred, y_true =  self._get_y_regressor(DM = DM, model = model)
            y_prob = None
        return y_pred, y_true, y_prob

    def _get_graph_rmsds(self, model: xgb.Booster, graph_jobs: List[Dict[str,Any]], col_index: Optional[np.array] = None) -> pd.DataFrame:
        """
        Calculate RMSD values for graph-based data and model predictions.

        This method calculates root mean square deviation (RMSD) values for graph-based data
        using the provided XGBoost model and graph jobs.

        Parameters:
            model (xgb.Booster): Trained XGBoost model.
            graph_jobs (List[Dict[str, Any]]): List of graph jobs.
            col_index (Optional[np.array]): Optional column index to select specific features.

        Returns:
            pd.DataFrame: A DataFrame containing calculated model RMSD and target RMSD values.

        Usage:
            Call this method to calculate RMSD values for graph-based data and model predictions.
            The method processes each graph job and calculates RMSD between predicted and target values.
            If col_index is provided, specific features can be selected for analysis.
        """
        Y_pred = []; Y_true = []
        
        for job in tqdm(graph_jobs, desc = 'evaluating decoy collections'):
            fname = job['filepath']; np_load_path = f'{self.graph_path}/{fname}'
            data = np.load(np_load_path)
            index = job['index']
            X = data['N'][index,:]
            n_obs, n_nodes, n_features = X.shape
            E = data['I'][index,1]
            E = np.repeat(E, X.shape[1], axis = 0).reshape(-1,1)
            X = X.reshape(-1, n_features)
            
            if col_index is not None:
                X = X[:,col_index]
            
            if self.energy_evals:
                E_min = E.min(); E_max = E.max()
                E_scaled = (E - E_min) / (E_max - E_min)
                X = np.concatenate([X, E, E_scaled], axis = 1)

            DM = xgb.DMatrix(data = X)

            if self.labelling == 'continuous':
                y_pred_NL = model.predict(DM).reshape(-1,1)
            else:
                y_prob_NL = model.predict(DM)
                y_pred_NL = self.ev_transform(x = y_prob_NL)
            
            y_pred_NL = y_pred_NL.reshape(n_obs, n_nodes, 1)
            y_pred = np.sqrt(np.mean(y_pred_NL**2, axis = 1))
            
            Y_pred.append(y_pred)
            Y_true.append(data['I'][index,2].reshape(-1,1))
        
        result = pd.DataFrame(data = {'model_rmsd': np.concatenate(Y_pred).ravel(), 'target_rmsd': np.concatenate(Y_true).ravel()})
        return result
    
    def evaluate_predictions(self, data: pd.DataFrame, **kwargs: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate model predictions and generate ranking-based results.

        This method evaluates model predictions and generates ranking-based results for each decoy collection.
        It computes various performance metrics based on different rankings, including model RMSD ranking and energy ranking.

        Parameters:
            data (pd.DataFrame): DataFrame containing the dataset for evaluation.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames - the first DataFrame
            contains detailed results for each decoy collection, and the second DataFrame contains aggregated
            ranking-based metrics.

        Usage:
            Call this method to evaluate model predictions and generate ranking-based results for each decoy collection.
            The method calculates various performance metrics such as top RMSD, top rank, and more for both model
            predictions and energy rankings. Additional keyword arguments can be included to customize the results.
        """
        results = []
        rmsd_results = []
        dc_dict = dict(tuple(data.groupby('decoy_collection_id')))
        for _, dc in dc_dict.items():
            iloc0 = dict(dc.iloc[0])
            cb_rmsd = dc['rmsd'].min()
            n_decoys = dc.shape[0]
            
            dc = dc.sort_values(by = ['rmsd'])
            dc['real_ranking'] = np.arange(1, n_decoys + 1)

            dc = dc.sort_values(by = ['model_rmsd'])
            dc['model_ranking'] = np.arange(1, n_decoys + 1)
            model_rmsds = np.copy(dc['rmsd'].values)
            model_ranks = np.copy(dc['real_ranking'].values)

            # Ranking w.r.t. energy function
            dc = dc.sort_values(by = ['energy_evaluation'])
            dc['energy_ranking'] = np.arange(1, n_decoys + 1)
            energy_rmsds = np.copy(dc['rmsd'].values)
            energy_ranks = np.copy(dc['real_ranking'].values)

            dc = dc.sort_values(by = ['rmsd'])

            #real_ranking = dc['real_ranking'].values
            #model_ranking = dc['model_ranking'].values
            #energy_ranking = dc['energy_ranking'].values
            
            #model_kendall_tau, _ = kendalltau(real_ranking, model_ranking)
            #energy_kendall_tau, _ = kendalltau(real_ranking, energy_ranking)

            # Calculate Spearman's Rank Correlation
            #model_spearman_corr, _ = spearmanr(real_ranking, model_ranking)
            #energy_spearman_corr, _ = spearmanr(real_ranking, energy_ranking)
            
            results.append(dc)
            rmsd_results.append({
                'protein_id': iloc0['protein_id'],
                'start_residue': int(iloc0['start_residue']),
                'end_residue': int(iloc0['end_residue']),
                'n_decoys': n_decoys,
                'cb_rmsd': cb_rmsd,
                # model performance
                'top_rmsd(model)': model_rmsds[0],
                'top5_rmsd(model)': model_rmsds[0:5].min(),
                'top_rank(model)': model_ranks[0],
                'top5_rank(model)': model_ranks[0:5].min(),
                # energy peformance
                'top_rmsd(energy)': energy_rmsds[0],
                'top5_rmsd(energy)': energy_rmsds[0:5].min(),
                'top_rank(energy)': energy_ranks[0],
                'top5_rank(energy)': energy_ranks[0:5].min(),
                # correlation performance
                #'model_kendall_tau': model_kendall_tau,
                #'energy_kendall_tau': energy_kendall_tau,
                #'model_spearman_corr': model_spearman_corr,
                #'energy_spearman_corr': energy_spearman_corr,
            })

        results = pd.concat(results)
        rmsd_results = pd.DataFrame(rmsd_results)

        for key, value in kwargs.items():
            results[key] = value
            rmsd_results[key] = value

        return results, rmsd_results

    def _get_decoy_analysis(self, model: xgb.Booster, jobs: List[Dict[str,Any]], col_index: np.array, data: pd.DataFrame, tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform decoy analysis and evaluate model predictions.

        This method analyzes decoy results, assigns model RMSD and target RMSD values to the dataset,
        and then evaluates the predictions using the specified data and tag.

        Parameters:
            model (xgb.Booster): Trained XGBoost model.
            jobs (List[Dict[str, Any]]): List of graph jobs.
            col_index (np.array): Column index.
            data (pd.DataFrame): Dataframe containing the dataset.
            tag (str): Tag for the analysis.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing decoy-level results and RMSD results.

        Usage:
            Call this method to perform decoy analysis and evaluate model predictions on a specific tag.
            The method integrates graph RMSD results with the dataset, evaluates predictions, and returns the results.
        """
        graph_results = self._get_graph_rmsds(graph_jobs = jobs, model = model, col_index = col_index)
        data = data[data['tag'] == tag].copy()
        data['model_rmsd'] = graph_results['model_rmsd'].values
        data['target_rmsd'] = graph_results['target_rmsd'].values
        decoy_level_results, rmsd_results = self.evaluate_predictions(data = data, tag = tag)
        return decoy_level_results, rmsd_results

    def train_test(self, n_train_batches: int, n_test_batches: int, model_params: Dict[str,Any], model_name: str, n_threads: Optional[int] = 4, **kwargs: Dict[str,Any]) -> None:
        """
        Train and test XGBoost model, evaluate its performance, and save results.

        Parameters:
            n_train_batches (int): Number of training batches.
            n_test_batches (int): Number of testing batches.
            model_params (Dict[str, Any]): XGBoost model hyperparameters.
            model_name (str): Name for the trained model.
            n_threads (int): Number of threads to use when loading (X,y) from generators. Default is 4.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            None

        Usage:
            Call this method to train and test an XGBoost model, evaluate its performance,
            and save the results.
        """
        data = pd.read_csv(self.graph_indexing_path)
        data = self._get_downsampled_train_observations(data = data)
        
        train_jobs, test_jobs = self._get_jobs(data = data, n_index_splits = 5, shuffle = True)
        train_eval_jobs, test_eval_jobs = self._get_jobs(data = data)
        self._set_train_transforms(jobs = train_jobs)
        
        train_job_batches = np.array_split(train_jobs, n_train_batches); n_train_job_batches = len(train_job_batches)
        test_job_batches = np.array_split(test_jobs, n_test_batches); n_test_job_batches = len(test_job_batches)

        group_patterns = [('nn(dist)_dists', r"^nn\(dist\)_(.*?)_(.*?)_dists$")]
        self._set_feature_groups(feature_patterns = group_patterns)
        col_index = self.feature_groups['nn(dist)_dists']

        num_boost_round = model_params['n_estimators']; del model_params['n_estimators']
        if self.labelling== 'discrete':
            model_params.update({'objective': 'multi:softprob', 'num_class': self.n_bins})
        
        evals = []
        # 1A73.A_44_53

        # model training
        for j, train_jobs in enumerate(train_job_batches):
            print(f'train iteration: {j+1}/{n_train_job_batches}')
            t0 = time.time()
            DM = self._get_xgb_DMatrix(jobs = train_jobs, col_index = col_index, n_threads = n_threads)
            t1 = time.time(); dur = t1 - t0; n_features = DM.num_col(); n_observations = DM.num_row()
            print(f'train iteration: {j+1}/{n_train_job_batches}, n_features = {n_features}, n_observations = {n_observations}, DMatrix load time: {dur:.4}s')
            model = xgb.train(model_params, dtrain = DM, num_boost_round = num_boost_round) if j == 0 else xgb.train(model_params, dtrain = DM, num_boost_round = num_boost_round, xgb_model = model)
            t2 = time.time(); dur = t2 - t1; print(f'train duration: {dur:.4}s')
            y_pred, y_true, y_prob  = self._get_y(DM = DM, model = model)
            del DM
            eval = self._get_model_eval(y_pred = y_pred, y_true = y_true, y_prob = y_prob, tag = 'train', iter = j+1, obs = n_observations)
            evals.append({**model_params, **eval})


        save_model = kwargs.get('save_model', False)
        # save model and reload it, this helps with CPU/GPU-RAM
        if save_model:
            model.save_model(f'{self.model_dir}/{model_name}.model')
            del model
            model = xgb.Booster()
            model.load_model(f'{self.model_dir}/{model_name}.model')

        # model testing
        for j, test_jobs in enumerate(test_job_batches):
            print(f'test iteration: {j+1}/{n_test_job_batches}')
            t0 = time.time()
            DM = self._get_xgb_DMatrix(jobs = test_jobs, col_index = col_index, n_threads = n_threads)
            t1 = time.time(); dur = t1 - t0; n_features = DM.num_col(); n_observations = DM.num_row()
            y_pred, y_true, y_prob  = self._get_y(DM = DM, model = model)
            del DM
            eval = self._get_model_eval(y_pred = y_pred, y_true = y_true, y_prob = y_prob, tag = 'test', iter = j+1, obs = n_observations)
            evals.append({**model_params, **eval})

        # evaluation for train and testing
        
        eval_data = pd.DataFrame(evals)
        eval_data.to_csv(f'{self.model_dir}/{model_name}_train_test_evals.csv')
        print(eval_data)

        decoy_level_results_train, rmsd_results_train = self._get_decoy_analysis(model = model, jobs = train_eval_jobs, col_index = col_index, data = data, tag = 'train')
        decoy_level_results_test, rmsd_results_test = self._get_decoy_analysis(model = model, jobs = test_eval_jobs, col_index = col_index, data = data, tag = 'test')

        decoy_level_results = pd.concat([decoy_level_results_train, decoy_level_results_test])
        rmsd_results = pd.concat([rmsd_results_train, rmsd_results_test])
        
        decoy_level_results.to_csv(f'{self.model_dir}/{model_name}_decoy_level_results.csv', index = False)
        rmsd_results.to_csv(f'{self.model_dir}/{model_name}_rmsd_results.csv', index = False)
        
        rmsd_res_pivot = pd.pivot_table(
            rmsd_results, 
            index = 'tag', 
            values = ['cb_rmsd','top_rmsd(model)', 'top5_rmsd(model)', 'top_rank(model)', 'top5_rank(model)', 'top_rmsd(energy)', 'top5_rmsd(energy)', 'top_rank(energy)', 'top5_rank(energy)'], 
            aggfunc = 'mean'
            )
        print(rmsd_res_pivot)
        
    def test(self, n_train_batches: int, n_test_batches: int, model_params: Dict[str,Any], model_name: str, **kwargs: Dict[str,Any]) -> None:
        data = pd.read_csv(self.graph_indexing_path)
        data = self._get_downsampled_train_observations(data = data)
        
        _, test_eval_jobs = self._get_jobs(data = data)
        test_job_batches = np.array_split(test_eval_jobs, n_test_batches); n_test_job_batches = len(test_job_batches)
        
        group_patterns = [('nn(dist)_dists', r"^nn\(dist\)_(.*?)_(.*?)_dists$")]
        self._set_feature_groups(feature_patterns = group_patterns)
        col_index = self.feature_groups['nn(dist)_dists']

        model = xgb.Booster()
        model.load_model(f'{self.model_dir}/{model_name}.model')
        evals = []
        # model testing
        for j, test_job in enumerate(test_job_batches):
            print(f'test iteration: {j+1}/{n_test_job_batches}')
            t0 = time.time()
            DM = self._get_xgb_DMatrix(jobs = test_job, col_index = col_index)
            t1 = time.time(); dur = t1 - t0; n_features = DM.num_col(); n_observations = DM.num_row()
            y_pred, y_true, y_prob  = self._get_y(DM = DM, model = model)
            del DM
            eval = self._get_model_eval(y_pred = y_pred, y_true = y_true, y_prob = y_prob, tag = 'test', iter = j+1, obs = n_observations)
            evals.append({**model_params, **eval})

        # evaluation for train and testing
        
        eval_data = pd.DataFrame(evals)
        eval_data.to_csv(f'{self.model_dir}/{model_name}_train_test_evals.csv')
        print(eval_data)

        decoy_level_results, rmsd_results = self._get_decoy_analysis(model = model, jobs = test_eval_jobs, col_index = col_index, data = data, tag = 'test')

        decoy_level_results.to_csv(f'{self.model_dir}/{model_name}_decoy_level_results.csv', index = False)
        rmsd_results.to_csv(f'{self.model_dir}/{model_name}_rmsd_results.csv', index = False)
        
        rmsd_res_pivot = pd.pivot_table(
            rmsd_results, 
            index = 'tag', 
            values = ['cb_rmsd','top_rmsd(model)', 'top5_rmsd(model)', 'top_rank(model)', 'top5_rank(model)', 'top_rmsd(energy)', 'top5_rmsd(energy)', 'top_rank(energy)', 'top5_rank(energy)'], 
            aggfunc = 'mean'
            )
        print(rmsd_res_pivot)

    def nested_model(self, 
                     n_train_batches: int, 
                     n_test_batches: int, 
                     model_name: str,
                     base_model: xgb.Booster, 
                     model_params: Dict[str,Any], 
                     n_threads: Optional[int] = 4,
                     **kwargs: Dict[str,Any]) -> None:
        
        data = pd.read_csv(self.graph_indexing_path)
        data = self._get_downsampled_train_observations(data = data)
        
        train_jobs, test_jobs = self._get_jobs(data = data, n_index_splits = 5, shuffle = True)
        train_eval_jobs, test_eval_jobs = self._get_jobs(data = data)
        self._set_train_transforms(jobs = train_jobs)
        
        train_job_batches = np.array_split(train_jobs, n_train_batches); n_train_job_batches = len(train_job_batches)
        test_job_batches = np.array_split(test_jobs, n_test_batches); n_test_job_batches = len(test_job_batches)

        group_patterns = [('nn(dist)_dists', r"^nn\(dist\)_(.*?)_(.*?)_dists$"), ('CM_group_residue_contacts', r"(.*?)_residue_contacts_g(.*?)$"), ('nn(dist)_spherical_angles', r"^nn\(dist\)_(.*?)_(.*?)_angle_spherical_(theta|psi)$")]
        self._set_feature_groups(feature_patterns = group_patterns, feature_unions = [['CM_group_residue_contacts', 'nn(dist)_spherical_angles']])
        base_model_col_index = self.feature_groups['nn(dist)_dists']
        col_index = self.feature_groups['CM_group_residue_contacts__nn(dist)_spherical_angles']
        

        num_boost_round = model_params['n_estimators']; del model_params['n_estimators']
        if self.labelling== 'discrete':
            model_params.update({'objective': 'multi:softprob', 'num_class': self.n_bins})
        
        evals = []

        # model training
        for j, train_jobs in enumerate(train_job_batches):
            print(f'train iteration: {j+1}/{n_train_job_batches}')
            DM_base = self._get_xgb_DMatrix(jobs = train_jobs, col_index = base_model_col_index, n_threads = n_threads)
            y_pred_base, _, _  = self._get_y(DM = DM_base, model = base_model)
            del DM_base
            t0 = time.time()
            DM = self._get_xgb_DMatrix(jobs = train_jobs, col_index = col_index, n_threads = n_threads, X_cols = y_pred_base)
            t1 = time.time(); dur = t1 - t0; n_features = DM.num_col(); n_observations = DM.num_row()
            print(f'train iteration: {j+1}/{n_train_job_batches}, n_features = {n_features}, n_observations = {n_observations}, DMatrix load time: {dur:.4}s')
            model = xgb.train(model_params, dtrain = DM, num_boost_round = num_boost_round) if j == 0 else xgb.train(model_params, dtrain = DM, num_boost_round = num_boost_round, xgb_model = model)
            t2 = time.time(); dur = t2 - t1; print(f'train duration: {dur:.4}s')
            y_pred, y_true, y_prob  = self._get_y(DM = DM, model = model)
            del DM
            eval = self._get_model_eval(y_pred = y_pred, y_true = y_true, y_prob = y_prob, tag = 'train', iter = j+1, obs = n_observations)
            evals.append({**model_params, **eval})


        save_model = kwargs.get('save_model', False)
        
        

        # model testing
        for j, test_jobs in enumerate(test_job_batches):
            print(f'test iteration: {j+1}/{n_test_job_batches}')
            t0 = time.time()
            DM_base = self._get_xgb_DMatrix(jobs = test_jobs, col_index = base_model_col_index, n_threads = n_threads)
            y_pred_base, _, _  = self._get_y(DM = DM_base, model = base_model)
            del DM_base
            DM = self._get_xgb_DMatrix(jobs = test_jobs, col_index = col_index, n_threads = n_threads, X_cols = y_pred_base)
            t1 = time.time(); dur = t1 - t0; n_features = DM.num_col(); n_observations = DM.num_row()
            y_pred, y_true, y_prob  = self._get_y(DM = DM, model = model)
            del DM
            eval = self._get_model_eval(y_pred = y_pred, y_true = y_true, y_prob = y_prob, tag = 'test', iter = j+1, obs = n_observations)
            evals.append({**model_params, **eval})

        # evaluation for train and testing
        
        eval_data = pd.DataFrame(evals)
        eval_data.to_csv(f'{self.model_dir}/{model_name}_train_test_evals.csv')
        print(eval_data)

        exit()

        decoy_level_results_train, rmsd_results_train = self._get_decoy_analysis(model = model, jobs = train_eval_jobs, col_index = col_index, data = data, tag = 'train')
        decoy_level_results_test, rmsd_results_test = self._get_decoy_analysis(model = model, jobs = test_eval_jobs, col_index = col_index, data = data, tag = 'test')

        decoy_level_results = pd.concat([decoy_level_results_train, decoy_level_results_test])
        rmsd_results = pd.concat([rmsd_results_train, rmsd_results_test])
        
        decoy_level_results.to_csv(f'{self.model_dir}/{model_name}_decoy_level_results.csv', index = False)
        rmsd_results.to_csv(f'{self.model_dir}/{model_name}_rmsd_results.csv', index = False)
        
        rmsd_res_pivot = pd.pivot_table(
            rmsd_results, 
            index = 'tag', 
            values = ['cb_rmsd','top_rmsd(model)', 'top5_rmsd(model)', 'top_rank(model)', 'top5_rank(model)', 'top_rmsd(energy)', 'top5_rmsd(energy)', 'top_rank(energy)', 'top5_rank(energy)'], 
            aggfunc = 'mean'
            )
        print(rmsd_res_pivot)




    def run_cv(self, param_grid: Dict[str,List[Any]], n_train_batches: Optional[int] = 5, n_test_batches: Optional[int] = 2) -> None:
        data = pd.read_csv(self.graph_indexing_path)
        data = self._get_downsampled_train_observations(data = data)
        train_jobs, test_jobs = self._get_jobs(data = data)
        self._set_train_transforms(jobs = train_jobs)
        
        n_rounds_default = 2000

        cv_evals = []
        param_combinations = list(itertools.product(*param_grid.values())); n_pc = len(param_combinations)
        train_job_batches = np.array_split(train_jobs, n_train_batches); n_train_job_batches = len(train_job_batches)
        test_job_batches = np.array_split(test_jobs, n_test_batches); n_test_job_batches = len(test_job_batches)

        group_patterns = [('nn(dist)_dists', r"^nn\(dist\)_(.*?)_(.*?)_dists$")]
        self._set_feature_groups(patterns = group_patterns)
        col_index = self.feature_groups['nn(dist)_dists']
            
        for i, param_combination in enumerate(param_combinations):
            print(f'iteration: {i+1}/{n_pc}')
            params = {**dict(zip(param_grid.keys(), param_combination)), **{'tree_method': 'gpu_hist'}}

            n_rounds = params.get('n_estimators', n_rounds_default)
            
            if self.labelling == 'discrete':
                params.update({'objective': 'multi:softprob', 'num_class': self.n_bins})
            
            for j, train_jobs in enumerate(train_job_batches):
                print(f'train iteration: {j+1}/{n_train_job_batches}')
                t0 = time.time()
                DM = self._get_xgb_DMatrix(jobs=train_jobs, col_index = col_index)
                t1 = time.time(); dur = t1 - t0; print(f'duration: {dur:.4}s')
                model = xgb.train(params, DM, n_rounds) if j == 0 else xgb.train(params, DM, n_rounds, xgb_model=model)
                t2 = time.time(); dur = t2 - t1; print(f'train duration: {dur:.4}s')
                if self.labelling == 'discrete':
                    y_pred, y_true, y_prob = self._get_y_classifier(DM=DM, model=model)
                    eval = self._get_classifier_eval(y_pred = y_pred, y_true = y_true, y_prob = y_prob, tag = 'train', iter = j + 1, obs = DM.num_row())
                else:
                    y_pred, y_true = self._get_y_regressor(DM = DM, model = model)
                    eval = self._get_regressor_eval(y_pred = y_pred, y_true = y_true, tag = 'train', iter = j + 1, obs = DM.num_row())
                del DM
                train_eval_j = {**params, **eval}
                cv_evals.append(train_eval_j)       

            
            for j, test_job in enumerate(test_job_batches):
                print(f'test iteration: {j+1}/{n_test_job_batches}')
                DM = self._get_xgb_DMatrix(jobs=test_job,col_index=col_index)
                if self.labelling == 'discrete':
                    y_pred, y_true, y_prob = self._get_y_classifier(DM=DM, model=model)
                    eval = self._get_classifier_eval(y_pred = y_pred, y_true = y_true, y_prob = y_prob, tag = 'test', iter = j + 1, obs = DM.num_row())
                else:
                    y_pred, y_true = self._get_y_regressor(DM = DM, model = model)
                    eval = self._get_regressor_eval(y_pred = y_pred, y_true = y_true, tag = 'test', iter = j + 1, obs = DM.num_row())
                del DM
                test_eval_j = {**params, **eval}
                print(test_eval_j)
                cv_evals.append(test_eval_j)

            

        data = pd.DataFrame(cv_evals)
        data.to_csv('xgb_nn_dist_params_again.csv')
        data['param_config'] = data['n_estimators'].astype(str) + "__" + data['eta'].astype(str)
        pivot = pd.pivot_table(data, index = ['param_config', 'tag'], values = ['mae', 'rmse', 'mse'], aggfunc = 'mean')
        import pdb; pdb.set_trace()
        x = 0
        y = 2

    def _get_feature_importance_analysis(self, model: xgb.Booster, n_features: int, col_index: Optional[np.array] = None) -> None:
        feature_data = []
        for score in ['weight','gain', 'cover', 'total_gain', 'total_cover']:
            arr = np.zeros(shape = n_features)
            importances = model.get_score(importance_type = score)

            for key, val in importances.items():
                arr[int(key[1:])] = val
            arr /= np.sum(arr)
            
            feature_col_index = col_index if col_index is not None else np.arange(0, n_features)
            if self.energy_evals:
                feature_col_index = np.append(feature_col_index, np.array([self.feature_indexing['energy_evaluation'], self.feature_indexing['energy_evaluation_MMS']]))
            
            key_importance_tracking = {}
            for i, idx in enumerate(feature_col_index):
                feature = self.index_feature_map[idx]
                if feature in key_importance_tracking:
                    key_importance_tracking[feature]['importance'].append(arr[i])
                else:
                    key_importance_tracking[feature] = {'score': score, 'importance': [arr[i]], 'n_dim': self.feature_indexing[feature].shape[0]}

            for key in key_importance_tracking:
                key_importance_tracking[key]['net_importance'] = np.sum(key_importance_tracking[key]['importance'])
                del key_importance_tracking[key]['importance']

            feature_data.append(pd.DataFrame.from_dict(key_importance_tracking, orient = 'index').sort_values(by = ['net_importance'], ascending = False).reset_index())        
        importance_data = pd.concat(feature_data)
        importance_data.to_csv(f'{model_name}_feature_importance.csv')

    def run_feature_analysis(self, n_train_batches: Optional[int] = 4, n_test_batches: Optional[int] = 3) -> None:
        data = pd.read_csv(self.graph_indexing_path)
        data = self._get_downsampled_train_observations(data = data)
        train_jobs, test_jobs = self._get_jobs(data = data, n_index_splits = 3, shuffle = True)
        train_eval_jobs, test_eval_jobs = self._get_jobs(data = data)
        train_job_batches = np.array_split(train_jobs, n_train_batches); n_train_job_batches = len(train_job_batches)
        test_job_batches = np.array_split(test_jobs, n_test_batches); n_test_job_batches = len(test_job_batches)
        self._set_train_transforms(jobs = train_jobs)

        group_patterns = [
            ('nn(dist)_dists', r"^nn\(dist\)_(.*?)_(.*?)_dists$"), 
            ('nn(dist)_angles', r"^nn\(dist\)_(.*?)_(.*?)_angle$"),
            ('nn(dist)_spherical_angles', r"^nn\(dist\)_(.*?)_(.*?)_angle_spherical_(theta|psi)$"),
            ('CM_group_residue_contacts', r"(.*?)_residue_contacts_g(.*?)$"),
            ('CM_groups', r"(.*?)_g(.*?)$")
            ]

        self._set_feature_groups(patterns = group_patterns, unions = [['nn(dist)_dists', 'nn(dist)_angles'], ['nn(dist)_dists', 'nn(dist)_angles', 'nn(dist)_spherical_angles']])

        params = {
            'objective': 'reg:squarederror',
            'tree_method' : 'gpu_hist',
            'max_depth': 2,
            'reg_alpha': 5,
            'reg_lambda': 5,
            'learning_rate' : 0.01,
            }
        
        if self.labelling== 'discrete':
            params.update({'objective': 'multi:softprob', 'num_class': self.n_bins})
        
        n_rounds = 2000
        
        for group_name, col_index in self.feature_groups.items():
            print(f'feature group: {group_name}')
            n_features = col_index.shape[0] + 2
            evals = []
            
            for j, train_jobs in enumerate(train_job_batches):
                t0 = time.time()
                DM = self._get_xgb_DMatrix(jobs = train_jobs, col_index = col_index)
                n_features = DM.num_col(); n_observations = DM.num_row()
                t1 = time.time(); dur = t1 - t0
                print(f'train iteration: {j+1}/{n_train_job_batches}, n_features = {n_features}, n_observations = {n_observations}, load time: {dur:.4}s')
                model = xgb.train(params, dtrain = DM, num_boost_round = n_rounds) if j == 0 else xgb.train(params, dtrain = DM, num_boost_round = n_rounds, xgb_model = model)
                t2 = time.time(); dur = t2 - t1; print(f'train duration: {dur:.4}s')
                
                if self.labelling == 'discrete':
                    y_pred, y_true, y_prob = self._get_y_classifier(DM=DM, model=model)
                    eval = self._get_classifier_eval(y_pred = y_pred, y_true = y_true, y_prob = y_prob, tag = 'train', iter = j + 1, obs = DM.num_row())
                else:
                    y_pred, y_true = self._get_y_regressor(DM = DM, model = model)
                    eval = self._get_regressor_eval(y_pred = y_pred, y_true = y_true, tag = 'train', iter = j + 1, obs = DM.num_row())
                
                del DM
                evals.append({**params, **eval})

            
            
            
            model_name = f'modelling/xgb_results/xgb_sq_objective_NE{n_rounds}_w(K+KS)_{group_name}'
            model.save_model(f'{model_name}.model')
            del model
            
            model = xgb.Booster()
            model.load_model(f'{model_name}.model')
            
            feature_data = []
            for score in ['weight','gain', 'cover', 'total_gain', 'total_cover']:
                arr = np.zeros(shape = n_features)
                importances = model.get_score(importance_type = score)

                for key, val in importances.items():
                    arr[int(key[1:])] = val
                arr /= np.sum(arr)
                
                #if col_index.shape != arr.shape and self.energy_evals:
                col_index_wK = np.append(col_index, np.array([self.feature_indexing['energy_evaluation'], self.feature_indexing['energy_evaluation_MMS']]))
                
                key_importance_tracking = {}
                for i, idx in enumerate(col_index_wK):
                    feature = self.index_feature_map[idx]
                    if feature in key_importance_tracking:
                        key_importance_tracking[feature]['importance'].append(arr[i])
                    else:
                        key_importance_tracking[feature] = {'score': score, 'importance': [arr[i]], 'n_dim': self.feature_indexing[feature].shape[0]}

                for key in key_importance_tracking:
                    key_importance_tracking[key]['net_importance'] = np.sum(key_importance_tracking[key]['importance'])
                    del key_importance_tracking[key]['importance']

                feature_data.append(pd.DataFrame.from_dict(key_importance_tracking, orient = 'index').sort_values(by = ['net_importance'], ascending = False).reset_index())
                    
            importance_data = pd.concat(feature_data)
            importance_data.to_csv(f'{model_name}_feature_importance.csv', index = False)
            print(importance_data.head(20))
            del feature_data
            
            
            for j, test_job in enumerate(test_job_batches):
                print(f'test iteration: {j+1}/{n_test_job_batches}')
                DM = self._get_xgb_DMatrix(jobs = test_job, col_index = col_index)
                if self.labelling == 'discrete':
                    y_pred, y_true, y_prob = self._get_y_classifier(DM = DM, model = model)
                    eval = self._get_classifier_eval(y_pred = y_pred, y_true = y_true, y_prob = y_prob, tag = 'test', iter = j + 1, obs = DM.num_row())
                else:
                    y_pred, y_true = self._get_y_regressor(DM = DM, model = model)
                    eval = self._get_regressor_eval(y_pred = y_pred, y_true = y_true, tag = 'test', iter = j + 1, obs = DM.num_row())
                del DM
                evals.append({**params, **eval})

            eval_data = pd.DataFrame(evals)

            eval_data.to_csv(f'{model_name}_train_test_evals.csv')
            print(eval_data)
            
            
            results = self._get_graph_rmsds(test_eval_jobs, model, col_index)
            data_test = data[data['tag'] == 'test'].copy()
            data_test['model_rmsd'] = results['model_rmsd'].values
            data_test['target_rmsd'] = results['target_rmsd'].values
            res_test, rmsd_res_test = self.evaluate_predictions(data_test, tag = 'test')
            
            results = self._get_graph_rmsds(train_eval_jobs, model, col_index)
            data_train = data[data['tag'] == 'train'].copy()
            data_train['model_rmsd'] = results['model_rmsd'].values
            data_train['target_rmsd'] = results['target_rmsd'].values
            res_train , rmsd_res_train = self.evaluate_predictions(data_train, tag = 'train')
            
            rmsd_res = pd.concat([rmsd_res_test, rmsd_res_train])
            rmsd_res.to_csv(f'{model_name}_rmsd_ranking_results.csv')

            rmsd_res_pivot = pd.pivot_table(rmsd_res, index = 'tag', values = ['cb_rmsd','top_rmsd(model)', 'top5_rmsd(model)', 'top_rank(model)', 'top5_rank(model)', 'top_rmsd(energy)', 'top5_rmsd(energy)', 'top_rank(energy)', 'top5_rank(energy)'], aggfunc = 'mean')
            print(group_name)
            print(rmsd_res_pivot)
            del model

    def with_or_without_K(self, n_train_batches: Optional[int] = 6, n_test_batches: Optional[int] = 2, n_estimators: Optional[int] = 3000) -> None:
        
        data = pd.read_csv(self.graph_indexing_path)
        data = self._get_downsampled_train_observations(data = data)
        train_jobs, test_jobs = self._get_jobs(data = data, n_index_splits = 5, shuffle = True)
        train_eval_jobs, test_eval_jobs = self._get_jobs(data = data)
        self._set_train_transforms(jobs = train_jobs)
        train_job_batches = np.array_split(train_jobs, n_train_batches); n_train_job_batches = len(train_job_batches)
        test_job_batches = np.array_split(test_jobs, n_test_batches); n_test_job_batches = len(test_job_batches)

        group_patterns = [
            ('nn(dist)_dists', r"^nn\(dist\)_(.*?)_(.*?)_dists$"), 
            ('nn(dist)_angles', r"^nn\(dist\)_(.*?)_(.*?)_angle$"),
            ('nn(dist)_spherical_angles', r"^nn\(dist\)_(.*?)_(.*?)_angle_spherical_(theta|psi)$"),
            ('CM_group_residue_contacts', r"(.*?)_residue_contacts_g(.*?)$"),
            ('CM_groups', r"(.*?)_g(.*?)$")
            ]
        
        self._set_feature_groups(
            patterns = group_patterns, 
            unions = [
                    ['nn(dist)_dists', 'nn(dist)_angles','nn(dist)_spherical_angles'], 
                    ['nn(dist)_dists', 'nn(dist)_angles', 'nn(dist)_spherical_angles', 'CM_group_residue_contacts']
                ]
                )
        
        params = {'objective': 'reg:squarederror', 'tree_method' : 'gpu_hist', 'max_depth': 2, 'min_child_weight': 5, 'eta': 0.1, 'reg_alpha': 1000, 'reg_lambda': 1000, 'n_estimators': n_estimators}
        
        if self.labelling== 'discrete':
            params.update({'objective': 'multi:softprob', 'num_class': self.n_bins})

        feature_groups = {}
        for group_name, col_index in self.feature_groups.items():
            feature_groups[(group_name, True)] = col_index
            feature_groups[(group_name, False)] = col_index
        
        
        for (group_name, K_switch), col_index in feature_groups.items():
            print(f'feature group: {group_name}, include K = {K_switch}')
            evals = []
            num_boost_round = params.get('n_estimators', n_estimators)
            
            if K_switch:
                n_features = col_index.shape[0] + 2
                self.energy_evals = True
                model_name = f'modelling/xgb_K_results/xgb_NE{num_boost_round}_w(K)_{group_name}'
            else:
                n_features = col_index.shape[0]
                self.energy_evals = False
                model_name = f'modelling/xgb_K_results/xgb_NE{num_boost_round}_wo(K)_{group_name}'
            
            evals = []
            
            # training model
            for j, train_jobs in enumerate(train_job_batches):
                t0 = time.time()
                DM = self._get_xgb_DMatrix(jobs = train_jobs, col_index = col_index)
                t1 = time.time(); dur = t1 - t0; n_features = DM.num_col(); n_observations = DM.num_row()
                print(f'train iteration: {j+1}/{n_train_job_batches}, n_features = {n_features}, n_observations = {n_observations}, DMatrix load time: {dur:.4}s')
                model = xgb.train(params, dtrain = DM, num_boost_round = num_boost_round) if j == 0 else xgb.train(params, dtrain = DM, num_boost_round = num_boost_round, xgb_model = model)
                t2 = time.time(); dur = t2 - t1; print(f'train duration: {dur:.4}s')
                y_pred, y_true = self._get_y_regressor(DM = DM, model = model)
                eval = self._get_regressor_eval(y_pred = y_pred, y_true = y_true, tag = 'train', iter = j + 1, obs = n_observations)
                evals.append({**params, **eval})
                del DM
            
            model.save_model(f'{model_name}.model')
            del model
            model = xgb.Booster()
            model.load_model(f'{model_name}.model')

            print('extracting feature information')
            feature_data = []
            for score in ['weight','gain', 'cover', 'total_gain', 'total_cover']:
                arr = np.zeros(shape = n_features)
                importances = model.get_score(importance_type = score)

                for key, val in importances.items():
                    arr[int(key[1:])] = val
                arr /= np.sum(arr)
                
                feature_col_index = col_index
                if K_switch:
                    feature_col_index = np.append(feature_col_index, np.array([self.feature_indexing['energy_evaluation'], self.feature_indexing['energy_evaluation_MMS']]))
                
                key_importance_tracking = {}
                for i, idx in enumerate(feature_col_index):
                    feature = self.index_feature_map[idx]
                    if feature in key_importance_tracking:
                        key_importance_tracking[feature]['importance'].append(arr[i])
                    else:
                        key_importance_tracking[feature] = {'score': score, 'importance': [arr[i]], 'n_dim': self.feature_indexing[feature].shape[0]}

                for key in key_importance_tracking:
                    key_importance_tracking[key]['net_importance'] = np.sum(key_importance_tracking[key]['importance'])
                    del key_importance_tracking[key]['importance']

                feature_data.append(pd.DataFrame.from_dict(key_importance_tracking, orient = 'index').sort_values(by = ['net_importance'], ascending = False).reset_index())
                    
            importance_data = pd.concat(feature_data)
            gain = importance_data[importance_data['score'] == 'gain'].sort_values(by = ['net_importance'], ascending = False)
            print(gain.head(20))
            importance_data.to_csv(f'{model_name}_feature_importance.csv', index = False)

            del feature_data; del importance_data; del key_importance_tracking
            
            # testing model
            for j, test_job in enumerate(test_job_batches):
                t0 = time.time()
                DM = self._get_xgb_DMatrix(jobs = test_job, col_index = col_index)
                t1 = time.time(); dur = t1 - t0; n_features = DM.num_col(); n_observations = DM.num_row()
                print(f'train iteration: {j+1}/{n_test_job_batches}, n_features = {n_features}, n_observations = {n_observations}, DMatrix load time: {dur:.4}s')
                y_pred, y_true = self._get_y_regressor(DM = DM, model = model)
                eval = self._get_regressor_eval(y_pred = y_pred, y_true = y_true, tag = 'test', iter = j + 1, obs = DM.num_row())
                del DM
                evals.append({**params, **eval})

            eval_data = pd.DataFrame(evals)
            eval_data.to_csv(f'{model_name}_train_test_evals.csv')
            print(eval_data)
            del eval_data
            
            test_results = self._get_graph_rmsds(test_eval_jobs, model, col_index)
            data_test = data[data['tag'] == 'test'].copy()
            data_test['model_rmsd'] = test_results['model_rmsd'].values
            data_test['target_rmsd'] = test_results['target_rmsd'].values
            res_test, rmsd_res_test = self.evaluate_predictions(data_test, tag = 'test')
            
            train_results = self._get_graph_rmsds(train_eval_jobs, model, col_index)
            data_train = data[data['tag'] == 'train'].copy()
            data_train['model_rmsd'] = train_results['model_rmsd'].values
            data_train['target_rmsd'] = train_results['target_rmsd'].values
            res_train , rmsd_res_train = self.evaluate_predictions(data_train, tag = 'train')
            
            rmsd_res = pd.concat([rmsd_res_test, rmsd_res_train])
            rmsd_res.to_csv(f'{model_name}_rmsd_ranking_results.csv')

            res = pd.concat([res_test, res_train])
            res.to_csv(f'{model_name}_decoy_level_results.csv')

            rmsd_res_pivot = pd.pivot_table(rmsd_res, index = 'tag', values = ['cb_rmsd','top_rmsd(model)', 'top5_rmsd(model)', 'top_rank(model)', 'top5_rank(model)', 'top_rmsd(energy)', 'top5_rmsd(energy)', 'top_rank(energy)', 'top5_rank(energy)'], aggfunc = 'mean')
            print(group_name)
            print(rmsd_res_pivot)
            del model; del res; del train_results; del test_results; del res_test; del res_train

    def run(self, n_train_batches: Optional[int] = 6, n_test_batches: Optional[int] = 2, **kwargs: Dict[str,Any]) -> None:
        
        # train test split already defined
        data = pd.read_csv(self.graph_indexing_path)
        data = self._get_downsampled_train_observations(data = data)
        
        train_jobs, test_jobs = self._get_jobs(data = data, n_index_splits = 3, shuffle = True)
        #train_jobs, test_jobs = self._get_jobs(data = data)
        self._set_train_transforms(jobs = train_jobs)

        '''
        params = {
            'objective': 'reg:squarederror',
            'tree_method' : 'gpu_hist',
            'max_depth': 2,
            'reg_alpha': 5,
            'reg_lambda': 5,
            'learning_rate' : 0.01,
            } # n est = 2000
        '''
        
        
        
        params = {
            #'objective': 'reg:squarederror',
            'tree_method' : 'gpu_hist',
            'max_depth': 2,
            #'min_child_weight': 9,
            'eta': 0.05,
            'reg_alpha': 1000,
            'reg_lambda': 1000,
            }
        
        group_patterns = [('nn(dist)_dists', r"^nn\(dist\)_(.*?)_(.*?)_dists$")]
        self._set_feature_groups(patterns = group_patterns)
        col_index = self.feature_groups['nn(dist)_dists']
        
        if self.labelling== 'discrete':
            params.update({'objective': 'multi:softprob', 'num_class': self.n_bins})
        
        n_rounds = 4000

        evals = []
        
        train_job_batches = np.array_split(train_jobs, n_train_batches); n_train_job_batches = len(train_job_batches)
        test_job_batches = np.array_split(test_jobs, n_test_batches); n_test_job_batches = len(test_job_batches)
        
        
        for j, train_jobs in enumerate(train_job_batches):
            print(f'train iteration: {j+1}/{n_train_job_batches}')
            t0 = time.time()
            DM = self._get_xgb_DMatrix(jobs = train_jobs, col_index = col_index)
            t1 = time.time(); dur = t1 - t0; n_features = DM.num_col(); n_observations = DM.num_row()
            print(f'train iteration: {j+1}/{n_train_job_batches}, n_features = {n_features}, n_observations = {n_observations}, DMatrix load time: {dur:.4}s')
            model = xgb.train(params, dtrain = DM, num_boost_round = n_rounds) if j == 0 else xgb.train(params, dtrain = DM, num_boost_round = n_rounds, xgb_model = model)
            t2 = time.time(); dur = t2 - t1; print(f'train duration: {dur:.4}s')
            #y = self._get_y(DM = DM, model = model)
            #eval = self._get_model_eval({**y, **kwargs,})

            if self.labelling == 'discrete':
                y_pred, y_true, y_prob = self._get_y_classifier(DM=DM, model=model)
                eval = self._get_classifier_eval(y_pred = y_pred, y_true = y_true, y_prob = y_prob, tag = 'train', iter = j + 1, obs = DM.num_row())
            else:
                y_pred, y_true = self._get_y_regressor(DM = DM, model = model)
                eval = self._get_regressor_eval(y_pred = y_pred, y_true = y_true, tag = 'train', iter = j + 1, obs = DM.num_row())
            del DM
            evals.append({**params, **eval})

        del train_job_batches

        save_model = kwargs.get('save_model', False)
        model_name = f'xgboost_sq_objective_nn(dist)_dists_NE{n_rounds}_alpha1000_lambda1000_max_depth_2_eta01'
        
        # save model and reload it, this helps with CPU/GPU-RAM
        if save_model:
            model.save_model(f'{model_name}.model')
            del model
            model = xgb.Booster()
            model.load_model(f'{model_name}.model')

        
        feature_importance_analysis = kwargs.get('feature_importance_analysis', False)
        if feature_importance_analysis:
            self._get_feature_importance_analysis(model = model, n_features = n_features)

            

        

        for j, test_job in enumerate(test_job_batches):
            print(f'test iteration: {j+1}/{n_test_job_batches}')
            DM = self._get_xgb_DMatrix(jobs = test_job, col_index = col_index)
            if self.labelling == 'discrete':
                y_pred, y_true, y_prob = self._get_y_classifier(DM = DM, model = model)
                eval = self._get_classifier_eval(y_pred = y_pred, y_true = y_true, y_prob = y_prob, tag = 'test', iter = j + 1, obs = DM.num_row())
            else:
                y_pred, y_true = self._get_y_regressor(DM = DM, model = model)
                eval = self._get_regressor_eval(y_pred = y_pred, y_true = y_true, tag = 'test', iter = j + 1, obs = DM.num_row())
            del DM
            evals.append({**params, **eval})

        eval_data = pd.DataFrame(evals)

        eval_data.to_csv(f'{model_name}_train_test_evals.csv')
        print(eval_data)

        data = pd.read_csv(self.graph_indexing_path)
        train_jobs, test_jobs = self._get_jobs(data = data)
        self._set_train_transforms(jobs = train_jobs)
        
        results = self._get_graph_rmsds(test_jobs, model,col_index)
        data_test = data[data['tag'] == 'test'].copy()
        data_test['model_rmsd'] = results['model_rmsd'].values
        data_test['target_rmsd'] = results['target_rmsd'].values
        res_test, rmsd_res_test = self.evaluate_predictions(data_test, tag = 'test')
        
        results = self._get_graph_rmsds(train_jobs, model,col_index)
        data_train = data[data['tag'] == 'train'].copy()
        data_train['model_rmsd'] = results['model_rmsd'].values
        data_train['target_rmsd'] = results['target_rmsd'].values
        res_train , rmsd_res_train = self.evaluate_predictions(data_train, tag = 'train')
        
        rmsd_res = pd.concat([rmsd_res_test, rmsd_res_train])
        rmsd_res.to_csv(f'{model_name}_rmsd_ranking_results.csv')

        rmsd_res_pivot = pd.pivot_table(rmsd_res, index = 'tag', values = ['cb_rmsd','top_rmsd(model)', 'top5_rmsd(model)', 'top_rank(model)', 'top5_rank(model)', 'top_rmsd(energy)', 'top5_rmsd(energy)', 'top_rank(energy)', 'top5_rank(energy)'], aggfunc = 'mean')
        print(rmsd_res_pivot)
        import pdb; pdb.set_trace()
        x = 1        

        print(rmsd_res_pivot)
        print('testing hopeful')

    def build_stacking_regressor(self, n_train_batches: Optional[int] = 6, n_test_batches: Optional[int] = 2) -> None:
        
        # 
        data = pd.read_csv(self.graph_indexing_path)
        data = self._get_downsampled_train_observations(data = data)
        train_jobs, test_jobs = self._get_jobs(data = data)
        self._set_train_transforms(jobs = train_jobs)

        train_job_batches = np.array_split(train_jobs, n_train_batches); n_train_job_batches = len(train_job_batches)
        test_job_batches = np.array_split(test_jobs, n_test_batches); n_test_job_batches = len(test_job_batches)

        aux = [
            'residue_type',
            'phi','omega','psi',
            'bb_com_coords','sc_com_coords',
            'bb_pairwise_distance_matrix', 'bb_pairwise_angle_matrix',
            'bb_gyration_tensor','bb_radius_of_gyration','sc_radius_of_gyration'
            ]
        aux_regex_matches = '|'.join(aux)


        patterns = [
            ('COM', r"^(.*?)_COM_(dist|angle)$"),
            ('nn(dist)_dists', r"^nn\(dist\)_(.*?)_(.*?)_dists$"),
            ('nn(dist)_angles', r"^nn\(dist\)_(.*?)_(.*?)_angle$"),
            ('entropy', r"^(.*?)_entropy\(decoy\)$"),
            #('aux', r"(" + '|'.join(aux_regex_matches) + ")")
        ]
        unions = [['COM','nn(dist)_dists', 'nn(dist)_angles','entropy']]
        self._set_feature_groups(patterns = patterns, unions = unions)
        feature_groups = {**self.feature_groups, **{'all': None}}
       
        models = []
        params = {'tree_method': 'gpu_hist', 'reg_alpha': 5, 'reg_lambda': 10, 'max_depth': 1}
        n_rounds = 2000
        n_groups = len(feature_groups)
        loaded_models = [mod.strip('.model') for mod in os.listdir(self.model_dir) if mod.endswith('.model')]


        for i, (feature, index) in enumerate(feature_groups.items()):
            model_name = f'{feature}_base_xgboost'
            if model_name in loaded_models:
                print('skipping:', feature)
                continue


            print(f'feature set {i+1}/{n_groups}')
            evals = []
            for j, train_jobs in enumerate(train_job_batches[0:5]):
                print(f'train iteration: {j+1}/{n_train_job_batches}')
                t0 = time.time()
                DM = self._get_xgb_DMatrix(jobs = train_jobs, col_index = index)
                t1 = time.time(); dur = t1 - t0; print(f'duration: {dur:.4}s')
                model = xgb.train(params, DM, n_rounds) if j == 0 else xgb.train(params, DM, n_rounds, xgb_model=model)
                t2 = time.time(); dur = t2 - t1; print(f'train duration: {dur:.4}s')
                y_pred, y_true = self._get_y_regressor(DM = DM, model = model)
                eval = self._get_regressor_eval(y_pred = y_pred, y_true = y_true, tag = 'base_regressor_train', iter = j + 1, obs = DM.num_row())
                del DM
                evals.append({**eval, **{'features': str(feature)}})

            Y_pred_train, Y_true_train = [], []
            for j, train_jobs in enumerate(train_job_batches[5:]):
                print(f'train iteration: {j+1}/{n_train_job_batches}')
                t0 = time.time()
                DM = self._get_xgb_DMatrix(jobs = train_jobs, col_index = index)
                t1 = time.time(); dur = t1 - t0; print(f'duration: {dur:.4}s')
                y_pred_train, y_true_train = self._get_y_regressor(DM = DM, model = model)
                Y_pred_train.append(y_pred_train); Y_true_train.append(y_true_train)
                eval = self._get_regressor_eval(y_pred = y_pred_train, y_true = y_true_train, tag = 'final_regressor_train', iter = j + 1, obs = DM.num_row())
                del DM
                y_pred_train, y_true_train = None, None
                evals.append({**eval, **{'features': str(feature)}})

            Y_pred_test, Y_true_test = [], []
            for j, test_job in enumerate(test_job_batches):
                print(f'test iteration: {j+1}/{n_test_job_batches}')
                DM = self._get_xgb_DMatrix(jobs=test_job, col_index = index)
                y_pred_test, y_true_test = self._get_y_regressor(DM = DM, model = model)
                Y_pred_test.append(y_pred_test); Y_true_test.append(y_true_test)
                eval = self._get_regressor_eval(y_pred = y_pred_test, y_true = y_true_test, tag = 'final_regressor_test', iter = j + 1, obs = DM.num_row())
                del DM
                y_pred_test, y_true_test = None, None
                evals.append({**eval, **{'features': str(feature)}})


            model.save_model(f'{self.model_dir}/{model_name}.model')
            model_evals = pd.DataFrame(evals)
            model_evals.to_csv(f'{self.model_dir}/{model_name}_evaluation.csv')
            np.savez(f'{self.model_dir}/{model_name}_stacking_info.npz', X_train = np.vstack(Y_pred_train), y_train = np.vstack(Y_true_train), X_test = np.vstack(Y_pred_test), y_test = np.vstack(Y_true_test))
            del model; del Y_pred_train; del Y_true_train; del Y_pred_test; del Y_true_test; del model_evals; del evals
            gc.collect()


        import pdb; pdb.set_trace()
        X_train = np.hstack([mod[1] for mod in models])
        y_train = models[0][2]
        X_test = np.hstack([mod[3] for mod in models])
        y_test = models[0][4]


        final_model = GradientBoostingRegressor(loss = 'squared_error')
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test).reshape(-1,1)
        final_eval = self._get_regressor_eval(y_pred = y_pred, y_true = y_test, tag = 'final_test', obs = y_test.shape[0])
        print(final_eval)
        import pdb; pdb.set_trace()
        x = 1


        np.savez('stacking_regressor_data.npz', X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)

    def build_final_regressor(self) -> None:
        model_npz_files = [mod for mod in os.listdir(self.model_dir) if mod.endswith('.npz')]
        X_tr = []; y_tr = []; X_te = []; y_te = []
        for npz_file in model_npz_files:
            print(npz_file)
            data = np.load(f'{self.model_dir}/{npz_file}')
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            X_tr.append(X_train)
            y_tr.append(y_train)
            X_te.append(X_test)
            y_te.append(y_test)

        
        X_train = np.hstack(X_tr)
        X_test= np.hstack(X_te)
 
        model = xgb.XGBRegressor(**{'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 2000, 'objective': 'reg:squarederror', 'subsample': 0.8, 'tree_method': 'gpu_hist'})  # Set "tree_method" to 'gpu_hist' for GPU acceleration
        #scaler = MinMaxScaler()
        #y_train = scaler.fit_transform(y_train)
        #y_test_scaled = scaler.transform(y_test)
        #y_pred_train = scaler.inverse_transform(y_pred_train)
        #y_pred_test = scaler.inverse_transform(y_pred_test)
        import pdb; pdb.set_trace()
        model.fit(X_train, y_train.ravel())
        y_pred_train = model.predict(X_train).reshape(-1,1)
        train_eval = self._get_regressor_eval(y_pred = y_pred_train, y_true = y_train)
        print(train_eval)
        y_pred_test = model.predict(X_test).reshape(-1,1)
        test_eval = self._get_regressor_eval(y_pred = y_pred_test, y_true = y_test)
        print(test_eval)
        
    def full_train(self, n_train_batches: Optional[int] = 8) -> None:
        data = pd.read_csv(self.graph_indexing_path)
        data['tag'] = 'train'
        train_jobs, _ = self._get_jobs(data = data, n_index_splits = 4, shuffle = True, n_train_batches = n_train_batches)
        self._set_train_transforms(jobs = train_jobs)
        train_job_batches = np.array_split(train_jobs, n_train_batches); n_train_job_batches = len(train_job_batches)
        

        group_patterns = [('nn(dist)_dists', r"^nn\(dist\)_(.*?)_(.*?)_dists$")]
        self._set_feature_groups(patterns = group_patterns)
        col_index = self.feature_groups['nn(dist)_dists']
        
        params = {'objective': 'reg:squarederror', 'tree_method' : 'gpu_hist', 'max_depth': 3, 'min_child_weight': 5, 'eta': 0.1, 'reg_alpha': 1000, 'reg_lambda': 1000}
        num_boost_round = params.get('n_estimators', 5000)
        col_index = self.feature_groups['nn(dist)_dists']
        feature_groups = {}
        model_name = f'xgb_NE5000_nn(dists)_dists_fully_trained_TB{n_train_batches}'
        evals = []
        # training model
        for j, train_jobs in enumerate(train_job_batches):
            t0 = time.time()
            DM = self._get_xgb_DMatrix(jobs = train_jobs, col_index = col_index)
            t1 = time.time(); dur = t1 - t0; n_features = DM.num_col(); n_observations = DM.num_row()
            print(f'train iteration: {j+1}/{n_train_job_batches}, n_features = {n_features}, n_observations = {n_observations}, DMatrix load time: {dur:.4}s')
            model = xgb.train(params, dtrain = DM, num_boost_round = num_boost_round) if j == 0 else xgb.train(params, dtrain = DM, num_boost_round = num_boost_round, xgb_model = model)
            t2 = time.time(); dur = t2 - t1; print(f'train duration: {dur:.4}s')
            y_pred, y_true = self._get_y_regressor(DM = DM, model = model)
            eval = self._get_regressor_eval(y_pred = y_pred, y_true = y_true, tag = 'train', iter = j + 1, obs = n_observations)
            evals.append({**params, **eval})
            del DM
        
            
        model.save_model(f'{model_name}.model')
        del model

        train_evals = pd.DataFrame(evals)
        print(train_evals)
            
    def decoy_testing(self, model_path: str, **kwargs: Dict[str,Any]) -> None:
        model = xgb.Booster()
        model.load_model(fname = model_path)
        model_name = model_path.split('/')[-1].strip('.model')
        graph_name = self.graph_indexing_path.split('/')[-1].strip('.csv')

        data = pd.read_csv(self.graph_indexing_path)
        data = self._get_downsampled_train_observations(data = data)
        _, test_eval_jobs = self._get_jobs(data = data)

        group_patterns = [('nn(dist)_dists', r"^nn\(dist\)_(.*?)_(.*?)_dists$")]
        self._set_feature_groups(patterns = group_patterns)
        col_index = self.feature_groups['nn(dist)_dists']

        results = self._get_graph_rmsds(test_eval_jobs, model, col_index)
        data['model_rmsd'] = results['model_rmsd'].values
        data['target_rmsd'] = results['target_rmsd'].values
        results, rmsd_results = self.evaluate_predictions(data, tag = 'test', model_name = model_name, **kwargs)
        print(rmsd_results['top_rmsd(model)'].mean(), rmsd_results['top_rmsd(energy)'].mean())
        results.to_csv(f'{self.graph_path}/model({model_name})_graphs({graph_name})_decoy_level_results.csv', index = False)
        rmsd_results.to_csv(f'{self.graph_path}/model({model_name})_graphs({graph_name})_rmsd_results.csv', index = False)
        


if __name__ == '__main__':
    engine = NodeLevelModel(graph_path = 'data/graphs', graph_indexing_path = 'data/graphs/decoys_n1181000.csv', energy_evals = True)
    #engine.full_train(n_train_batches = 2)


    #model = xgb.Booster()
    #model.load_model('modelling/xgb_K_results/xgb_NE3000_w(K)_nn(dist)_dists.model')
    
    #engine.nested_model(n_train_batches = 5,
    #                    n_test_batches = 3,
    #                    model_name = 'test_name',
    #                    base_model = model, 
    #                    model_params = {'n_estimators': 2000, 'objective': 'reg:squarederror', 'tree_method' : 'gpu_hist', 'max_depth': 2, 'min_child_weight': 5, 'eta': 0.1})
    

    #engine = NodeLevelModel(graph_path = 'dataset_A/rosetta_graphs', graph_indexing_path = 'dataset_A/rosetta_graphs/decoys_n24500.csv', energy_evals = True)
    #engine.train_test(n_train_batches = 1, 
    #                  n_test_batches = 1, 
    #                  model_name = 'baby_xgb_model', 
    #                  model_params = {'n_estimators': 3000, 'objective': 'reg:squarederror', 'tree_method' : 'gpu_hist', 'max_depth': 2, 'min_child_weight': 5, 'eta': 0.1, 'reg_alpha': 1000, 'reg_lambda': 1000})
    
    
    
    
    
    '''
    param_grid = {
        'max_depth': [1],
        'reg_alpha': [2],
        'reg_lambda': [2],
        #'min_child_weight': [7,8,9],
        'n_estimators': [2500,3000,3500],
        'eta': [0.05,0.1]
    }

    # TUNE N_ESTIMATORS + ETA;
    
    engine.with_or_without_K(n_train_batches = 8, n_test_batches = 4)
    engine.run(n_train_batches = 8, n_test_batches = 4)
    #engine.run_cv(param_grid = param_grid)
    #engine.run_feature_analysis(n_train_batches = 8, n_test_batches = 4)

    #engine.build_final_regressor()
    '''





