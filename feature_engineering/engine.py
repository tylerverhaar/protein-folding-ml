import os, time, Bio
import multiprocessing as mp
import pandas as pd
from typing import List, Any, Dict, Union, Callable, Optional, Tuple
from Bio.PDB import PDBParser
from feature_engineering.classes.base_sequence import BaseSequence
from feature_engineering.classes.decoy_loop import DecoyLoop
from feature_engineering.classes.decoy_sequence import DecoySequence
from feature_engineering.classes.decoy_set import DecoySet
import concurrent.futures

class FeatureEngineeringEngine:

    def __init__(self, pdb_path: str, decoy_path: str, graph_path: str) -> None:
        """
        Initialize the FeatureEngineeringEngine.

        Args:
            pdb_path (str): Path to the PDB file.
            decoy_path (str): Path to the decoy file.
            graph_path (str): Path to the graph file.

        Raises:
            AssertionError: If pdb_path or decoy_path does not exist.

        Note:
            If graph_path does not exist, an empty file will be created at the specified path.
        """
        assert os.path.exists(pdb_path), f"pdb_path '{pdb_path}' does not exist."
        assert os.path.exists(decoy_path), f"decoy_path '{decoy_path}' does not exist."

        self.pdb_path = pdb_path
        self.decoy_path = decoy_path
        self.graph_path = graph_path

        if not os.path.exists(graph_path):
            print('creating directory at: {}'.format(graph_path))
            os.mkdir(graph_path)

    def get_pdb_structure(self, protein_pdb_path: str, protein_id: str) -> Bio.PDB.Structure.Structure:
        """
        Retrieve the PDB structure for the specified protein.

        Args:
            protein_pdb_path (str): Path to the directory containing the PDB files.
            protein_id (str): ID of the protein to retrieve the structure for.

        Returns:
            Bio.PDB.Structure.Structure: PDB structure object representing the protein.

        """
        try:
            parser = PDBParser()
            protein_pdb_path = f'{protein_pdb_path}/{protein_id}.pdb'
            structure = parser.get_structure("protein", protein_pdb_path)
            return structure
        except:
            return None

    def get_base_sequence(self, protein_pdb_path: str, protein_id: str) -> BaseSequence:
        """
        Retrieve the base sequence for the specified protein.

        Args:
            protein_pdb_path (str): Path to the directory containing the PDB files.
            protein_id (str): ID of the protein to retrieve the base sequence for.

        Returns:
            BaseSequence: Base sequence object representing the protein's sequence.

        """
        # get raw structure
        structure = self.get_pdb_structure(protein_pdb_path = protein_pdb_path, protein_id = protein_id)

        if structure is None:
            return None
        
        # define base sequence derived from structure
        sequence = BaseSequence(protein_id = protein_id, structure = structure)
        return sequence

    def get_decoys_pdb_representation(self, decoy_path: str, decoy_file_id: str, exclude_hydrogen: Optional[bool] = False) -> List[str]:
        """
        Read and retrieve the PDB representation of decoys from a given file.

        Args:
            loop_decoy_path (str): Path to the directory containing the decoy files.
            decoy_file_id (str): Identifier of the specific decoy file.

        Returns:
            List[str]: List of strings representing the PDB lines of the decoy.

        """
        decoy_id_path = f'{decoy_path}/{decoy_file_id}.txt'
        with open(decoy_id_path, 'r') as f:
            flines = f.readlines()
            f.close()

        if exclude_hydrogen:
            flines = [line for line in flines if line[-1] != 'H']
            
        return flines

    def get_decoy_residue_decomposition(self, decoy_file_lines: List[str], start_residue: int, end_residue: int) -> List[DecoyLoop]:
        """
        Extracts the residue decomposition of decoys from the lines of a decoy file.

        Args:
            decoy_file_lines (List[str]): List of strings representing the lines of the decoy file.
            start_residue (int): Starting residue number for the residue decomposition.
            end_residue (int): Ending residue number for the residue decomposition.

        Returns:
            List[DecoyLoop]: List of DecoyLoop objects.

        """
        decoys = []
        decoy_id = 1
        
        for idx in range(len(decoy_file_lines)):
            if decoy_file_lines[idx][0:4] == 'LOOP':
                vals = [float(v) for v in decoy_file_lines[idx].strip('\n').split(' ')[-2:]]
                current_loop = {
                    'decoy_id' : decoy_id,
                    'energy_evaluation' : 1 * vals[0] + 10 * vals[1],
                    'residues' : {i : {'atoms' : [], 'read' : False} for i in range(start_residue, end_residue+1)}
                }
            elif decoy_file_lines[idx][0:3] == 'END':
                decoy_id += 1
                loop = DecoyLoop(**current_loop)
                if loop.invalid_loop:
                    return []
                decoys.append(loop)
            elif decoy_file_lines[idx][0:12] == 'DIS_FEATURES':
                continue
            else:
                assert decoy_file_lines[idx][0:4] == 'ATOM', f'check index = {idx}'
                ln = decoy_file_lines[idx]
                atom_number = int(ln[6:11])
                atom_name = ln[12:16].replace(" ", "")
                residue_name = ln[17:20].replace(" ", "")
                chain_id = ln[21:22].replace(" ", "")
                residue_number = int(ln[22:26])
                x_coord = float(ln[30:38])
                y_coord = float(ln[38:46])
                z_coord = float(ln[46:54])
                current_loop['residues'][residue_number]['atoms'].append({
                        'atom_number' : atom_number,
                        'atom_name' : atom_name,
                        'x_coord' : x_coord,
                        'y_coord' : y_coord,
                        'z_coord' : z_coord
                    })
                if not current_loop['residues'][residue_number]['read']:
                    current_loop['residues'][residue_number].update({
                        'chain_id' : chain_id,
                        'residue_name' : residue_name,
                        'read' : True
                    })
        return decoys

    def get_decoy_loops(self, decoy_path: str, decoy_file_id: str) -> List[DecoyLoop]:
        """
        Retrieve the loops from a decoy file.

        Parameters:
        -----------
        decoy_path : str
            The path to the directory containing the decoy files.
        decoy_file_id : str
            The identifier of the decoy file.

        Returns:
        --------
        List[DecoyLoop]:
            The list of DecoyLoop objects representing the loops in the decoy file.
        """
        # Derive protein ID and loop endpoints
        _, start_residue, end_residue = decoy_file_id.split('_')
        start_residue = int(start_residue)
        end_residue = int(end_residue) if '-' not in end_residue else int(end_residue.split('-')[0])

        # Get PDB lines from decoy files
        decoy_file_lines = self.get_decoys_pdb_representation(decoy_path = decoy_path, decoy_file_id = decoy_file_id)

        # Construct loops for each decoy
        decoy_loops = self.get_decoy_residue_decomposition(decoy_file_lines = decoy_file_lines, start_residue = start_residue, end_residue = end_residue)
        return decoy_loops
    
    def process_decoy(
            self, 
            decoy_id: int, 
            decoy_loop: DecoyLoop, 
            protein_id: str, 
            base_structure: Bio.PDB.Structure.Structure, 
            start_residue: int, 
            end_residue: int
            ):

            decoy_sequence = DecoySequence(protein_id = protein_id,
                                           structure = base_structure,
                                           start_residue = start_residue,
                                           end_residue = end_residue,
                                           decoy_loop = decoy_loop)
            
            if decoy_sequence.invalid_sequence:
                return decoy_id, None
            
            decoy_sequence.set_decoy_loop()
            decoy_sequence.set_residue_features(dist_k_neighbors = 30, seq_k_neighbors = 2)
            return decoy_id, decoy_sequence

    def decoy_collection_process_wrapper(self, decoy_file_id: str, n_threads: Optional[int] = 1, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process decoys for a given protein.

        Parameters:
        -----------
        decoy_file_id : str
            The identifier of the decoy file.
        
        Returns:
        --------
        Dict[str, Any]:
            A dictionary containing runtime information and metadata about the processed decoys.
            The dictionary includes the following keys:
            - 'loading_time': The time taken to load the protein structure and decoy loops.
            - 'decoy_processing_time': The time taken to process the decoys.
            - 'total_time': The total processing time.
            - 'protein_id': The identifier of the protein.
            - 'start_residue': The start residue index of the decoy.
            - 'end_residue': The end residue index of the decoy.
        """
        protein_pdb_path = self.pdb_path
        decoy_path = self.decoy_path

        t0 = time.time()
        print('decoy_file_id',decoy_file_id)
        protein_id, start_residue, end_residue = decoy_file_id.split('_')
        start_residue = int(start_residue)
        end_residue = int(end_residue) if '-' not in end_residue else int(end_residue.split('-')[0])
        
        # define base structure and decoy loops
        base_structure = self.get_pdb_structure(protein_pdb_path = protein_pdb_path, protein_id = protein_id)
        if base_structure is None:
            return
        
        # extract decoy loops
        decoy_loops = self.get_decoy_loops(decoy_path = decoy_path, decoy_file_id = decoy_file_id)

        if len(decoy_loops) == 0:
            return
        
        # saving decoy results
        decoy_dict = {}
        jobs = [(decoy_id, decoy_loop, protein_id, base_structure, start_residue, end_residue) for decoy_id, decoy_loop in enumerate(decoy_loops)]
        t1 = time.time()

        n_threads = kwargs.get('n_threads', n_threads)
        
        if n_threads > 1:
            print('starting multithreading for no fucking reason')
            with concurrent.futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
                future_results = [executor.submit(self.process_decoy, *args) for args in jobs]
                for future in concurrent.futures.as_completed(future_results):
                    decoy_id, decoy_sequence = future.result()
                    if decoy_sequence is not None:
                        decoy_dict[decoy_id] = decoy_sequence

        else:
            for job in jobs:
                decoy_id, decoy_sequence = self.process_decoy(*job)
                if decoy_sequence is not None:
                        decoy_dict[decoy_id] = decoy_sequence
        t2 = time.time()
        decoy_set = DecoySet(decoy_dict = decoy_dict)
        try:
            decoy_set.set_decoy_entropy_values()
        except:
            print(f'error computing entropy for : {protein_id} loop {start_residue} - {end_residue}')
            return
        t3 = time.time()
        decoy_set.write_graph_representations(decoy_file_id = decoy_file_id, write_path = self.graph_path)
        t4 = time.time()
        # runtime information
        data = {
            'loading_time' : t1 - t0,
            'decoy_processing_time' : t2 - t1,
            'total_time' : t4 - t0,
            'protein_id' : protein_id,
            'start_residue' : start_residue,
            'end_residue' : end_residue
        }
        return data
    
    def run_debug(self, 
                  n_decoys: Optional[int] = -1, 
                  filter_completed: Optional[bool] = True,
                  msg_freq: Optional[int] = 10,
                  **kwargs: Dict[str, Any]
                  ) -> None:
        """
        Runs the main "run" function in a single thread for debugging purposes.

        Args:
            n_decoys (Optional[int]): The number of decoys to process. Defaults to 20.
            filter_completed (Optional[bool]): Whether to filter out completed decoys. Defaults to True.
            msg_freq (Optional[int]): The frequency of progress messages to display. Defaults to 10.
        """
        t0 = time.time()

        # Get a list of completed decoy file IDs
        completed_decoy_file_ids = [fname.strip('.npz') for fname in os.listdir(path = self.graph_path) if '.npz' in fname]

        # Get a list of all decoy file IDs
        decoy_file_ids = [fname.strip('.txt') for fname in os.listdir(path = self.decoy_path) if '.txt' in fname]

        if filter_completed:
            # Filter out completed decoys from the list
            decoy_file_ids = [fname for fname in decoy_file_ids if fname not in completed_decoy_file_ids]

        decoy_file_ids = decoy_file_ids[0:n_decoys] if n_decoys > 0 else decoy_file_ids
        
        print(f'generating {len(decoy_file_ids)} graphs')

        results = []
        t0_fl = time.time()
        for i, decoy_file_id in enumerate(decoy_file_ids):
            if (i % msg_freq == 0) and (i != 0):
                ti_fl = time.time()
                print(f'Completed: {i}/{len(decoy_file_ids)} iterations in {(ti_fl - t0_fl) / 60:.4}m, average rate: {(ti_fl - t0_fl) / (i+1):.4}s')
            t0_process = time.time()
            res = self.decoy_collection_process_wrapper(decoy_file_id, n_threads = 1)
            t1_process = time.time(); process_dur = t1_process - t0_process; print(f'processed {decoy_file_id} in {process_dur:.4}s')
            results.append(res)

        t1 = time.time()
        print(f'run_debug - completion: {t1 - t0:.4}')
        
    def run(
            self,
            n_decoys: Optional[int] = -1,
            n_processes: Optional[int] = mp.cpu_count() - 2, 
            filter_completed: Optional[bool] = True,
            **kwargs: Dict[str, Any]
            ) -> None:
        """
        Runs the main "run" function using multiple processes.

        Args:
            n_processes (Optional[int]): The number of processes to use. Defaults to mp.cpu_count() - 2.
            filter_completed (Optional[bool]): Whether to filter out completed decoys. Defaults to True.
        """
        
        if n_processes == 1:
            self.run_debug(n_decoys = n_decoys, filter_completed = filter_completed)

        # Get a list of completed decoy file IDs
        completed_decoy_file_ids = [fname.strip('.npz') for fname in os.listdir(path = self.graph_path) if '.npz' in fname]

        # Get a list of all decoy file IDs
        decoy_file_ids = [fname.strip('.txt') for fname in os.listdir(path = self.decoy_path) if '.txt' in fname]

        if filter_completed:
            # Filter out completed decoys from the list
            decoy_file_ids = [fname for fname in decoy_file_ids if fname not in completed_decoy_file_ids]

        decoy_file_ids = decoy_file_ids[0:n_decoys] if n_decoys > 0 else decoy_file_ids

        print(f'processing {len(decoy_file_ids)} decoys')
        t0 = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers = n_processes) as executor:
            print(f'initiated ProcessPoolExecutor with {n_processes} processes')
            futures = [executor.submit(self.decoy_collection_process_wrapper, fid) for fid in decoy_file_ids]
            results = [future.result() for future in futures]

        t1 = time.time()
        print(f'run - completion: {(t1 - t0) / 60:.4}m')
        


if __name__ == '__main__':
    pdb_path = 'data/pdb'
    decoy_path = 'data/decoys'
    graph_path = 'data/graphs_v1'
    