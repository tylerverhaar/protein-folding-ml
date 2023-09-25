import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any


class DataValidation:

    def __init__(self, graph_path: Optional[str] = 'data/new_graphs') -> None:
        self.graph_path = graph_path
        self.npz_files = [f for f in os.listdir(graph_path) if f.endswith('npz')]

    def _get_npz_file_info(self, npz_file: str) -> Dict:
        assert npz_file.endswith('.npz')
        fpath = f'{self.graph_path}/{npz_file}'
        data = np.load(fpath)
        files = data.files
        info = {'npz_file': npz_file, 'files': ','.join(files)}
        for key in files:
            D = data[key]
            info[f'{key}.shape'] = D.shape; info[f'{key}.n_obs'] = D.shape[0]
            del D
        return info

    def run(self) -> None:
        file_info = []
        with ThreadPoolExecutor(max_workers = 12) as executor:
            future_to_file_info = {executor.submit(self._get_npz_file_info, npz_file): npz_file for npz_file in self.npz_files}
            for future in as_completed(future_to_file_info):
                npz_file = future_to_file_info[future]
                try:
                    ret = future.result()
                    file_info.append(ret)
                except Exception as exc:
                    print(f'Error processing {npz_file}: {exc}')
        data = pd.DataFrame(file_info)
        import pdb; pdb.set_trace()
        print(data)


if __name__ == '__main__':
    dv = DataValidation()
    dv.run()
