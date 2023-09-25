from typing import Dict, Any
from feature_engineering.classes.residue import Residue


class DecoyLoop:

    def __init__(self, decoy_id: int, energy_evaluation: float, residues: Dict[int, Dict[str, Any]]) -> None:
        self.decoy_id = decoy_id
        self.energy_evaluation = energy_evaluation
        self.residue_dict = residues
        self.invalid_loop = False
        self.set_loop()

    def set_loop(self):
        self.loop = []
        try:
            for res_id, res_data in self.residue_dict.items():
                id = res_id
                residue_type = res_data['residue_name']
                chain_id = res_data['chain_id']
                residue = Residue(id = id, residue_type = residue_type, chain_id = chain_id, raw_atoms = res_data['atoms'], in_loop = True)
                self.loop.append(residue)
            self.n_residues = len(self.loop)
        except Exception as e:
            print('invalid loop')
            print(e)
            self.invalid_loop = True
