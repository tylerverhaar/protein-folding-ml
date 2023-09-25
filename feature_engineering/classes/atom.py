import numpy as np
from typing import Optional
from mendeleev import element
from feature_engineering.constants import pdb_atom_map, element_info_map, RESIDUE_ATOM_ID_MAPPING, ID_VDW_MAPPING, STRING_CHAR_RESIDUE_MAP, PDB_ATOM_SPECIAL_CASES



class Atom:

    def __init__(self, name: str, number: int, x_coord: float, y_coord: float, z_coord: float, residue_type: str) -> None:
        """
        Initialize an Atom object with the given attributes.

        Args:
            name (str): The name of the atom.
            number (int): The number of the atom.
            x_coord (float): The x-coordinate of the atom.
            y_coord (float): The y-coordinate of the atom.
            z_coord (float): The z-coordinate of the atom.
        """
        self.name = name
        self.pdb_element_name = name.strip('1234567890')

        if self.pdb_element_name in PDB_ATOM_SPECIAL_CASES:
            self.element_name = self.pdb_element_name[0]
        else:
            print(f'non reductive element_name = {self.pdb_element_name}')
            self.element_name = self.pdb_element_name
        
        self.number = number
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.z_coord = z_coord
        self.p = np.array([x_coord, y_coord, z_coord])
        self.w = 1
        self.residue_type = residue_type

    def __str__(self) -> str:
        """
        Return a string representation of the Atom object.

        Returns:
            str: A string representation of the Atom object.
        """
        atom_string = f'{self.name}_{self.number}_(w = {self.w})_(x = {self.x_coord}, y = {self.y_coord}, z = {self.z_coord})'
        return atom_string


    def _set_atomic_radius(self, units: Optional[str] = 'A') -> None:
        N = 100 if units == 'A' else 1
        self.atomic_radius = element_info_map[self.element_name]['atomic_radius']
        self.atomic_radius = 0 if self.atomic_radius is None else self.atomic_radius / N

    def _set_atomic_mass(self) -> None:
        self.atomic_mass = element_info_map[self.element_name]['atomic_mass']
        self.atomic_mass = 0 if self.atomic_mass is None else self.atomic_mass

    def _set_van_der_walls_radius(self, units: Optional[str] = 'A') -> None:
        N = 100 if units == 'A' else 1
        res_char = STRING_CHAR_RESIDUE_MAP[self.residue_type]
        if self.name in RESIDUE_ATOM_ID_MAPPING[res_char]:
            self.vdw_radius = ID_VDW_MAPPING[RESIDUE_ATOM_ID_MAPPING[res_char][self.name]]['radius']
            self.potential = ID_VDW_MAPPING[RESIDUE_ATOM_ID_MAPPING[res_char][self.name]]['welldepth']
        else:
            self.vdw_radius = element_info_map[self.element_name]['vdw_radius']
            self.vdw_radius = 0 if self.vdw_radius is None else self.vdw_radius / N

    def _set_covalent_radius(self, units: Optional[str] = 'A') -> None:
        N = 100 if units == 'A' else 1
        self.covalent_radius = element_info_map[self.element_name]['covalent_radius_bragg']
        self.covalent_radius = 0 if self.covalent_radius is None else self.covalent_radius / N

    def set_atomic_attributes(self) -> None:
        self._set_atomic_mass()
        #if self.name in ['CA', 'N', 'O', 'C']:
        self._set_atomic_radius()
        self._set_van_der_walls_radius()
        self._set_covalent_radius()

    def get_weight(self, weight: str) -> float:
        return 1 if weight == 'uniform' else self.atomic_mass

            
    def copy_atomic_attributes(self, atom: object) -> None:
        #self.name = atom.name
        self.atomic_radius = atom.atomic_radius
        self.atomic_mass = atom.atomic_mass
        self.covalent_radius = atom.covalent_radius
        self.vdw_radius = atom.vdw_radius
        #self.potential = atom.potential
