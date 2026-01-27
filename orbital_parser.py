import re
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional
import os
import glob
import sys


class OrbitalGAMESSParser:
    
    def __init__(self, distance_cutoff: float = 4.0, debug: bool = False, min_atoms: int = 2,
                 include_orbital_type: bool = True, include_m_quantum: bool = True,
                 global_target_type: str = 'mcscf_energy'):
        self.distance_cutoff = distance_cutoff
        self.debug = debug
        self.min_atoms = min_atoms
        self.include_orbital_type = include_orbital_type
        self.include_m_quantum = include_m_quantum
        self.global_target_type = global_target_type
        elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 
                    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca']
        self.atomic_numbers = {elem.upper(): i+1 for i, elem in enumerate(elements)}
        
    def parse_gamess_file(self, filepath: str) -> Dict:
        """Parse GAMESS .out file and extract orbital-level data"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        data = {'coordinates': None, 'atoms': None, 'orbital_data': None,
                'density_matrix': None, 'kei_bo_matrix': None,
                'final_mcscf_energy': None, 'total_kinetic_energy': None,
                'filename': filepath}
        
        # Extract atomic coordinates and symbols
        coords, atoms = self._extract_coordinates(content)
        if coords is None or atoms is None:
            raise ValueError(f"Could not extract coordinates from {filepath}")
        if len(atoms) == 1:
            raise ValueError(f"Single atom system found: {atoms[0]} (excluded)")
        if len(atoms) < self.min_atoms:
            raise ValueError(f"System too small: {len(atoms)} atoms (minimum required: {self.min_atoms})")
        
        data.update({'coordinates': coords, 'atoms': atoms})
        
        # Extract orbital-level data
        extractors = [
            ('density_matrix', self._extract_density_matrix, (content,), "density matrix"),
            ('kei_bo_matrix', self._extract_kei_bo_matrix_orbital, (content, atoms), "KEI-BO orbital data"),
            ('final_mcscf_energy', self._extract_mcscf_energy, (content,), "FINAL MCSCF ENERGY"),
            ('total_kinetic_energy', self._extract_total_kinetic_energy, (content,), "TOTAL KINETIC ENERGY"),
            ('orbital_characters', self._extract_orbital_characters, (content,), "orbital characters")
        ]
        
        for key, func, args, desc in extractors:
            result = func(*args)
            if result is None and key not in ['orbital_characters', 'total_kinetic_energy']:
                raise ValueError(f"Could not extract {desc} from {filepath}")
            data[key] = result
        
        # Create orbital-level data structure
        data['orbital_data'] = self._create_orbital_data(atoms, data['density_matrix'], data.get('orbital_characters'))
        
        return data
    
    def _extract_total_kinetic_energy(self, content: str) -> Optional[float]:
        """Extract the TOTAL KINETIC ENERGY from GAMESS output"""
        match = re.search(r"TOTAL KINETIC ENERGY\s+=\s+([-+]?\d+\.\d+)", content)
        if match:
            return float(match.group(1))
        return None
    
    def _extract_coordinates(self, content: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Extract atomic coordinates and symbols (always in BOHR units)"""
        pattern = r'ATOM\s+ATOMIC\s+COORDINATES \(BOHR\)\s*\n\s*CHARGE\s+X\s+Y\s+Z\s*\n(.*?)(?=\n\s*INTERNUCLEAR|\n\s*\n|\Z)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        return self._parse_coordinate_block(match.group(1)) if match else (None, None)
    
    def _parse_coordinate_block(self, coord_text: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Parse coordinate block text and convert from Bohr to Angstrom"""
        lines = [line for line in coord_text.strip().split('\n') 
                 if line.strip() and not any(skip in line for skip in ['ATOM', '----', 'CHARGE', 'COORDINATES'])]
        
        coords, atoms = [], []
        bohr_to_ang = 0.529177
        
        for line in lines:
            parts = line.split()
            atom, x, y, z = parts[0], float(parts[2]), float(parts[3]), float(parts[4])
            atoms.append(atom)
            coords.append([x * bohr_to_ang, y * bohr_to_ang, z * bohr_to_ang])
        
        return (np.array(coords), atoms) if len(coords) >= self.min_atoms else (None, None)

    def _extract_density_matrix(self, content: str) -> Optional[np.ndarray]:
        """Extract the density matrix (trying multiple common GAMESS headers)"""
        # List of headers to look for, in order of preference
        headers = [
            # The header found in your successful MCSCF files
            r'ORIGINAL ORIENTED DENSITY MATRIX',
            # The header found in o-methane.log (Line 6378) - likely the most relevant for bonding
            r'DENSITY MATRIX FOR QUASI-ATOMIC AND BONDING ORBITALS',
            # Alternative from o-methane.log (Line 6266)
            r'DENSITY MATRIX IN THE ORTHOGONAL QUASI-ATOMIC SVD MO BASIS',
            # Standard generic fallback
            r'TOTAL DENSITY MATRIX'
        ]
        
        for header in headers:
            # Construct regex: Header -> capture content -> stop at "MULTIPLY" or double newline
            pattern = f'{header}\\s+(.*?)(?=\\n\\s*MULTIPLY|\\n\\s*\\n\\s*[A-Z]|$)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                if self.debug:
                    print(f"  Found density matrix with header: '{header}'")
                return self._parse_matrix(match.group(1))

        return None

    def _parse_matrix(self, matrix_text: str) -> Optional[np.ndarray]:
        """Parse matrix from GAMESS output format"""
        lines = [line.strip() for line in matrix_text.strip().split('\n') if line.strip()]
        
        max_index = 0
        for line in lines:
            row_match = re.match(r'^\s*(\d+)', line)
            if row_match:
                max_index = max(max_index, int(row_match.group(1)))
            if re.match(r'^\s*\d+(?:\s+\d+)*\s*$', line):
                indices = [int(x) for x in line.split()]
                max_index = max(max_index, max(indices))
        
        if max_index == 0:
            return None
        
        matrix = np.zeros((max_index, max_index))
        i = 0
        
        while i < len(lines):
            line = lines[i]
            if re.match(r'^\s*\d+(?:\s+\d+)*\s*$', line):
                col_indices = [int(x) - 1 for x in line.split()]
                i += 1
                
                while i < len(lines) and not re.match(r'^\s*\d+(?:\s+\d+)*\s*$', lines[i]):
                    row_match = re.match(r'^\s*(\d+)\s+(.*)', lines[i])
                    if row_match:
                        try:
                            row_idx = int(row_match.group(1)) - 1
                            values = [float(x) for x in row_match.group(2).split()]
                            for j, val in enumerate(values):
                                if j < len(col_indices):
                                    col_idx = col_indices[j]
                                    matrix[row_idx][col_idx] = matrix[col_idx][row_idx] = val
                        except ValueError:
                            pass
                    i += 1
            else:
                i += 1
        
        return matrix

    def _extract_orbital_characters(self, content: str) -> Optional[Dict[int, Dict[str, float]]]:
        """Extract orbital character percentages"""
        pattern = r'PRINT OFF ORIENTATION INFORMATION FOR VALENCE ORBITAL CHARACTER PERCENT\.\s*ORB I\s+PERCENT S\s+PERCENT P\s+PERCENT D\s+PERCENT F\s*(.*?)(?=\n\s*END OF CURRENT|\n\s*[A-Z]|$)'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            return None
        
        orbital_chars = {}
        lines = match.group(1).strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                try:
                    orb_idx = int(parts[0])
                    s_percent = float(parts[1])
                    p_percent = float(parts[2])
                    d_percent = float(parts[3])
                    f_percent = float(parts[4])
                    
                    orbital_chars[orb_idx] = {
                        'S': s_percent,
                        'P': p_percent,
                        'D': d_percent,
                        'F': f_percent
                    }
                except ValueError:
                    continue
        
        return orbital_chars if orbital_chars else None

    def _extract_kei_bo_matrix_orbital(self, content: str, atoms: List[str]) -> Optional[np.ndarray]:
        """Extract KEI-BO matrix from the novel density matrix (kinetic energy weighted)"""
        # Get the novel density matrix (kinetic energy weighted)
        novel_density = self._extract_novel_density_matrix(content)
        
        if novel_density is None:
            return None
        
        if self.debug:
            print(f"Novel density matrix shape: {novel_density.shape}")
            print(f"Novel density matrix (first 5x5):")
            size = min(5, novel_density.shape[0])
            for i in range(size):
                row_str = " ".join([f"{novel_density[i,j]:8.4f}" for j in range(size)])
                print(f"  {row_str}")
        
        # Use novel density matrix as KEI-BO matrix (off-diagonal elements)
        kei_bo_matrix = novel_density.copy()
        
        # Zero out diagonal elements since we only want orbital-orbital interactions
        np.fill_diagonal(kei_bo_matrix, 0.0)
        
        return kei_bo_matrix
    
    def _extract_novel_density_matrix(self, content: str) -> Optional[np.ndarray]:
        """Extract the novel density matrix (kinetic energy weighted)"""
        pattern = r'FULL PRINT OUT OF NOVEL ORIENTED DENSITY\s+(.*?)(?=\n\s*OLD TOTAL KINETIC|\n\s*\n\s*[A-Z]|$)'
        match = re.search(pattern, content, re.DOTALL)
        return self._parse_matrix(match.group(1)) if match else None

    def _create_orbital_data(self, atoms: List[str], density_matrix: np.ndarray, 
                           orbital_chars: Optional[Dict[int, Dict[str, float]]]) -> Dict:
        """Create orbital-level data structure using 1s/4-orbital rule
        
        Features are controlled by flags:
        - Always includes: atomic_num
        - Optional: orbital_type (S=0, P=1) if include_orbital_type=True
        - Optional: m_quantum (-1,0,1) if include_m_quantum=True
        - NEVER includes occupation (that's the prediction target)
        """
        orbital_data = {
            'orbital_positions': [],
            'orbital_features': [],  # 1-3 features: [atomic_num] or [atomic_num, orbital_type] or [atomic_num, orbital_type, m_quantum]
            'orbital_occupations': [],
            'orbital_hybridization': [],  # NEW: s%, p%, d%, f% as targets
            'parent_atoms': [],
            'orbital_types': [],
            'orbital_m_quantum': []
        }
        
        orbital_idx = 0
        n_atoms = len(atoms)
        
        if self.debug:
            print(f"Creating orbital data for {n_atoms} atoms, density matrix size: {density_matrix.shape}")
        
        for atom_idx, atom in enumerate(atoms):
            atom_upper = atom.upper()
            atomic_number = self.atomic_numbers.get(atom_upper, 1)
            
            # Use your 1s/4-orbital rule
            if atom_upper in ['H', 'HE']:
                n_orbitals = 1
                orbital_types = ['S']
                m_quantums = [0]
            else:
                n_orbitals = 4  # 2s, 2px, 2py, 2pz (or equivalent)
                orbital_types = ['S', 'P', 'P', 'P']
                m_quantums = [0, -1, 0, 1]
            
            for i in range(n_orbitals):
                if orbital_idx >= density_matrix.shape[0]:
                    # Pad with zeros if density matrix is smaller
                    occupation = 0.0
                else:
                    occupation = density_matrix[orbital_idx, orbital_idx]
                
                # Store orbital information
                orbital_data['orbital_positions'].append(atom_idx)
                orbital_data['parent_atoms'].append(atomic_number)
                orbital_data['orbital_types'].append(0 if orbital_types[i] == 'S' else 1)  # S=0, P=1
                orbital_data['orbital_m_quantum'].append(m_quantums[i])
                orbital_data['orbital_occupations'].append(occupation)
                
                # INPUT FEATURES: Configurable 1-3 features (NEVER includes occupation)
                features = [atomic_number]  # Always include atomic number
                
                if self.include_orbital_type:
                    features.append(0 if orbital_types[i] == 'S' else 1)  # S=0, P=1
                
                if self.include_m_quantum:
                    features.append(m_quantums[i])  # -1, 0, 1
                
                orbital_data['orbital_features'].append(features)
                
                # TARGET: Hybridization percentages [s%, p%, d%, f%]
                if orbital_chars and orbital_idx + 1 in orbital_chars:
                    chars = orbital_chars[orbital_idx + 1]
                    hybridization = [chars['S'], chars['P'], chars['D'], chars['F']]
                else:
                    # Default hybridization based on orbital type
                    hybridization = [
                        1.0 if orbital_types[i] == 'S' else 0.0,  # S character
                        1.0 if orbital_types[i] == 'P' else 0.0,  # P character
                        0.0,  # D character
                        0.0   # F character
                    ]
                orbital_data['orbital_hybridization'].append(hybridization)
                
                orbital_idx += 1
                
                if self.debug:
                    print(f"  Orbital {orbital_idx}: {atom}{atom_idx+1}-{orbital_types[i]}{m_quantums[i]}, occ={occupation:.4f}")
        
        return orbital_data

    def convert_to_pyg(self, data: Dict) -> Data:
        """Convert parsed data to PyTorch Geometric format with orbitals as nodes"""
        coords = data['coordinates']
        atoms = data['atoms']
        orbital_data = data['orbital_data']
        kei_bo_matrix = data['kei_bo_matrix']
        final_mcscf_energy = data['final_mcscf_energy']
        total_kinetic_energy = data['total_kinetic_energy']
        
        num_orbitals = len(orbital_data['orbital_features'])
        
        # Node features: 1-3 input features [atomic_num] or [atomic_num, orbital_type] or [atomic_num, orbital_type, m_quantum]
        # Occupation is NEVER an input - it's the prediction target
        node_features = torch.tensor(orbital_data['orbital_features'], dtype=torch.float)
        
        # Node targets: orbital occupations from density matrix diagonal
        node_targets = torch.tensor(orbital_data['orbital_occupations'], dtype=torch.float)
        
        # Hybridization targets: [s%, p%, d%, f%] for each orbital
        hybrid_targets = torch.tensor(orbital_data['orbital_hybridization'], dtype=torch.float)
        
        # Select global target
        if self.global_target_type == 'kinetic_energy':
            if total_kinetic_energy is None:
                raise ValueError(f"TOTAL KINETIC ENERGY not found in file, but global_target_type='kinetic_energy'")
            global_target = torch.tensor([total_kinetic_energy], dtype=torch.float)
        else:
            global_target = torch.tensor([final_mcscf_energy], dtype=torch.float)
        
        # Create orbital positions for distance calculations
        orbital_positions = []
        for orb_idx in range(num_orbitals):
            parent_atom_idx = orbital_data['orbital_positions'][orb_idx]
            orbital_positions.append(coords[parent_atom_idx])
        orbital_positions = np.array(orbital_positions)
        
        # Create edges between orbitals based on distance cutoff and KEI-BO values
        distances = cdist(orbital_positions, orbital_positions)
        
        edges = []
        edge_features = []
        edge_targets = []
        
        for i in range(num_orbitals):
            for j in range(i + 1, num_orbitals):
                # Include edge if:
                # 1. Distance between parent atoms is within cutoff
                # 2. KEI-BO value is non-zero (above threshold)
                if (distances[i, j] <= self.distance_cutoff and 
                    abs(kei_bo_matrix[i, j]) > 1e-8):
                    
                    edges.append([i, j])
                    edges.append([j, i])  # Both directions
                    
                    edge_features.extend([distances[i, j], distances[i, j]])
                    edge_targets.extend([kei_bo_matrix[i, j], kei_bo_matrix[i, j]])
        
        if not edges:
            # If no edges found, create minimal connectivity
            edges = [[0, 1], [1, 0]] if num_orbitals > 1 else [[0, 0]]
            edge_features = [distances[0, 1], distances[0, 1]] if num_orbitals > 1 else [0.0]
            edge_targets = [0.0, 0.0] if num_orbitals > 1 else [0.0]
        
        return Data(
            x=node_features,
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_features, dtype=torch.float).view(-1, 1),
            y=node_targets,
            edge_y=torch.tensor(edge_targets, dtype=torch.float),
            global_y=global_target,
            hybrid_y=hybrid_targets,  # NEW: Hybridization targets [N_orbitals x 4]
            pos=torch.tensor(orbital_positions, dtype=torch.float),
            num_nodes=num_orbitals
        )
    
    def parse_and_convert(self, filepath: str) -> Data:
        """Parse GAMESS file and convert to PyTorch Geometric format"""
        return self.convert_to_pyg(self.parse_gamess_file(filepath))
    
    def process_multiple_files(self, filepaths: List[str]) -> List[Data]:
        """Process multiple GAMESS files"""
        return [self.parse_and_convert(filepath) for filepath in filepaths]


def get_all_files_per_folder(base_path: str) -> Dict[str, List[str]]:
    """Get all .log files organized by folder"""
    if not os.path.exists(base_path):
        raise ValueError(f"Base path {base_path} does not exist")
    
    # Check for direct .log files
    direct_files = glob.glob(os.path.join(base_path, "*.log"))
    if direct_files:
        folder_name = os.path.basename(base_path) or "root"
        print(f"Found {len(direct_files)} .log files directly in {base_path}")
        return {folder_name: direct_files}
    
    # Look in child directories
    child_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not child_dirs:
        raise ValueError(f"No .log files found directly in {base_path} and no child directories found")
    
    folder_files = {}
    print(f"Scanning {len(child_dirs)} folders in {base_path}:")
    
    for folder_name in child_dirs:
        log_files = glob.glob(os.path.join(base_path, folder_name, "**/*.log"), recursive=True)
        if log_files:
            folder_files[folder_name] = log_files
        print(f"  {folder_name}: {len(log_files)} .log files")
    
    return folder_files


def analyze_orbital_files(filepaths: List[str]) -> None:
    """Analyze GAMESS files and print orbital-level data"""
    parser = OrbitalGAMESSParser(debug=True)
    successful, failed_by_type = [], {}
    
    print(f"Analyzing {len(filepaths)} GAMESS files for orbital data...\n")
    
    for i, filepath in enumerate(filepaths, 1):
        filename = os.path.basename(filepath)
        print(f"=== {i}/{len(filepaths)}: {filename} ===")
        
        try:
            data = parser.parse_gamess_file(filepath)
            successful.append(filepath)
            atoms, coords = data['atoms'], data['coordinates']
            orbital_data = data['orbital_data']
            kei_bo_matrix = data['kei_bo_matrix']
            final_energy = data['final_mcscf_energy']
            print(f"Atoms ({len(atoms)}):")
            for j, (atom, coord) in enumerate(zip(atoms, coords)):
                print(f"  {j+1:2d} {atom:2s}: [{coord[0]:7.3f}, {coord[1]:7.3f}, {coord[2]:7.3f}] Å")
            print(f"\nOrbitals ({len(orbital_data['orbital_features'])}):")
            for j, features in enumerate(orbital_data['orbital_features']):
                parent_atom_idx = orbital_data['orbital_positions'][j]
                atom_symbol = atoms[parent_atom_idx]
                orbital_type_val = orbital_data['orbital_types'][j]
                orbital_type = ['S', 'P', 'D', 'F'][orbital_type_val]
                m_quantum = orbital_data['orbital_m_quantum'][j]
                occupation = orbital_data['orbital_occupations'][j]
                print(f"  {j+1:2d} {atom_symbol}{parent_atom_idx+1}-{orbital_type}{m_quantum}: occ={occupation:.4f}")
            print(f"\nSignificant KEI-BO orbital interactions:")
            count = 0
            for i in range(len(orbital_data['orbital_features'])):
                for j in range(i+1, len(orbital_data['orbital_features'])):
                    if abs(kei_bo_matrix[i, j]) > 1e-6:
                        print(f"  Orb{i+1}-Orb{j+1}: {kei_bo_matrix[i, j]:.6f}")
                        count += 1
                        if count >= 10:  # Limit output
                            break
                if count >= 10:
                    break
            print(f"MCSCF Energy: {final_energy:.8f} hartree\n")
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            failed_by_type.setdefault(error_type, []).append((filename, error_msg))
            print(f"FAILED: {error_type} - {error_msg}\n")
    
    # Summary
    total_failed = sum(len(files) for files in failed_by_type.values())
    if len(filepaths) > 0:
        success_rate = 100 * len(successful) / len(filepaths)
        print(f"=== ORBITAL ANALYSIS SUMMARY ===")
        print(f"Successfully parsed: {len(successful)}/{len(filepaths)} files ({success_rate:.1f}%)")
    
    if failed_by_type:
        print(f"Failed: {total_failed} files")
        for error_type, file_list in failed_by_type.items():
            print(f"  {error_type}: {len(file_list)} files")
            for fname, emsg in file_list:
                print(f"    - {fname}: {emsg}")


def main():
    """Main function for orbital parser testing"""
    if len(sys.argv) != 2:
        examples = ["python orbital_parser.py .", "python orbital_parser.py data/beb"]
        print(f"Usage: python orbital_parser.py <directory_path>\nExamples:\n" + '\n'.join(f"  {ex}" for ex in examples))
        sys.exit(1)
    
    try:
        folder_files = get_all_files_per_folder(sys.argv[1])
        if not folder_files:
            print("No folders with .log files found")
            return
        
        all_files = [file for files in folder_files.values() for file in files]
        if not all_files:
            print("No .log files found in subdirectories.")
            return
            
        print(f"\nFound {len(all_files)} total .log files across {len(folder_files)} folders")
        analyze_orbital_files(all_files)  # Analyze first 5 files for testing
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()