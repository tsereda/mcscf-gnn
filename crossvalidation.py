import torch
from torch_geometric.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import glob
import os
import json
from datetime import datetime

from orbital_gnn import OrbitalTripleTaskGNN, compute_metrics
from orbital_parser import OrbitalGAMESSParser
from orbital_trainer import process_orbital_files
from visualization import create_summary_plots, create_comprehensive_analysis


import torch
from torch_geometric.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import glob
import os
import json
from datetime import datetime

from orbital_gnn import OrbitalTripleTaskGNN, compute_metrics
from orbital_parser import OrbitalGAMESSParser
from orbital_trainer import process_orbital_files
from visualization import create_summary_plots, create_comprehensive_analysis

def get_all_files_per_folder(base_path: str) -> Dict[str, List[str]]:
    """
    Get ALL .log files from each child folder of base_path.
    
    Returns:
        Dictionary mapping folder_name -> list_of_all_log_files
    """
    if not os.path.exists(base_path):
        raise ValueError(f"Base path {base_path} does not exist")
    
    folder_files = {}
    
    # Get all immediate child directories
    child_dirs = [d for d in os.listdir(base_path) 
                  if os.path.isdir(os.path.join(base_path, d))]
    
    if not child_dirs:
        raise ValueError(f"No child directories found in {base_path}")
    
    print(f"Scanning {len(child_dirs)} folders in {base_path}:")
    
    for folder_name in child_dirs:
        folder_path = os.path.join(base_path, folder_name)
        
        # Find all .log files in this folder (including subdirectories)
        log_files = glob.glob(os.path.join(folder_path, "**/*.log"), recursive=True)
        
        if log_files:
            folder_files[folder_name] = log_files
            print(f"  📂 {folder_name}: {len(log_files)} .log files")
        else:
            print(f"  📂 {folder_name}: No .log files found")
    
    return folder_files


def create_file_to_folder_mapping(folder_files: Dict[str, List[str]]) -> Dict[str, str]:
    """Create mapping from file path to folder name."""
    file_to_folder = {}
    for folder_name, files in folder_files.items():
        for filepath in files:
            file_to_folder[filepath] = folder_name
    return file_to_folder


def check_orbital_contains_element(filepath: str, target_element: str) -> bool:
    """Check if a molecule file contains a specific element (orbital-level check)."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Quick search for the element in atomic coordinates section
        import re
        pattern = r'ATOM\s+ATOMIC\s+COORDINATES \(BOHR\)\s*\n\s*CHARGE\s+X\s+Y\s+Z\s*\n(.*?)(?=\n\s*INTERNUCLEAR|\n\s*\n|\Z)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            coord_text = match.group(1)
            lines = coord_text.strip().split('\n')
            
            for line in lines:
                if not line.strip() or 'ATOM' in line or '----' in line or 'CHARGE' in line or 'COORDINATES' in line:
                    continue
                
                parts = line.split()
                if len(parts) >= 4:  # atom, charge/coord data format
                    atom_symbol = parts[0]
                    if atom_symbol.upper() == target_element.upper():
                        return True
                elif len(parts) >= 5:  # atom, charge, x, y, z format  
                    atom_symbol = parts[0]
                    if atom_symbol.upper() == target_element.upper():
                        return True
        
        return False
    except Exception:
        return False


def get_all_elements_in_dataset(all_files: List[str]) -> List[str]:
    """Find all unique elements present in the dataset."""
    elements = set()
    
    print("Scanning dataset for unique elements...")
    
    for filepath in all_files:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Extract elements from atomic coordinates section
            import re
            pattern = r'ATOM\s+ATOMIC\s+COORDINATES \(BOHR\)\s*\n\s*CHARGE\s+X\s+Y\s+Z\s*\n(.*?)(?=\n\s*INTERNUCLEAR|\n\s*\n|\Z)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            
            if match:
                coord_text = match.group(1)
                lines = coord_text.strip().split('\n')
                
                for line in lines:
                    if not line.strip() or 'ATOM' in line or '----' in line or 'CHARGE' in line or 'COORDINATES' in line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 4:
                        atom_symbol = parts[0].strip()
                        # Clean up element symbol (remove numbers, special chars)
                        element = ''.join(c for c in atom_symbol if c.isalpha())
                        if element:
                            elements.add(element.upper())
        except Exception:
            continue
    
    unique_elements = sorted(list(elements))
    print(f"Found elements: {unique_elements}")
    return unique_elements


def split_files_by_element_with_folders(all_files: List[str], file_to_folder: Dict[str, str], 
                                       target_element: str = 'H') -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split files into those containing and not containing a target element, tracking folders.
    
    Args:
        all_files: List of all file paths
        file_to_folder: Mapping from file path to folder name
        target_element: Element to split on (default: 'H' for hydrogen)
        
    Returns:
        (files_without_element, files_with_element, folders_without_element, folders_with_element)
    """
    files_with_element = []
    files_without_element = []
    folders_with_element = set()
    folders_without_element = set()
    
    print(f"Splitting {len(all_files)} files by presence of element '{target_element}'...")
    
    for filepath in all_files:
        folder_name = file_to_folder.get(filepath, "unknown")
        
        if check_orbital_contains_element(filepath, target_element):
            files_with_element.append(filepath)
            folders_with_element.add(folder_name)
        else:
            files_without_element.append(filepath)
            folders_without_element.add(folder_name)
    
    folders_without_element_list = sorted(list(folders_without_element))
    folders_with_element_list = sorted(list(folders_with_element))
    
    print(f"Files without {target_element}: {len(files_without_element)} from folders: {folders_without_element_list}")
    print(f"Files with {target_element}: {len(files_with_element)} from folders: {folders_with_element_list}")
    
    return files_without_element, files_with_element, folders_without_element_list, folders_with_element_list


def prepare_random_split_data(all_files: List[str],
                              split_ratio: float = 0.2,
                              random_seed: int = 42,
                              batch_size: int = 16,
                              include_orbital_type: bool = True,
                              include_m_quantum: bool = True,
                              global_target_type: str = None) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Prepare data with random train/validation split.
    
    Args:
        all_files: List of all file paths
        split_ratio: Fraction of data to use for validation (default: 0.2 for 80/20 split)
        random_seed: Random seed for reproducibility
        batch_size: Batch size for DataLoaders
    
    Returns:
        (train_loader, val_loader, fold_info)
    """
    print(f"\n{'='*70}")
    print(f"RANDOM SPLIT VALIDATION")
    print(f"{'='*70}")
    print(f"Split ratio: {split_ratio:.1%} validation, {(1-split_ratio):.1%} training")
    print(f"Random seed: {random_seed}")
    print(f"Total files: {len(all_files)}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle and split files
    shuffled_files = all_files.copy()
    np.random.shuffle(shuffled_files)
    
    split_idx = int(len(shuffled_files) * (1 - split_ratio))
    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]
    
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    # Process files with orbital parser
    parser = OrbitalGAMESSParser(
        distance_cutoff=4.0,
        debug=False,
        include_orbital_type=include_orbital_type,
        include_m_quantum=include_m_quantum,
        global_target_type=global_target_type
    )
    
    print(f"Processing training files...")
    train_graphs = process_orbital_files(parser, train_files)
    
    print(f"Processing validation files...")
    val_graphs = process_orbital_files(parser, val_files)
    
    if len(train_graphs) == 0:
        raise ValueError("No valid training graphs for random split")
    if len(val_graphs) == 0:
        print(f"    Warning: No valid validation graphs, using copy of first training graph")
        val_graphs = [train_graphs[0]]
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    
    # Fold information
    fold_info = {
        'validation_folder': f'random_split_{split_ratio}',
        'training_folders': ['random_split_train'],
        'train_graphs': len(train_graphs),
        'val_graphs': len(val_graphs),
        'train_files': len(train_files),
        'val_files': len(val_files),
        'split_type': 'random_split',
        'split_ratio': split_ratio,
        'random_seed': random_seed,
        'fold_num': 1
    }
    
    return train_loader, val_loader, fold_info


def prepare_element_based_fold_data(all_files: List[str], 
                                  file_to_folder: Dict[str, str],
                                  fold_num: int,
                                  available_elements: List[str],
                                  batch_size: int = 16,
                                  include_orbital_type: bool = True,
                                  include_m_quantum: bool = True) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Prepare orbital data for element-based cross-validation.
    Each fold validates on molecules containing a specific element.
    
    Args:
        all_files: All available files
        file_to_folder: Mapping from file path to folder name
        fold_num: Fold number (0 to len(available_elements)-1)
        available_elements: List of all elements found in dataset
        batch_size: Batch size for DataLoaders
        
    Returns:
        (train_loader, val_loader, fold_info)
    """
    if fold_num >= len(available_elements):
        raise ValueError(f"Fold number {fold_num} exceeds available elements {len(available_elements)}")
    
    target_element = available_elements[fold_num]
    files_without_element, files_with_element, folders_without_element, folders_with_element = split_files_by_element_with_folders(
        all_files, file_to_folder, target_element
    )
    
    # Train on molecules without target element, validate on those with it
    train_files = files_without_element
    val_files = files_with_element
    validation_description = f"with_{target_element}"
    training_description = f"without_{target_element}"
    
    print(f"Training: {training_description} ({len(train_files)} files)")
    print(f"Validation: {validation_description} ({len(val_files)} files)")
    
    # Process files with orbital parser
    parser = OrbitalGAMESSParser(
        distance_cutoff=4.0, 
        debug=False,
        include_orbital_type=include_orbital_type,
        include_m_quantum=include_m_quantum
    )
    
    print(f"Processing training files...")
    train_graphs = process_orbital_files(parser, train_files)
    
    print(f"Processing validation files...")
    val_graphs = process_orbital_files(parser, val_files)
    
    if len(train_graphs) == 0:
        raise ValueError(f"No valid training graphs for element-based fold {fold_num} (element: {target_element})")
    if len(val_graphs) == 0:
        print(f"    Warning: No valid validation graphs for element {target_element}, using copy of first training graph")
        val_graphs = [train_graphs[0]]
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    
    # Fold information with actual folder names
    fold_info = {
        'validation_folder': validation_description,
        'training_folders': [training_description],
        'validation_actual_folders': folders_with_element,
        'training_actual_folders': folders_without_element,
        'train_graphs': len(train_graphs),
        'val_graphs': len(val_graphs),
        'train_files': len(train_files),
        'val_files': len(val_files),
        'split_type': 'element_based',
        'target_element': target_element,
        'fold_num': fold_num
    }
    
    return train_loader, val_loader, fold_info


def prepare_single_fold_data(folder_files: Dict[str, List[str]], 
                             validation_folder: str,
                             batch_size: int = 16,
                             include_orbital_type: bool = True,
                             include_m_quantum: bool = True) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Prepare orbital data for a single fold with specified validation folder.
    
    Args:
        folder_files: Dictionary mapping folder_name -> list_of_files
        validation_folder: Name of folder to use for validation
        batch_size: Batch size for DataLoaders
    
    Returns:
        (train_loader, val_loader, fold_info)
    """
    # Split files into train and validation
    val_files = folder_files[validation_folder]
    train_files = []
    train_folders = []
    
    for folder_name, files in folder_files.items():
        if folder_name != validation_folder:
            train_files.extend(files)
            train_folders.append(folder_name)
    
    print(f"Training: {len(train_folders)} folders, {len(train_files)} files")
    print(f"Validation: 1 folder ({validation_folder}), {len(val_files)} files")
    
    # Process files with orbital parser
    parser = OrbitalGAMESSParser(
        distance_cutoff=4.0, 
        debug=False,
        include_orbital_type=include_orbital_type,
        include_m_quantum=include_m_quantum
    )
    
    print(f"Processing training files...")
    train_graphs = process_orbital_files(parser, train_files)
    
    print(f"Processing validation files...")
    val_graphs = process_orbital_files(parser, val_files)
    
    if len(train_graphs) == 0:
        raise ValueError(f"No valid training graphs for fold with validation folder: {validation_folder}")
    if len(val_graphs) == 0:
        print(f"    Warning: No valid validation graphs, using copy of first training graph")
        val_graphs = [train_graphs[0]]
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    
    # Fold information
    fold_info = {
        'validation_folder': validation_folder,
        'training_folders': train_folders,
        'train_graphs': len(train_graphs),
        'val_graphs': len(val_graphs),
        'train_files': len(train_files),
        'val_files': len(val_files),
        'split_type': 'per_molecule'
    }
    
    return train_loader, val_loader, fold_info


def generate_detailed_orbital_validation_report(model: OrbitalTripleTaskGNN, val_loader: DataLoader, 
                                               fold_info: Dict, device: torch.device,
                                               normalizer=None) -> str:
    """Generate detailed validation report with orbital-by-orbital analysis."""
    model.eval()
    
    # Collect all predictions
    all_occupation_preds, all_occupation_targets = [], []
    all_kei_bo_preds, all_kei_bo_targets = [], []
    all_energy_preds, all_energy_targets = [], []
    all_s_preds, all_s_targets = [], []
    all_p_preds, all_p_targets = [], []
    all_d_preds, all_d_targets = [], []
    all_f_preds, all_f_targets = [], []
    all_molecules = []
    
    with torch.no_grad():
        mol_idx = 1
        for batch in val_loader:
            batch = batch.to(device)
            
            # Store original targets BEFORE normalization
            original_occupation_targets = batch.y.clone()
            original_kei_bo_targets = batch.edge_y.clone()
            original_energy_targets = batch.global_y.clone()
            original_s_targets = batch.hybrid_y[:, 0:1].clone()
            original_p_targets = batch.hybrid_y[:, 1:2].clone()
            original_d_targets = batch.hybrid_y[:, 2:3].clone()
            original_f_targets = batch.hybrid_y[:, 3:4].clone()
            
            # Apply normalization if normalizer provided
            if normalizer:
                normalizer.normalize_batch(batch)
            
            # Get predictions (in normalized space if normalizer is used)
            # Model returns 7 predictions: occupation, kei_bo, energy, s%, p%, d%, f%
            preds = model(batch)
            occupation_pred, kei_bo_pred, energy_pred = preds[0], preds[1], preds[2]
            s_pred, p_pred, d_pred, f_pred = preds[3], preds[4], preds[5], preds[6]
            
            # Denormalize predictions if normalizer was used
            if normalizer:
                denorm_preds = normalizer.denormalize_predictions(*preds)
                occupation_pred, kei_bo_pred, energy_pred = denorm_preds[0], denorm_preds[1], denorm_preds[2]
                s_pred, p_pred, d_pred, f_pred = denorm_preds[3], denorm_preds[4], denorm_preds[5], denorm_preds[6]
            
            # Use original targets for comparison
            batch.y = original_occupation_targets
            batch.edge_y = original_kei_bo_targets
            batch.global_y = original_energy_targets
            
            # Process each molecule in the batch
            batch_size = batch.batch.max().item() + 1
            
            for b in range(batch_size):
                # Find orbitals and edges for this molecule
                orbital_mask = (batch.batch == b)
                mol_orbitals = orbital_mask.sum().item()
                
                # Edge mask - edges where both source and target belong to this molecule
                edge_mask = orbital_mask[batch.edge_index[0]] & orbital_mask[batch.edge_index[1]]
                
                # Extract data for this molecule
                mol_occupation_preds = occupation_pred[orbital_mask].cpu()
                mol_occupation_targets = batch.y[orbital_mask].cpu()
                mol_kei_bo_preds = kei_bo_pred[edge_mask].cpu()
                mol_kei_bo_targets = batch.edge_y[edge_mask].cpu()
                mol_energy_pred = energy_pred[b].cpu()
                mol_energy_target = batch.global_y[b].cpu()
                
                mol_x = batch.x[orbital_mask].cpu()
                mol_edge_attr = batch.edge_attr[edge_mask].cpu()
                mol_edge_index = batch.edge_index[:, edge_mask].cpu()
                
                # Adjust edge indices to be relative to this molecule
                orbital_offset = orbital_mask.nonzero(as_tuple=False)[0, 0].item()
                mol_edge_index = mol_edge_index - orbital_offset
                
                # Extract hybridization predictions for this molecule
                mol_s_preds = s_pred[orbital_mask].cpu()
                mol_s_targets = original_s_targets[orbital_mask].cpu()
                mol_p_preds = p_pred[orbital_mask].cpu()
                mol_p_targets = original_p_targets[orbital_mask].cpu()
                mol_d_preds = d_pred[orbital_mask].cpu()
                mol_d_targets = original_d_targets[orbital_mask].cpu()
                mol_f_preds = f_pred[orbital_mask].cpu()
                mol_f_targets = original_f_targets[orbital_mask].cpu()
                
                mol_data = {
                    'mol_idx': mol_idx,
                    'num_orbitals': mol_orbitals,
                    'num_interactions': len(mol_kei_bo_preds),
                    'orbital_features': mol_x.numpy(),
                    'occupation_preds': mol_occupation_preds.numpy().flatten(),
                    'occupation_targets': mol_occupation_targets.numpy().flatten(),
                    'kei_bo_preds': mol_kei_bo_preds.numpy().flatten(),
                    'kei_bo_targets': mol_kei_bo_targets.numpy().flatten(),
                    'energy_pred': mol_energy_pred.item(),
                    'energy_target': mol_energy_target.item(),
                    'edge_distances': mol_edge_attr.numpy().flatten(),
                    'edge_index': mol_edge_index.numpy(),
                    's_preds': mol_s_preds.numpy().flatten(),
                    's_targets': mol_s_targets.numpy().flatten(),
                    'p_preds': mol_p_preds.numpy().flatten(),
                    'p_targets': mol_p_targets.numpy().flatten(),
                    'd_preds': mol_d_preds.numpy().flatten(),
                    'd_targets': mol_d_targets.numpy().flatten(),
                    'f_preds': mol_f_preds.numpy().flatten(),
                    'f_targets': mol_f_targets.numpy().flatten()
                }
                
                all_molecules.append(mol_data)
                mol_idx += 1
            
            # Collect overall predictions
            all_occupation_preds.append(occupation_pred.cpu())
            all_occupation_targets.append(batch.y.cpu())
            all_kei_bo_preds.append(kei_bo_pred.cpu())
            all_kei_bo_targets.append(batch.edge_y.cpu())
            all_energy_preds.append(energy_pred.cpu())
            all_energy_targets.append(batch.global_y.cpu())
            all_s_preds.append(s_pred.cpu())
            all_s_targets.append(original_s_targets.cpu())
            all_p_preds.append(p_pred.cpu())
            all_p_targets.append(original_p_targets.cpu())
            all_d_preds.append(d_pred.cpu())
            all_d_targets.append(original_d_targets.cpu())
            all_f_preds.append(f_pred.cpu())
            all_f_targets.append(original_f_targets.cpu())
    
    # Calculate overall metrics
    all_occupation_preds = torch.cat(all_occupation_preds)
    all_occupation_targets = torch.cat(all_occupation_targets)
    all_kei_bo_preds = torch.cat(all_kei_bo_preds)
    all_kei_bo_targets = torch.cat(all_kei_bo_targets)
    all_energy_preds = torch.cat(all_energy_preds)
    all_energy_targets = torch.cat(all_energy_targets)
    all_s_preds = torch.cat(all_s_preds)
    all_s_targets = torch.cat(all_s_targets)
    all_p_preds = torch.cat(all_p_preds)
    all_p_targets = torch.cat(all_p_targets)
    all_d_preds = torch.cat(all_d_preds)
    all_d_targets = torch.cat(all_d_targets)
    all_f_preds = torch.cat(all_f_preds)
    all_f_targets = torch.cat(all_f_targets)
    
    occupation_metrics = compute_metrics(all_occupation_preds, all_occupation_targets)
    kei_bo_metrics = compute_metrics(all_kei_bo_preds, all_kei_bo_targets)
    energy_metrics = compute_metrics(all_energy_preds, all_energy_targets)
    s_metrics = compute_metrics(all_s_preds, all_s_targets)
    p_metrics = compute_metrics(all_p_preds, all_p_targets)
    d_metrics = compute_metrics(all_d_preds, all_d_targets)
    f_metrics = compute_metrics(all_f_preds, all_f_targets)
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("ORBITAL VALIDATION RESULTS - DETAILED ANALYSIS")
    report.append("=" * 80)
    
    # Format fold information header
    if fold_info.get('split_type') == 'element_based':
        element = fold_info.get('target_element', 'Unknown')
        report.append(f"FOLD INFORMATION ({element})")
    else:
        report.append(f"FOLD INFORMATION")
    
    report.append("-" * 40)
    
    # Show actual folders for element-based splits
    if fold_info.get('split_type') == 'element_based':
        if 'training_actual_folders' in fold_info:
            report.append(f"Training folders: {', '.join(fold_info['training_actual_folders'])}")
        if 'validation_actual_folders' in fold_info:
            report.append(f"Validation folders: {', '.join(fold_info['validation_actual_folders'])}")
    else:
        report.append(f"Training folders: {', '.join(fold_info['training_folders'])}")
        report.append(f"Validation folders: {fold_info['validation_folder']}")
    
    report.append(f"Training samples: {fold_info['train_graphs']}")
    report.append(f"Validation samples: {fold_info['val_graphs']}")
    report.append("")
    report.append("OVERALL PERFORMANCE METRICS")
    report.append("-" * 40)
    report.append(f"Orbital Occupations:")
    report.append(f"  MSE: {occupation_metrics['mse']:.6f}")
    report.append(f"KEI-BO Interactions:")
    report.append(f"  MSE: {kei_bo_metrics['mse']:.6f}")
    report.append(f"Molecular Energy:")
    report.append(f"  MSE: {energy_metrics['mse']:.6f}")
    report.append(f"Hybridization Percentages:")
    report.append(f"  s%: MSE={s_metrics['mse']:.6f}")
    report.append(f"  p%: MSE={p_metrics['mse']:.6f}")
    report.append(f"  d%: MSE={d_metrics['mse']:.6f}")
    report.append(f"  f%: MSE={f_metrics['mse']:.6f}")
    report.append("")
    report.append("MOLECULE-BY-MOLECULE ORBITAL ANALYSIS")
    report.append("=" * 80)
    
    for mol in all_molecules:
        report.append(f"MOLECULE {mol['mol_idx']}")
        report.append("-" * 20)
        report.append(f"Orbitals: {mol['num_orbitals']}, Interactions: {mol['num_interactions']}")
        report.append("")
        
        # Energy prediction
        energy_abs_error = abs(mol['energy_pred'] - mol['energy_target'])
        report.append(f"ENERGY PREDICTION:")
        report.append(f"Predicted: {mol['energy_pred']:12.6f}, Target: {mol['energy_target']:12.6f}, AbsError: {energy_abs_error:.6f}")
        report.append("")
        
        # Orbital occupation analysis
        report.append("ORBITAL OCCUPATIONS:")
        report.append("Orb#  AtomicNum  OrbType  Predicted    Target      AbsError")
        report.append("-" * 70)
        
        occupation_abs_errors = np.abs(mol['occupation_preds'] - mol['occupation_targets'])
        for i in range(mol['num_orbitals']):
            features = mol['orbital_features'][i]
            atomic_num = int(features[0])
            # Handle variable feature dimensions: [atomic_num] or [atomic_num, orb_type] or [atomic_num, orb_type, m_quantum]
            orb_type = int(features[1]) if len(features) > 1 else -1  # -1 indicates not available
            
            report.append(f"{i+1:4d} {atomic_num:9d} {orb_type:7d} "
                         f"{mol['occupation_preds'][i]:10.6f} {mol['occupation_targets'][i]:10.6f} "
                         f"{occupation_abs_errors[i]:10.6f}")
        
        occupation_mse = np.mean(occupation_abs_errors ** 2)
        report.append(f"Occupation MSE: {occupation_mse:.6f}")
        report.append("")
        
        # Hybridization analysis
        report.append("HYBRIDIZATION PERCENTAGES:")
        report.append("Orb#    s%_Pred   s%_Targ   p%_Pred   p%_Targ   d%_Pred   d%_Targ   f%_Pred   f%_Targ")
        report.append("-" * 90)
        
        s_abs_errors = np.abs(mol['s_preds'] - mol['s_targets'])
        p_abs_errors = np.abs(mol['p_preds'] - mol['p_targets'])
        d_abs_errors = np.abs(mol['d_preds'] - mol['d_targets'])
        f_abs_errors = np.abs(mol['f_preds'] - mol['f_targets'])
        
        for i in range(mol['num_orbitals']):
            report.append(f"{i+1:4d} "
                         f"{mol['s_preds'][i]:9.4f} {mol['s_targets'][i]:9.4f} "
                         f"{mol['p_preds'][i]:9.4f} {mol['p_targets'][i]:9.4f} "
                         f"{mol['d_preds'][i]:9.4f} {mol['d_targets'][i]:9.4f} "
                         f"{mol['f_preds'][i]:9.4f} {mol['f_targets'][i]:9.4f}")
        
        s_mse = np.mean(s_abs_errors ** 2)
        p_mse = np.mean(p_abs_errors ** 2)
        d_mse = np.mean(d_abs_errors ** 2)
        f_mse = np.mean(f_abs_errors ** 2)
        report.append(f"Hybridization MSE: s={s_mse:.6f}, p={p_mse:.6f}, d={d_mse:.6f}, f={f_mse:.6f}")
        report.append("")
        
        # KEI-BO interaction analysis
        if mol['num_interactions'] > 0:
            report.append("KEI-BO INTERACTIONS:")
            report.append("Inter   Orbitals   Distance   Predicted    Target      AbsError")
            report.append("-" * 65)
            
            kei_bo_abs_errors = np.abs(mol['kei_bo_preds'] - mol['kei_bo_targets'])
            edge_index = mol['edge_index']
            
            for i in range(mol['num_interactions']):
                orb1 = int(edge_index[0, i]) + 1  # Convert to 1-indexed
                orb2 = int(edge_index[1, i]) + 1
                distance = mol['edge_distances'][i]
                predicted = mol['kei_bo_preds'][i]
                target = mol['kei_bo_targets'][i]
                abs_error = kei_bo_abs_errors[i]
                
                report.append(f"{i+1:5d} {orb1:3d}-{orb2:<3d} {distance:9.3f} "
                             f"{predicted:10.6f} {target:10.6f} {abs_error:10.6f}")
            
            kei_bo_mse = np.mean(kei_bo_abs_errors ** 2)
            report.append(f"KEI-BO MSE: {kei_bo_mse:.6f}")
        else:
            report.append("No KEI-BO interactions found for this molecule.")
        
        report.append("=" * 80)
    
    return "\n".join(report)


def save_detailed_orbital_validation_report(model: OrbitalTripleTaskGNN, val_loader: DataLoader, 
                                          fold_info: Dict, device: torch.device, 
                                          run_dir: str, fold_num: int, normalizer=None, validation_mode: str = None):
    """Save detailed orbital validation report to text file."""
    report = generate_detailed_orbital_validation_report(model, val_loader, fold_info, device, normalizer)
    
    # Include validation_mode in filename if provided
    if validation_mode:
        report_path = os.path.join(run_dir, f"{validation_mode}_fold_{fold_num:02d}_{fold_info['validation_folder']}_orbital_validation.txt")
    else:
        report_path = os.path.join(run_dir, f"fold_{fold_num:02d}_{fold_info['validation_folder']}_orbital_validation.txt")
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Detailed orbital validation report saved to {report_path}")
    return report_path


def save_combined_orbital_results(all_results: List[Dict], all_fold_info: List[Dict], config: Dict, run_dir: str):
    """Save all orbital results in one combined JSON file."""
    
    # Helper function to convert numpy/torch values to Python floats
    def to_python_float(value):
        if hasattr(value, 'item'):  # torch tensor or numpy scalar
            return float(value.item())
        elif isinstance(value, (np.float32, np.float64)):  # numpy float
            return float(value)
        else:
            return float(value)  # regular python number
    
    # Prepare combined results
    combined_results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'n_folds': len(all_results),
            'config': config,
            'approach': 'orbital_centric'
        },
        'summary_statistics': {},
        'fold_results': []
    }
    
    # Calculate summary statistics across all folds - Updated for orbital naming
    val_occupation_mse = [to_python_float(r['val_metrics']['occupation'][-1]['mse']) for r in all_results]
    val_kei_bo_mse = [to_python_float(r['val_metrics']['kei_bo'][-1]['mse']) for r in all_results]
    val_energy_mse = [to_python_float(r['val_metrics']['energy'][-1]['mse']) for r in all_results]
    train_occupation_mse = [to_python_float(r['train_metrics']['occupation'][-1]['mse']) for r in all_results]
    train_kei_bo_mse = [to_python_float(r['train_metrics']['kei_bo'][-1]['mse']) for r in all_results]
    train_energy_mse = [to_python_float(r['train_metrics']['energy'][-1]['mse']) for r in all_results]
    
    # Add hybridization summary statistics
    val_s_mse = [to_python_float(r['val_metrics']['s_percent'][-1]['mse']) for r in all_results]
    val_p_mse = [to_python_float(r['val_metrics']['p_percent'][-1]['mse']) for r in all_results]
    val_d_mse = [to_python_float(r['val_metrics']['d_percent'][-1]['mse']) for r in all_results]
    val_f_mse = [to_python_float(r['val_metrics']['f_percent'][-1]['mse']) for r in all_results]
    train_s_mse = [to_python_float(r['train_metrics']['s_percent'][-1]['mse']) for r in all_results]
    train_p_mse = [to_python_float(r['train_metrics']['p_percent'][-1]['mse']) for r in all_results]
    train_d_mse = [to_python_float(r['train_metrics']['d_percent'][-1]['mse']) for r in all_results]
    train_f_mse = [to_python_float(r['train_metrics']['f_percent'][-1]['mse']) for r in all_results]
    
    combined_results['summary_statistics'] = {
        'val_occupation_mse': {
            'mean': float(np.mean(val_occupation_mse)),
            'std': float(np.std(val_occupation_mse)),
            'min': float(np.min(val_occupation_mse)),
            'max': float(np.max(val_occupation_mse)),
            'values': val_occupation_mse
        },
        'val_kei_bo_mse': {
            'mean': float(np.mean(val_kei_bo_mse)),
            'std': float(np.std(val_kei_bo_mse)),
            'min': float(np.min(val_kei_bo_mse)),
            'max': float(np.max(val_kei_bo_mse)),
            'values': val_kei_bo_mse
        },
        'val_energy_mse': {
            'mean': float(np.mean(val_energy_mse)),
            'std': float(np.std(val_energy_mse)),
            'min': float(np.min(val_energy_mse)),
            'max': float(np.max(val_energy_mse)),
            'values': val_energy_mse
        },
        'train_occupation_mse': {
            'mean': float(np.mean(train_occupation_mse)),
            'std': float(np.std(train_occupation_mse)),
            'min': float(np.min(train_occupation_mse)),
            'max': float(np.max(train_occupation_mse)),
            'values': train_occupation_mse
        },
        'train_kei_bo_mse': {
            'mean': float(np.mean(train_kei_bo_mse)),
            'std': float(np.std(train_kei_bo_mse)),
            'min': float(np.min(train_kei_bo_mse)),
            'max': float(np.max(train_kei_bo_mse)),
            'values': train_kei_bo_mse
        },
        'train_energy_mse': {
            'mean': float(np.mean(train_energy_mse)),
            'std': float(np.std(train_energy_mse)),
            'min': float(np.min(train_energy_mse)),
            'max': float(np.max(train_energy_mse)),
            'values': train_energy_mse
        },
        'val_s_percent_mse': {
            'mean': float(np.mean(val_s_mse)),
            'std': float(np.std(val_s_mse)),
            'min': float(np.min(val_s_mse)),
            'max': float(np.max(val_s_mse)),
            'values': val_s_mse
        },
        'val_p_percent_mse': {
            'mean': float(np.mean(val_p_mse)),
            'std': float(np.std(val_p_mse)),
            'min': float(np.min(val_p_mse)),
            'max': float(np.max(val_p_mse)),
            'values': val_p_mse
        },
        'val_d_percent_mse': {
            'mean': float(np.mean(val_d_mse)),
            'std': float(np.std(val_d_mse)),
            'min': float(np.min(val_d_mse)),
            'max': float(np.max(val_d_mse)),
            'values': val_d_mse
        },
        'val_f_percent_mse': {
            'mean': float(np.mean(val_f_mse)),
            'std': float(np.std(val_f_mse)),
            'min': float(np.min(val_f_mse)),
            'max': float(np.max(val_f_mse)),
            'values': val_f_mse
        },
        'train_s_percent_mse': {
            'mean': float(np.mean(train_s_mse)),
            'std': float(np.std(train_s_mse)),
            'min': float(np.min(train_s_mse)),
            'max': float(np.max(train_s_mse)),
            'values': train_s_mse
        },
        'train_p_percent_mse': {
            'mean': float(np.mean(train_p_mse)),
            'std': float(np.std(train_p_mse)),
            'min': float(np.min(train_p_mse)),
            'max': float(np.max(train_p_mse)),
            'values': train_p_mse
        },
        'train_d_percent_mse': {
            'mean': float(np.mean(train_d_mse)),
            'std': float(np.std(train_d_mse)),
            'min': float(np.min(train_d_mse)),
            'max': float(np.max(train_d_mse)),
            'values': train_d_mse
        },
        'train_f_percent_mse': {
            'mean': float(np.mean(train_f_mse)),
            'std': float(np.std(train_f_mse)),
            'min': float(np.min(train_f_mse)),
            'max': float(np.max(train_f_mse)),
            'values': train_f_mse
        }
    }
    
    # Add individual fold results - Updated for orbital naming
    for i, (results, fold_info) in enumerate(zip(all_results, all_fold_info)):
        fold_result = {
            'fold_number': i + 1,
            'validation_folder': fold_info['validation_folder'],
            'dataset_info': {
                'train_graphs': fold_info['train_graphs'],
                'val_graphs': fold_info['val_graphs'],
                'train_files': fold_info['train_files'],
                'val_files': fold_info['val_files'],
            },
            'final_metrics': {
                'train_occupation_mse': to_python_float(results['train_metrics']['occupation'][-1]['mse']),
                'train_occupation_mae': to_python_float(results['train_metrics']['occupation'][-1]['mae']),
                'val_occupation_mse': to_python_float(results['val_metrics']['occupation'][-1]['mse']),
                'val_occupation_mae': to_python_float(results['val_metrics']['occupation'][-1]['mae']),
                'train_kei_bo_mse': to_python_float(results['train_metrics']['kei_bo'][-1]['mse']),
                'train_kei_bo_mae': to_python_float(results['train_metrics']['kei_bo'][-1]['mae']),
                'val_kei_bo_mse': to_python_float(results['val_metrics']['kei_bo'][-1]['mse']),
                'val_kei_bo_mae': to_python_float(results['val_metrics']['kei_bo'][-1]['mae']),
                'train_energy_mse': to_python_float(results['train_metrics']['energy'][-1]['mse']),
                'train_energy_mae': to_python_float(results['train_metrics']['energy'][-1]['mae']),
                'val_energy_mse': to_python_float(results['val_metrics']['energy'][-1]['mse']),
                'val_energy_mae': to_python_float(results['val_metrics']['energy'][-1]['mae']),
                'train_s_percent_mse': to_python_float(results['train_metrics']['s_percent'][-1]['mse']),
                'train_s_percent_mae': to_python_float(results['train_metrics']['s_percent'][-1]['mae']),
                'val_s_percent_mse': to_python_float(results['val_metrics']['s_percent'][-1]['mse']),
                'val_s_percent_mae': to_python_float(results['val_metrics']['s_percent'][-1]['mae']),
                'train_p_percent_mse': to_python_float(results['train_metrics']['p_percent'][-1]['mse']),
                'train_p_percent_mae': to_python_float(results['train_metrics']['p_percent'][-1]['mae']),
                'val_p_percent_mse': to_python_float(results['val_metrics']['p_percent'][-1]['mse']),
                'val_p_percent_mae': to_python_float(results['val_metrics']['p_percent'][-1]['mae']),
                'train_d_percent_mse': to_python_float(results['train_metrics']['d_percent'][-1]['mse']),
                'train_d_percent_mae': to_python_float(results['train_metrics']['d_percent'][-1]['mae']),
                'val_d_percent_mse': to_python_float(results['val_metrics']['d_percent'][-1]['mse']),
                'val_d_percent_mae': to_python_float(results['val_metrics']['d_percent'][-1]['mae']),
                'train_f_percent_mse': to_python_float(results['train_metrics']['f_percent'][-1]['mse']),
                'train_f_percent_mae': to_python_float(results['train_metrics']['f_percent'][-1]['mae']),
                'val_f_percent_mse': to_python_float(results['val_metrics']['f_percent'][-1]['mse']),
                'val_f_percent_mae': to_python_float(results['val_metrics']['f_percent'][-1]['mae']),
            }
        }
        combined_results['fold_results'].append(fold_result)
    
    # Save combined results
    results_path = os.path.join(run_dir, 'orbital_results.json')
    with open(results_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nCombined orbital results saved to: {results_path}")
    
    # Print summary statistics
    print(f"\nORBITAL CROSS-VALIDATION SUMMARY:")
    print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 68)
    
    metrics_names = {
        'val_occupation_mse': 'Val Occupation MSE',
        'val_kei_bo_mse': 'Val KEI-BO MSE',
        'val_energy_mse': 'Val Energy MSE',
        'val_s_percent_mse': 'Val s% MSE',
        'val_p_percent_mse': 'Val p% MSE',
        'val_d_percent_mse': 'Val d% MSE',
        'val_f_percent_mse': 'Val f% MSE',
        'train_occupation_mse': 'Train Occupation MSE',
        'train_kei_bo_mse': 'Train KEI-BO MSE',
        'train_energy_mse': 'Train Energy MSE',
        'train_s_percent_mse': 'Train s% MSE',
        'train_p_percent_mse': 'Train p% MSE',
        'train_d_percent_mse': 'Train d% MSE',
        'train_f_percent_mse': 'Train f% MSE',
    }
    
    for key, name in metrics_names.items():
        stats = combined_results['summary_statistics'][key]
        print(f"{name:<20} {stats['mean']:<12.6f} {stats['std']:<12.6f} {stats['min']:<12.6f} {stats['max']:<12.6f}")