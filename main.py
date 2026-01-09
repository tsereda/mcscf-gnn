"""
FIXED VERSION: Global normalization now properly excludes validation fold to prevent data leakage

Key changes:
1. compute_global_normalizer now accepts exclude_files parameter
2. Global normalizer is recomputed for each fold, excluding that fold's validation files
3. This prevents data leakage while maintaining global statistics across training data
"""

import os
import json
from datetime import datetime
import wandb
import sys
import numpy as np

from orbital_gnn import create_orbital_model
from orbital_trainer import OrbitalGAMESSTrainer, process_orbital_files
from orbital_parser import OrbitalGAMESSParser
from torch_geometric.data import DataLoader
from normalization import DataNormalizer
from crossvalidation import (
    get_all_files_per_folder,
    create_file_to_folder_mapping,
    prepare_single_fold_data,
    prepare_element_based_fold_data,
    prepare_random_split_data,
    get_all_elements_in_dataset,
    save_detailed_orbital_validation_report,
    save_combined_orbital_results
)
from visualization import (
    create_summary_plots, 
    create_comprehensive_analysis,
    ModelVisualizer
)


def compute_fold_global_normalizer(all_files, exclude_files, config, run_dir, fold_num):
    """
    Compute normalization statistics from training data only (excluding validation fold).
    
    This is the CORRECT way to do global normalization - compute stats from all TRAINING
    data across folds, but exclude the current validation fold.
    
    Args:
        all_files: List of all file paths
        exclude_files: Files in current validation fold to exclude
        config: Configuration dictionary
        run_dir: Directory to save normalizer
        fold_num: Current fold number for naming
        
    Returns:
        Fitted DataNormalizer instance
    """
    print("\n" + "="*70)
    print(f"COMPUTING GLOBAL NORMALIZATION STATISTICS (Fold {fold_num})")
    print("="*70)
    
    # Exclude validation files
    files_to_use = [f for f in all_files if f not in exclude_files]
    print(f"Using {len(files_to_use)}/{len(all_files)} files (excluding {len(exclude_files)} validation files)")
    
    # Parse training files only
    parser = OrbitalGAMESSParser(distance_cutoff=4.0, debug=False)
    train_graphs = process_orbital_files(parser, files_to_use)
    
    if len(train_graphs) == 0:
        raise ValueError("No valid training graphs found")
    
    print(f"Successfully processed {len(train_graphs)} training graphs")
    
    # Create temporary loader with training data only
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=False)
    
    # Fit normalizer on training data
    global_normalizer = DataNormalizer(method=config['normalization']['method'])
    global_normalizer.fit(train_loader)
    
    # Save normalizer
    norm_path = os.path.join(run_dir, f"fold_{fold_num:02d}_global_normalizer.pkl")
    global_normalizer.save(norm_path)
    
    print(f"Fold {fold_num} global normalizer saved to: {norm_path}")
    print("="*70)
    
    return global_normalizer


def main():
    """Main training script with n-fold cross-validation, normalization, GradNorm, and WandB support."""
    
    # Check if running as WandB sweep
    is_sweep = len(sys.argv) > 1 and sys.argv[1] == '--sweep'
    
    # Default configuration
    default_config = {
        'model': {
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.25,
            'global_pooling_method': 'sum',
            'orbital_embedding_dim': 32,
            'use_rbf_distance': False,
            'num_rbf': 50,
            'rbf_cutoff': 5.0,
            'include_hybridization': True
        },
        'training': {
            'learning_rate': 0.0001,
            'weight_decay': 1e-4,
            'num_epochs': 250,
            'batch_size': 16,
            'print_frequency': 50,
            'occupation_weight': 1.5,
            'kei_bo_weight': 2.0,
            'energy_weight': 0.1,
            'hybrid_weight': 1.0,
            'use_uncertainty_weighting': True
        },
        'gradnorm': {
            'enabled': False,
            'alpha': 1.5,
            'learning_rate': 0.025
        },
        'normalization': {
            'enabled': False,
            'method': 'standardize',
            'global_norm': False,
        },
        'data': {
            'base_path': 'data',
            'validation_mode': 'per_element',  # 'random_split', 'per_element', or 'per_molecule'
            'validation_subset': None,  # None = all, or list like ['H', 'C'] or ['b2', 'h2o']
            'val_split_ratio': 0.2,  # Only for random_split
            'random_seed': 42  # For random_split reproducibility
        },
        'wandb': {
            'enabled': is_sweep,
            'project': 'orbital-gamess-gnn-sweep',
            'sweep_elements': ['C', 'H']
        },
        'use_first_epoch_weighting': False,
    }
    
    # Initialize WandB if in sweep mode
    if is_sweep:
        run = wandb.init()
        
        # Override config with sweep parameters
        config = default_config.copy()
        
        config['normalization']['enabled'] = wandb.config.normalization_enabled
        if wandb.config.normalization_enabled:
            config['normalization']['global_norm'] = wandb.config.normalization_global
        
        config['model']['global_pooling_method'] = wandb.config.pooling_method
        config['training']['occupation_weight'] = wandb.config.occupation_weight
        config['training']['kei_bo_weight'] = wandb.config.kei_bo_weight
        config['training']['energy_weight'] = wandb.config.energy_weight
        config['model']['hidden_dim'] = wandb.config.hidden_dim
        config['model']['orbital_embedding_dim'] = getattr(
            wandb.config, 
            'orbital_embedding_dim', 
            config['model']['orbital_embedding_dim']
        )
        config['model']['num_layers'] = getattr(
            wandb.config,
            'num_layers',
            config['model']['num_layers']
        )
        config['training']['num_epochs'] = getattr(
            wandb.config,
            'epochs',
            config['training']['num_epochs']
        )
        
        # RBF distance encoding parameters
        config['model']['use_rbf_distance'] = getattr(
            wandb.config,
            'use_rbf_distance',
            config['model']['use_rbf_distance']
        )
        config['model']['num_rbf'] = getattr(
            wandb.config,
            'num_rbf',
            config['model']['num_rbf']
        )
        config['model']['rbf_cutoff'] = getattr(
            wandb.config,
            'rbf_cutoff',
            config['model']['rbf_cutoff']
        )
        
        # Loss balancing strategy - handle both GradNorm and uncertainty weighting
        strategy = getattr(wandb.config, 'loss_balancing_strategy', 'static')
        config['gradnorm']['enabled'] = (strategy == 'gradnorm')
        config['training']['use_uncertainty_weighting'] = (strategy == 'uncertainty_weighting')
        config['use_first_epoch_weighting'] = (strategy == 'first_epoch')
        
        # Hybridization parameter
        config['model']['include_hybridization'] = getattr(
            wandb.config,
            'include_hybridization',
            True  # Default to True for backward compatibility
        )
        
        # Element baselines parameter (physics-informed inductive bias)
        config['model']['use_element_baselines'] = getattr(
            wandb.config,
            'use_element_baselines',
            True  # Default to True for physics-informed learning
        )
        
        # Validation configuration
        config['data']['validation_mode'] = getattr(wandb.config, 'validation_method', 'per_element')
        config['data']['validation_subset'] = getattr(wandb.config, 'validation_subset', None)
        config['data']['val_split_ratio'] = getattr(wandb.config, 'val_split_ratio', 0.2)
        config['data']['random_seed'] = getattr(wandb.config, 'random_seed', 42)
        
        print(f"\n{'='*70}")
        print("WANDB SWEEP RUN")
        print(f"{'='*70}")
        print(f"Validation Method: {config['data']['validation_mode']}")
        print(f"Validation Subset: {config['data']['validation_subset'] if config['data']['validation_subset'] else 'All'}")
        if config['data']['validation_mode'] == 'random_split':
            print(f"Val Split Ratio: {config['data']['val_split_ratio']}")
            print(f"Random Seed: {config['data']['random_seed']}")
        print(f"Normalization: {'Global' if wandb.config.normalization_global else 'Per-fold'} (enabled={wandb.config.normalization_enabled})")
        print(f"Loss Balancing Strategy: {strategy}")
        print(f"Include Hybridization: {config['model']['include_hybridization']}")
        print(f"Use Element Baselines: {config['model']['use_element_baselines']}")
        print(f"Weights: Occupation={wandb.config.occupation_weight:.2f}, KEI-BO={wandb.config.kei_bo_weight:.2f}, Energy={wandb.config.energy_weight:.2f}")
        print(f"Hidden Dim: {config['model']['hidden_dim']}")
        print(f"Num Layers: {config['model']['num_layers']}")
        print(f"Orbital Embedding Dim: {config['model']['orbital_embedding_dim']}")
        print(f"RBF Distance Encoding: {config['model']['use_rbf_distance']} (num_rbf={config['model']['num_rbf']}, cutoff={config['model']['rbf_cutoff']})")
        print(f"Epochs: {config['training']['num_epochs']}")
    else:
        config = default_config
        print("Starting Orbital N-Fold Cross-Validation Training")
    
    print(f"\nConfiguration:")
    print(f"  Model: {config['model']['hidden_dim']}D hidden, {config['model']['num_layers']} layers, {config['model']['dropout']} dropout")
    print(f"  Pooling: {config['model']['global_pooling_method']}")
    print(f"  Orbital Embedding: {config['model']['orbital_embedding_dim']}D")
    print(f"  Training: LR={config['training']['learning_rate']}, WD={config['training']['weight_decay']}, {config['training']['num_epochs']} epochs")
    
    if config['gradnorm']['enabled']:
        print(f"  Loss Balancing: GradNorm (alpha={config['gradnorm']['alpha']}, lr={config['gradnorm']['learning_rate']})")
    elif config['use_first_epoch_weighting']:
        print(f"  Loss Balancing: First-Epoch Weighting")
    else:
        print(f"  Loss Balancing: Static (Occupation={config['training']['occupation_weight']}, KEI-BO={config['training']['kei_bo_weight']}, Energy={config['training']['energy_weight']})")
    
    norm_mode = "GLOBAL" if config['normalization'].get('global_norm', False) else "PER-FOLD"
    if config['normalization']['enabled']:
        print(f"  Normalization: ENABLED ({norm_mode}, method={config['normalization']['method']})")
    else:
        print(f"  Normalization: DISABLED")
    
    try:
        folder_files = get_all_files_per_folder(config['data']['base_path'])
        
        if not folder_files:
            raise ValueError("No folders with .log files found")
        
        validation_mode = config['data']['validation_mode']
        validation_subset = config['data']['validation_subset']
        
        # Collect all files
        all_files = []
        for folder_name, files in folder_files.items():
            all_files.extend(files)
        
        if validation_mode == 'random_split':
            # Random 80/20 split
            n_folds = 1
            elements_to_validate = None
            
            print(f"\nPlanning random split validation:")
            print(f"  Split ratio: {config['data']['val_split_ratio']:.1%} validation")
            print(f"  Random seed: {config['data']['random_seed']}")
            print(f"  Total files: {len(all_files)}")
            
        elif validation_mode == 'per_element':
            # Element-based cross-validation
            file_to_folder = create_file_to_folder_mapping(folder_files)
            available_elements = get_all_elements_in_dataset(all_files)
            
            # Filter elements if validation_subset is specified
            if validation_subset:
                elements_to_validate = [e for e in validation_subset if e in available_elements]
                if not elements_to_validate:
                    raise ValueError(f"None of specified elements {validation_subset} found in dataset {available_elements}")
                print(f"\nValidation subset specified: Using elements {elements_to_validate}")
            else:
                elements_to_validate = available_elements
                print(f"\nUsing all available elements: {elements_to_validate}")
            
            n_folds = len(elements_to_validate)
            fold_descriptions = [f"Val: {element}" for element in elements_to_validate]
            
            print(f"\nPlanning element-based cross-validation:")
            print(f"Using {len(elements_to_validate)} elements: {elements_to_validate}")
            for i, (element, desc) in enumerate(zip(elements_to_validate, fold_descriptions), 1):
                print(f"  Fold {i:2d}: {desc}")
        
        elif validation_mode == 'per_molecule':
            # Per-molecule/folder cross-validation
            folders_with_files = [f for f, files in folder_files.items() if len(files) > 0]
            
            # Filter folders if validation_subset is specified
            if validation_subset:
                folders_to_validate = [f for f in validation_subset if f in folders_with_files]
                if not folders_to_validate:
                    raise ValueError(f"None of specified folders {validation_subset} found in dataset {folders_with_files}")
                print(f"\nValidation subset specified: Using folders {folders_to_validate}")
            else:
                folders_to_validate = folders_with_files
                print(f"\nUsing all available folders")
            
            n_folds = len(folders_to_validate)
            elements_to_validate = None
            
            print(f"\nPlanning {n_folds}-fold cross-validation:")
            for i, folder_name in enumerate(folders_to_validate, 1):
                print(f"  Fold {i:2d}: Validation = {folder_name} ({len(folder_files[folder_name])} files)")
        else:
            raise ValueError(f"Invalid validation_mode: {validation_mode}. Must be 'random_split', 'per_element', or 'per_molecule'")
        
        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        norm_suffix = ""
        if config['normalization']['enabled']:
            norm_suffix = "_global_norm" if config['normalization'].get('global_norm', False) else "_per_fold_norm"
        
        gradnorm_suffix = "_gradnorm" if config['gradnorm']['enabled'] else ""
        pooling_suffix = f"_{config['model']['global_pooling_method']}"
        
        if is_sweep:
            run_dir = f"runs/orbital_sweep_{wandb.run.id}_{timestamp}{norm_suffix}{gradnorm_suffix}{pooling_suffix}"
        else:
            run_dir = f"runs/orbital_cross_validation_{timestamp}{norm_suffix}{gradnorm_suffix}{pooling_suffix}"
        
        os.makedirs(run_dir, exist_ok=True)
        print(f"\nResults will be saved to: {run_dir}")
        
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize visualizer
        visualizer = ModelVisualizer()
        
        # Run training for each fold
        all_results = []
        all_fold_info = []
        all_trainers = []
        
        for fold_num in range(n_folds):
            if validation_mode == 'random_split':
                print(f"\n{'='*70}")
                print(f"RANDOM SPLIT VALIDATION")
                print(f"{'='*70}")
                
                train_loader, val_loader, fold_info = prepare_random_split_data(
                    all_files,
                    split_ratio=config['data']['val_split_ratio'],
                    random_seed=config['data']['random_seed'],
                    batch_size=config['training']['batch_size']
                )
                
                # For random split, we need to extract val_files from fold_info
                # Since we don't have direct access, we'll use a proportion of all_files
                split_idx = int(len(all_files) * (1 - config['data']['val_split_ratio']))
                np.random.seed(config['data']['random_seed'])
                shuffled = all_files.copy()
                np.random.shuffle(shuffled)
                val_files = shuffled[split_idx:]
                
            elif validation_mode == 'per_element':
                target_element = elements_to_validate[fold_num]
                element_index = available_elements.index(target_element)
                
                fold_description = fold_descriptions[fold_num]
                print(f"\n{'='*70}")
                print(f"FOLD {fold_num + 1}/{n_folds} - {fold_description}")
                print(f"{'='*70}")
                
                train_loader, val_loader, fold_info = prepare_element_based_fold_data(
                    all_files, file_to_folder, element_index, available_elements, config['training']['batch_size']
                )
                
                # Get validation files for this fold
                val_files = [f for f in all_files if file_to_folder.get(f) in fold_info.get('validation_actual_folders', [])]
                
            elif validation_mode == 'per_molecule':
                validation_folder = folders_to_validate[fold_num]
                print(f"\n{'='*70}")
                print(f"FOLD {fold_num + 1}/{n_folds} - Validation Folder: {validation_folder}")
                print(f"{'='*70}")
                
                train_loader, val_loader, fold_info = prepare_single_fold_data(
                    folder_files, validation_folder, config['training']['batch_size']
                )
                
                val_files = folder_files[validation_folder]
            
            # Determine normalizer
            normalizer = None
            if config['normalization']['enabled']:
                if config['normalization'].get('global_norm', False):
                    # FIXED: Compute global normalizer excluding validation fold
                    normalizer = compute_fold_global_normalizer(
                        all_files, 
                        val_files,
                        config, 
                        run_dir, 
                        fold_num + 1
                    )
                    print(f"Using GLOBAL normalizer (excluding validation fold)")
                else:
                    # Per-fold normalization (unchanged)
                    normalizer = DataNormalizer(method=config['normalization']['method'])
                    normalizer.fit(train_loader)
                    norm_stats_path = os.path.join(run_dir, f"fold_{fold_num + 1:02d}_normalizer.pkl")
                    normalizer.save(norm_stats_path)
                    print(f"Using PER-FOLD normalizer")
            
            # Create orbital model
            model = create_orbital_model(
                hidden_dim=config['model']['hidden_dim'],
                num_layers=config['model']['num_layers'],
                dropout=config['model']['dropout'],
                global_pooling_method=config['model']['global_pooling_method'],
                orbital_embedding_dim=config['model']['orbital_embedding_dim'],
                use_rbf_distance=config['model']['use_rbf_distance'],
                num_rbf=config['model']['num_rbf'],
                rbf_cutoff=config['model']['rbf_cutoff'],
                include_hybridization=config['model']['include_hybridization'],
                use_element_baselines=config['model'].get('use_element_baselines', True)
            )
            
            # Create trainer
            trainer = OrbitalGAMESSTrainer(
                model=model,
                learning_rate=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay'],
                occupation_weight=config['training']['occupation_weight'],
                kei_bo_weight=config['training']['kei_bo_weight'],
                energy_weight=config['training']['energy_weight'],
                hybrid_weight=config['training'].get('hybrid_weight', 1.0),
                normalizer=normalizer,
                use_uncertainty_weighting=config['training'].get('use_uncertainty_weighting', True),
                use_gradnorm=config['gradnorm']['enabled'],
                gradnorm_alpha=config['gradnorm']['alpha'],
                gradnorm_lr=config['gradnorm']['learning_rate'],
                wandb_enabled=is_sweep
            )
            
            # Train
            results = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config['training']['num_epochs'],
                print_frequency=config['training']['print_frequency']
            )
            
            # Save training curves
            fold_curves_path = os.path.join(run_dir, f"fold_{fold_num + 1:02d}_{fold_info['validation_folder']}_curves.png")
            trainer.plot_training_curves(
                fold_curves_path, 
                title_suffix=f" - Fold {fold_num + 1} (Val: {fold_info['validation_folder']})"
            )
            
            # Generate validation report
            save_detailed_orbital_validation_report(
                trainer.model, val_loader, fold_info, trainer.device, run_dir, fold_num + 1, normalizer
            )
            
            # Store results
            all_results.append(results)
            all_fold_info.append(fold_info)
            all_trainers.append(trainer)
            
            # Print fold summary
            final_val_occupation = results['val_metrics']['occupation'][-1]
            final_val_kei_bo = results['val_metrics']['kei_bo'][-1]
            final_val_energy = results['val_metrics']['energy'][-1]
            print(f"\nFold {fold_num + 1} Complete:")
            print(f"    Occupation: Val MSE={final_val_occupation['mse']:.6f}")
            print(f"    KEI-BO: Val MSE={final_val_kei_bo['mse']:.6f}")
            print(f"    Energy: Val MSE={final_val_energy['mse']:.6f}")
        
        # Compute average metrics across folds for WandB
        if is_sweep and config['wandb']['enabled']:
            avg_val_occupation_mse = np.mean([r['val_metrics']['occupation'][-1]['mse'] for r in all_results])
            avg_val_kei_bo_mse = np.mean([r['val_metrics']['kei_bo'][-1]['mse'] for r in all_results])
            avg_val_energy_mse = np.mean([r['val_metrics']['energy'][-1]['mse'] for r in all_results])
            avg_val_mse = (avg_val_occupation_mse + avg_val_kei_bo_mse + avg_val_energy_mse) / 3.0
            
            wandb.log({
                'avg_val_mse': avg_val_mse,
                'avg_val_occupation_mse': avg_val_occupation_mse,
                'avg_val_kei_bo_mse': avg_val_kei_bo_mse,
                'avg_val_energy_mse': avg_val_energy_mse
            })
            
            print(f"\n{'='*70}")
            print("SWEEP METRICS")
            print(f"{'='*70}")
            print(f"Average Validation MSE: {avg_val_mse:.6f}")
            print(f"  Occupation MSE: {avg_val_occupation_mse:.6f}")
            print(f"  KEI-BO MSE: {avg_val_kei_bo_mse:.6f}")
            print(f"  Energy MSE: {avg_val_energy_mse:.6f}")
        
        print(f"\n{'='*70}")
        print("CREATING SUMMARY VISUALIZATIONS")
        print(f"{'='*70}")
        
        if validation_mode == 'random_split':
            summary_folder_names = ['random_split']
        elif validation_mode == 'per_element':
            summary_folder_names = elements_to_validate
        else:  # per_molecule
            summary_folder_names = folders_to_validate
        
        create_summary_plots(all_results, run_dir, summary_folder_names)
        save_combined_orbital_results(all_results, all_fold_info, config, run_dir)
        
        print(f"\n{'='*70}")
        print("ORBITAL CROSS-VALIDATION COMPLETE")
        print(f"{'='*70}")
        print(f"All results saved to: {run_dir}")
        
        if is_sweep:
            wandb.finish()
        
    except Exception as e:
        print(f"Orbital cross-validation failed: {e}")
        import traceback
        traceback.print_exc()
        if is_sweep:
            wandb.finish(exit_code=1)


if __name__ == "__main__":
    main()