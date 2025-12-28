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
    get_all_elements_in_dataset,
    save_detailed_orbital_validation_report,
    save_combined_orbital_results
)
from visualization import (
    create_summary_plots, 
    create_comprehensive_analysis,
    ModelVisualizer
)


def compute_global_normalizer(all_files, config, run_dir):
    """
    Compute normalization statistics from ALL data files.
    
    Args:
        all_files: List of all file paths
        config: Configuration dictionary
        run_dir: Directory to save normalizer
        
    Returns:
        Fitted DataNormalizer instance
    """
    print("\n" + "="*70)
    print("COMPUTING GLOBAL NORMALIZATION STATISTICS")
    print("="*70)
    print(f"Processing all {len(all_files)} files to compute global statistics...")
    
    # Parse all files
    parser = OrbitalGAMESSParser(distance_cutoff=4.0, debug=False)
    all_graphs = process_orbital_files(parser, all_files)
    
    if len(all_graphs) == 0:
        raise ValueError("No valid graphs found in dataset")
    
    print(f"Successfully processed {len(all_graphs)} graphs")
    
    # Create temporary loader with all data
    global_loader = DataLoader(all_graphs, batch_size=32, shuffle=False)
    
    # Fit normalizer on all data
    global_normalizer = DataNormalizer(method=config['normalization']['method'])
    global_normalizer.fit(global_loader)
    
    # Save global normalizer
    global_norm_path = os.path.join(run_dir, "global_normalizer.pkl")
    global_normalizer.save(global_norm_path)
    
    print(f"Global normalizer saved to: {global_norm_path}")
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
            'orbital_embedding_dim': 32  # Updated for orbital model
        },
        'training': {
            'learning_rate': 0.0001,
            'weight_decay': 1e-4,
            'num_epochs': 250,
            'batch_size': 16,
            'print_frequency': 50,
            'occupation_weight': 1.5,  # Updated naming for orbital tasks
            'keibo_weight': 2.0,
            'energy_weight': 0.1
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
            'validation_mode': 'per_element'
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
        
        # Update from wandb.config
        config['normalization']['enabled'] = wandb.config.normalization_enabled
        if wandb.config.normalization_enabled:
            config['normalization']['global_norm'] = wandb.config.normalization_global
        
        config['model']['global_pooling_method'] = wandb.config.pooling_method
        config['training']['occupation_weight'] = wandb.config.occupation_weight
        config['training']['keibo_weight'] = wandb.config.keibo_weight
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
        
        # CLEAN: Single parameter controls loss balancing strategy
        strategy = getattr(wandb.config, 'loss_balancing_strategy', 'gradnorm')
        config['gradnorm']['enabled'] = (strategy == 'gradnorm')
        config['use_first_epoch_weighting'] = (strategy == 'first_epoch')
        
        print(f"\n{'='*70}")
        print("WANDB SWEEP RUN")
        print(f"{'='*70}")
        print(f"Normalization: {'Global' if wandb.config.normalization_global else 'Per-fold'} (enabled={wandb.config.normalization_enabled})")
        print(f"Loss Balancing Strategy: {strategy}")
        print(f"Weights: Occupation={wandb.config.occupation_weight:.2f}, KEI-BO={wandb.config.keibo_weight:.2f}, Energy={wandb.config.energy_weight:.2f}")
        print(f"Hidden Dim: {config['model']['hidden_dim']}")
        print(f"Num Layers: {config['model']['num_layers']}")
        print(f"Orbital Embedding Dim: {config['model']['orbital_embedding_dim']}")
    else:
        config = default_config
        print("Starting Orbital N-Fold Cross-Validation Training")
    
    print(f"\nConfiguration:")
    print(f"  Model: {config['model']['hidden_dim']}D hidden, {config['model']['num_layers']} layers, {config['model']['dropout']} dropout")
    print(f"  Pooling: {config['model']['global_pooling_method']}")
    print(f"  Orbital Embedding: {config['model']['orbital_embedding_dim']}D")
    print(f"  Training: LR={config['training']['learning_rate']}, WD={config['training']['weight_decay']}, {config['training']['num_epochs']} epochs")
    
    # Loss weighting strategy
    if config['gradnorm']['enabled']:
        print(f"  Loss Balancing: GradNorm (alpha={config['gradnorm']['alpha']}, lr={config['gradnorm']['learning_rate']})")
    elif config['use_first_epoch_weighting']:
        print(f"  Loss Balancing: First-Epoch Weighting")
    else:
        print(f"  Loss Balancing: Static (Occupation={config['training']['occupation_weight']}, KEI-BO={config['training']['keibo_weight']}, Energy={config['training']['energy_weight']})")
    
    norm_mode = "GLOBAL" if config['normalization'].get('global_norm', False) else "PER-FOLD"
    if config['normalization']['enabled']:
        print(f"  Normalization: ENABLED ({norm_mode}, method={config['normalization']['method']})")
    else:
        print(f"  Normalization: DISABLED")
    
    try:
        # Get all files from each folder
        folder_files = get_all_files_per_folder(config['data']['base_path'])
        
        if not folder_files:
            raise ValueError("No folders with .log files found")
        
        # Determine validation strategy
        validation_mode = config['data']['validation_mode']
        
        if validation_mode == 'per_element':
            # Collect all files
            all_files = []
            for folder_name, files in folder_files.items():
                all_files.extend(files)
            
            file_to_folder = create_file_to_folder_mapping(folder_files)
            available_elements = get_all_elements_in_dataset(all_files)
            
            # If in sweep mode, only use specified elements
            if is_sweep and config['wandb']['enabled']:
                sweep_elements = config['wandb']['sweep_elements']
                # Filter to only sweep elements that exist in dataset
                elements_to_validate = [e for e in sweep_elements if e in available_elements]
                if not elements_to_validate:
                    raise ValueError(f"None of sweep elements {sweep_elements} found in dataset {available_elements}")
                print(f"\nSWEEP MODE: Validating only on elements: {elements_to_validate}")
            else:
                elements_to_validate = available_elements
            
            n_folds = len(elements_to_validate)
            fold_descriptions = [f"Val: {element}" for element in elements_to_validate]
            
            print(f"\nPlanning element-based cross-validation:")
            print(f"Using {len(elements_to_validate)} elements: {elements_to_validate}")
            for i, (element, desc) in enumerate(zip(elements_to_validate, fold_descriptions), 1):
                print(f"  Fold {i:2d}: {desc}")
        
        elif validation_mode == 'per_molecule':
            folders_with_files = [f for f, files in folder_files.items() if len(files) > 0]
            n_folds = len(folders_with_files)
            all_files = []
            for folder_name, files in folder_files.items():
                all_files.extend(files)
            elements_to_validate = None
            
            print(f"\nPlanning {n_folds}-fold cross-validation:")
            for i, folder_name in enumerate(folders_with_files, 1):
                print(f"  Fold {i:2d}: Validation = {folder_name} ({len(folder_files[folder_name])} files)")
        else:
            raise ValueError(f"Invalid validation_mode: {validation_mode}")
        
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
        
        # Save configuration
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Compute global normalizer if enabled
        global_normalizer = None
        if config['normalization']['enabled'] and config['normalization'].get('global_norm', False):
            global_normalizer = compute_global_normalizer(all_files, config, run_dir)
        
        # Initialize visualizer
        visualizer = ModelVisualizer()
        
        # Run training for each fold
        all_results = []
        all_fold_info = []
        all_trainers = []
        
        for fold_num in range(n_folds):
            if validation_mode == 'per_element':
                # Get the element index in the original available_elements list
                target_element = elements_to_validate[fold_num]
                element_index = available_elements.index(target_element)
                
                fold_description = fold_descriptions[fold_num]
                print(f"\n{'='*70}")
                print(f"FOLD {fold_num + 1}/{n_folds} - {fold_description}")
                print(f"{'='*70}")
                
                train_loader, val_loader, fold_info = prepare_element_based_fold_data(
                    all_files, file_to_folder, element_index, available_elements, config['training']['batch_size']
                )
                
            elif validation_mode == 'per_molecule':
                validation_folder = folders_with_files[fold_num]
                print(f"\n{'='*70}")
                print(f"FOLD {fold_num + 1}/{n_folds} - Validation Folder: {validation_folder}")
                print(f"{'='*70}")
                
                train_loader, val_loader, fold_info = prepare_single_fold_data(
                    folder_files, validation_folder, config['training']['batch_size']
                )
            
            # Determine normalizer
            normalizer = None
            if config['normalization']['enabled']:
                if config['normalization'].get('global_norm', False):
                    normalizer = global_normalizer
                    print(f"Using GLOBAL normalizer")
                else:
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
                orbital_embedding_dim=config['model']['orbital_embedding_dim']
            )
            
            # Create trainer
            trainer = OrbitalGAMESSTrainer(
                model=model,
                learning_rate=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay'],
                occupation_weight=config['training']['occupation_weight'],
                keibo_weight=config['training']['keibo_weight'],
                energy_weight=config['training']['energy_weight'],
                normalizer=normalizer,
                use_gradnorm=config['gradnorm']['enabled'],
                gradnorm_alpha=config['gradnorm']['alpha'],
                gradnorm_lr=config['gradnorm']['learning_rate'],
                use_first_epoch_weighting=config['use_first_epoch_weighting'],
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
            final_val_keibo = results['val_metrics']['keibo'][-1]
            final_val_energy = results['val_metrics']['energy'][-1]
            print(f"\nFold {fold_num + 1} Complete:")
            print(f"    Occupation: Val MSE={final_val_occupation['mse']:.6f}")
            print(f"    KEI-BO: Val MSE={final_val_keibo['mse']:.6f}")
            print(f"    Energy: Val MSE={final_val_energy['mse']:.6f}")
        
        # Compute average metrics across folds for WandB
        if is_sweep and config['wandb']['enabled']:
            avg_val_occupation_mse = np.mean([r['val_metrics']['occupation'][-1]['mse'] for r in all_results])
            avg_val_keibo_mse = np.mean([r['val_metrics']['keibo'][-1]['mse'] for r in all_results])
            avg_val_energy_mse = np.mean([r['val_metrics']['energy'][-1]['mse'] for r in all_results])
            avg_val_mse = (avg_val_occupation_mse + avg_val_keibo_mse + avg_val_energy_mse) / 3.0
            
            # Log to WandB
            wandb.log({
                'avg_val_mse': avg_val_mse,
                'avg_val_occupation_mse': avg_val_occupation_mse,
                'avg_val_keibo_mse': avg_val_keibo_mse,
                'avg_val_energy_mse': avg_val_energy_mse
            })
            
            print(f"\n{'='*70}")
            print("SWEEP METRICS")
            print(f"{'='*70}")
            print(f"Average Validation MSE: {avg_val_mse:.6f}")
            print(f"  Occupation MSE: {avg_val_occupation_mse:.6f}")
            print(f"  KEI-BO MSE: {avg_val_keibo_mse:.6f}")
            print(f"  Energy MSE: {avg_val_energy_mse:.6f}")
        
        print(f"\n{'='*70}")
        print("CREATING SUMMARY VISUALIZATIONS")
        print(f"{'='*70}")
        
        # Determine folder names for summary
        if validation_mode == 'per_element':
            summary_folder_names = elements_to_validate
        else:
            summary_folder_names = folders_with_files
        
        # Create summary plots
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