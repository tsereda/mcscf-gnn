# CLAUDE.MD - Project Context for AI Assistance

## Project Overview
This is an Orbital-Centric Graph Neural Network (GNN) project for predicting quantum chemistry properties from GAMESS orbital calculations. The model uses multi-head attention mechanisms to learn orbital-orbital interactions, particularly Kinetic Energy Integrals (KEI) and Bond Order (BO) matrices.

## Key Concepts
- **Orbital GNN**: Graph neural network where nodes represent molecular orbitals
- **MCSCF**: Multi-Configurational Self-Consistent Field quantum chemistry method
- **KBO**: Kinetic Bond Order - derived from orbital overlap integrals
- **Target Types**: `mcscf_energy` or `kinetic_energy` (controlled by `global_target_type`)

## Project Structure

### Core Files
- `main.py` - Entry point for training and hyperparameter sweeps
- `orbital_gnn.py` - GNN model architecture with attention mechanisms
- `orbital_trainer.py` - Training loop and model training utilities
- `orbital_parser.py` - GAMESS output file parser for orbital data
- `crossvalidation.py` - Cross-validation strategies (element-based, random split)
- `normalization.py` - Data normalization utilities
- `physics_loss.py` - Physics-informed loss functions
- `visualization.py` - Plotting and analysis tools
- `paper_analysis.py` - Comprehensive analysis script for research papers

### Configuration Files
- `sweep.yml` - W&B hyperparameter sweep configuration
- `tr_job.yml` - Kubernetes training job configuration
- `tr_pod.yml` - Kubernetes pod specification
- `pvc.yml` - Persistent volume claim for data storage

## Architecture Details

### OrbitalAttentionMessagePassing (orbital_gnn.py)
- Edge-level multi-head attention for orbital-orbital interactions
- Key insight: KBOs arise from orbital overlap integrals depending on:
  - Relative orbital orientations (p_x vs p_y, σ vs π bonding)
  - Phase relationships (bonding vs antibonding)
  - Energy level matching
- Distance alone insufficient - requires learned attention over orbital pairs

### Model Variants
- `create_orbital_model()` - Base GNN model
- `create_orbital_model_with_attention()` - Enhanced version with attention

## Common Operations

### Running Training
```bash
python main.py --config <config_path>
```

### Hyperparameter Sweeps
Uses W&B sweeps configured in `sweep.yml`:
- Grid search over `global_target_type` (mcscf_energy, kinetic_energy)
- Physics constraints toggle
- Validation methods: 'all', 'element-based', 'random'

### Cross-Validation Methods
1. **Element-based**: Train on certain elements, test on others
2. **Random split**: Random train/val/test split
3. **Single fold**: Specific folder-based splits

## Data Pipeline
1. Parse GAMESS output files (`orbital_parser.py`)
2. Extract orbital features and target matrices
3. Create graph representations (orbitals as nodes)
4. Normalize features (`normalization.py`)
5. Split into train/val/test (`crossvalidation.py`)
6. Train with `OrbitalTrainer`

## Key Parameters in sweep.yml
- `global_target_type`: Energy type to extract (mcscf_energy or kinetic_energy)
- `use_physics_constraints`: Apply physics-based regularization
- `validation_method`: Cross-validation strategy

## Dependencies
- PyTorch & PyTorch Geometric (GNN framework)
- W&B (experiment tracking)
- NumPy, SciPy (numerical operations)
- Matplotlib (visualization)

## Development Notes
- Model uses batch normalization and dropout for regularization
- Attention mechanism crucial for capturing quantum mechanical interactions
- Physics-informed losses help enforce known chemical constraints
- Kubernetes configs suggest this runs on GPU clusters

## Git Worktree Setup
This project uses git worktrees for isolated development branches. **IMPORTANT**: When working in a worktree, you MUST create a symlink to the data directory from the main repository:

```bash
ln -s /path/to/main/repo/data /path/to/worktree/data
```

Without this symlink, all training will fail because the GAMESS .log files won't be accessible. The `data/` directory contains ~814 GAMESS output files organized into 59 folders (molecules, diatomics, dimers, etc.).

**Note on macOS symlinks**: On macOS, `find` may not follow symlinks properly from outside the directory. Use `cd data && find .` or rely on Python's `glob` which handles symlinks correctly.

## Recent Changes (from git log)
- Added global target type parameter for energy extraction
- Refactored code structure for improved readability
- Added comprehensive analysis script for paper
- Support for both MCSCF energy and kinetic energy targets

## Tips for AI Assistants
1. This is quantum chemistry ML - orbital interactions are complex
2. Pay attention to `global_target_type` when debugging energy-related issues
3. Cross-validation is element-based or random - consider chemical diversity
4. Attention mechanisms are physics-motivated, not just performance tricks
5. Check `sweep.yml` for current hyperparameter search space
6. Files are parsed from GAMESS output format (quantum chemistry software)
