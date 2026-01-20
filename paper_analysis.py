#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Orbital-Centric GNN Paper
Generates all figures needed for publication from WandB sweep data.

Figures generated:
1. Loss balancing strategy comparison (bar chart)
2. Validation strategy comparison (random vs per-molecule vs per-element)
3. Architecture ablation (attention vs non-attention, layers, hidden dim)
4. Feature importance (RBF, element baselines, hybridization, orbital features)
5. Pooling method comparison
6. Normalization strategy comparison
7. Hyperparameter sensitivity heatmaps
8. Best model performance summary table
9. Training convergence curves (if available)
10. Pareto frontier: accuracy vs complexity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette for consistency
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'success': '#3A7D44',
    'neutral': '#6C757D',
}

PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3A7D44', '#6C757D']


def load_and_clean_data(filepath):
    """Load WandB export and clean data."""
    df = pd.read_csv(filepath)
    
    # Convert boolean columns
    bool_cols = ['use_physics_constraints', 'use_attention', 'include_hybridization',
                 'include_m_quantum', 'include_orbital_type', 'normalization_enabled',
                 'normalization_global', 'use_element_baselines', 'use_rbf_distance']
    
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({'true': True, 'false': False, True: True, False: False})
    
    # Convert numeric columns
    numeric_cols = ['energy_weight', 'hidden_dim', 'num_layers', 'num_rbf', 
                    'occupation_weight', 'orbital_embedding_dim', 'rbf_cutoff',
                    'kei_bo_weight', 'epoch', 'Runtime',
                    'overall/avg_val_kei_bo_mse', 'random_split/avg_val_kei_bo_mse',
                    'per_molecule/avg_val_kei_bo_mse', 'per_element/avg_val_kei_bo_mse',
                    'random_split/avg_val_occupation_mse', 'random_split/avg_val_energy_mse',
                    'train_kei_bo_mse', 'train_occupation_mse', 'train_energy_mse',
                    'val_kei_bo_mse', 'val_occupation_mse', 'val_energy_mse',
                    'train_s_percent_mse', 'train_p_percent_mse', 'train_d_percent_mse', 'train_f_percent_mse',
                    'val_s_percent_mse', 'val_p_percent_mse', 'val_d_percent_mse', 'val_f_percent_mse']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter out failed runs (missing key metrics or very high MSE indicating failure)
    df = df[df['epoch'] >= 50]  # Only runs that trained for at least 50 epochs
    
    # Remove extreme outliers (likely failed runs)
    if 'overall/avg_val_kei_bo_mse' in df.columns:
        df = df[df['overall/avg_val_kei_bo_mse'] < 10]  # Remove catastrophic failures
    
    print(f"Loaded {len(df)} valid experiments")
    return df


def fig1_loss_balancing_comparison(df, output_dir):
    """Figure 1: Comparison of loss balancing strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    strategies = ['static', 'uncertainty_weighting', 'first_epoch', 'gradnorm']
    strategy_labels = ['Static', 'Uncertainty\nWeighting', 'First-Epoch', 'GradNorm']
    
    metrics = [
        ('random_split/avg_val_kei_bo_mse', 'KEI-BO MSE', 'Random Split'),
        ('random_split/avg_val_occupation_mse', 'Occupation MSE', 'Random Split'),
        ('per_element/avg_val_kei_bo_mse', 'KEI-BO MSE', 'Leave-One-Element-Out')
    ]
    
    for ax_idx, (metric, ylabel, title) in enumerate(metrics):
        means = []
        stds = []
        valid_strategies = []
        valid_labels = []
        
        for strat, label in zip(strategies, strategy_labels):
            subset = df[df['loss_balancing_strategy'] == strat]
            if len(subset) > 0 and metric in subset.columns:
                vals = subset[metric].dropna()
                if len(vals) > 0:
                    means.append(vals.mean())
                    stds.append(vals.std())
                    valid_strategies.append(strat)
                    valid_labels.append(label)
        
        if means:
            x_pos = np.arange(len(means))
            bars = axes[ax_idx].bar(x_pos, means, yerr=stds, capsize=5, 
                                    color=PALETTE[:len(means)], alpha=0.8, edgecolor='black', linewidth=1)
            axes[ax_idx].set_xticks(x_pos)
            axes[ax_idx].set_xticklabels(valid_labels)
            axes[ax_idx].set_ylabel(ylabel)
            axes[ax_idx].set_title(title, fontweight='bold')
            axes[ax_idx].set_yscale('log')
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                axes[ax_idx].annotate(f'{mean:.4f}',
                                      xy=(bar.get_x() + bar.get_width() / 2, height),
                                      xytext=(0, 3), textcoords="offset points",
                                      ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Impact of Multi-Task Loss Balancing Strategy', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_loss_balancing.png')
    plt.savefig(output_dir / 'fig1_loss_balancing.pdf')
    plt.close()
    print("Generated: fig1_loss_balancing.png/pdf")


def fig2_validation_strategy_comparison(df, output_dir):
    """Figure 2: Comparison across validation strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get best runs for each validation strategy
    val_strategies = ['random_split/avg_val_kei_bo_mse', 
                      'per_molecule/avg_val_kei_bo_mse', 
                      'per_element/avg_val_kei_bo_mse']
    
    labels = ['Random Split\n(80/20)', 'Leave-One-\nMolecule-Out', 'Leave-One-\nElement-Out']
    
    # Panel A: Box plots of all runs
    data_for_box = []
    valid_labels = []
    for strat, label in zip(val_strategies, labels):
        if strat in df.columns:
            vals = df[strat].dropna()
            if len(vals) > 0:
                data_for_box.append(vals.values)
                valid_labels.append(label)
    
    if data_for_box:
        bp = axes[0].boxplot(data_for_box, labels=valid_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], PALETTE[:len(data_for_box)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_ylabel('KEI-BO MSE')
        axes[0].set_title('(A) Distribution Across All Runs', fontweight='bold')
        axes[0].set_yscale('log')
    
    # Panel B: Best results comparison
    best_results = {}
    for strat, label in zip(val_strategies, labels):
        if strat in df.columns:
            vals = df[strat].dropna()
            if len(vals) > 0:
                best_results[label] = {
                    'min': vals.min(),
                    'median': vals.median(),
                    'mean': vals.mean()
                }
    
    if best_results:
        x_pos = np.arange(len(best_results))
        width = 0.25
        
        mins = [v['min'] for v in best_results.values()]
        medians = [v['median'] for v in best_results.values()]
        means = [v['mean'] for v in best_results.values()]
        
        axes[1].bar(x_pos - width, mins, width, label='Best', color=PALETTE[0], alpha=0.8)
        axes[1].bar(x_pos, medians, width, label='Median', color=PALETTE[1], alpha=0.8)
        axes[1].bar(x_pos + width, means, width, label='Mean', color=PALETTE[2], alpha=0.8)
        
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(list(best_results.keys()))
        axes[1].set_ylabel('KEI-BO MSE')
        axes[1].set_title('(B) Summary Statistics', fontweight='bold')
        axes[1].legend()
        axes[1].set_yscale('log')
    
    plt.suptitle('Generalization Performance Across Validation Strategies', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_validation_comparison.png')
    plt.savefig(output_dir / 'fig2_validation_comparison.pdf')
    plt.close()
    print("Generated: fig2_validation_comparison.png/pdf")


def fig3_architecture_ablation_simple(df, output_dir):
    """Figure 3 (Simple): 2-panel architecture ablation for main paper."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    metric = 'overall/avg_val_kei_bo_mse'
    if metric not in df.columns or df[metric].isna().all():
        metric = 'random_split/avg_val_kei_bo_mse'
    
    # Panel A: Attention vs Non-attention
    attn_data = []
    attn_labels = []
    for use_attn in [True, False]:
        subset = df[df['use_attention'] == use_attn]
        if len(subset) > 0:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                attn_data.append(vals.values)
                attn_labels.append('With Attention' if use_attn else 'Without Attention')
    
    if attn_data:
        bp = axes[0].boxplot(attn_data, labels=attn_labels, patch_artist=True)
        colors = [COLORS['success'], COLORS['quaternary']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_ylabel('KEI-BO MSE')
        axes[0].set_title('(A) Edge-Level Attention', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Add statistical test
        if len(attn_data) == 2 and len(attn_data[0]) > 1 and len(attn_data[1]) > 1:
            from scipy import stats
            try:
                stat, p_val = stats.mannwhitneyu(attn_data[0], attn_data[1], alternative='two-sided')
                significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                axes[0].annotate(f'p={p_val:.2e} ({significance})', xy=(0.5, 0.95), 
                               xycoords='axes fraction', ha='center', fontsize=10)
            except:
                pass
    
    # Panel B: Number of layers
    layer_data = []
    layer_labels = []
    for n_layers in sorted(df['num_layers'].dropna().unique()):
        subset = df[df['num_layers'] == n_layers]
        if len(subset) > 0:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                layer_data.append(vals.values)
                layer_labels.append(f'{int(n_layers)} Layers')
    
    if layer_data:
        bp = axes[1].boxplot(layer_data, labels=layer_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], PALETTE[:len(layer_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_ylabel('KEI-BO MSE')
        axes[1].set_title('(B) Network Depth', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistical test
        if len(layer_data) == 2 and len(layer_data[0]) > 1 and len(layer_data[1]) > 1:
            from scipy import stats
            try:
                stat, p_val = stats.mannwhitneyu(layer_data[0], layer_data[1], alternative='two-sided')
                significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                axes[1].annotate(f'p={p_val:.2e} ({significance})', xy=(0.5, 0.95), 
                               xycoords='axes fraction', ha='center', fontsize=10)
            except:
                pass
    
    plt.suptitle('Architecture Ablation Study', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_architecture_simple.png')
    plt.savefig(output_dir / 'fig3_architecture_simple.pdf')
    plt.close()
    print("Generated: fig3_architecture_simple.png/pdf")


def fig3_architecture_ablation(df, output_dir):
    """Figure 3: Architecture ablation study."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metric = 'overall/avg_val_kei_bo_mse'
    if metric not in df.columns or df[metric].isna().all():
        metric = 'random_split/avg_val_kei_bo_mse'
    
    # Panel A: Attention vs Non-attention
    attn_data = []
    attn_labels = []
    for use_attn in [True, False]:
        subset = df[df['use_attention'] == use_attn]
        if len(subset) > 0:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                attn_data.append(vals.values)
                attn_labels.append('With Attention' if use_attn else 'Without Attention')
    
    if attn_data:
        bp = axes[0, 0].boxplot(attn_data, labels=attn_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], [PALETTE[0], PALETTE[1]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0, 0].set_ylabel('KEI-BO MSE')
        axes[0, 0].set_title('(A) Edge-Level Attention', fontweight='bold')
    
    # Panel B: Number of layers
    layer_data = []
    layer_labels = []
    for n_layers in sorted(df['num_layers'].dropna().unique()):
        subset = df[df['num_layers'] == n_layers]
        if len(subset) > 0:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                layer_data.append(vals.values)
                layer_labels.append(f'{int(n_layers)} layers')
    
    if layer_data:
        bp = axes[0, 1].boxplot(layer_data, labels=layer_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], PALETTE[:len(layer_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0, 1].set_ylabel('KEI-BO MSE')
        axes[0, 1].set_title('(B) Number of Message Passing Layers', fontweight='bold')
    
    # Panel C: Hidden dimension
    hidden_data = []
    hidden_labels = []
    for h_dim in sorted(df['hidden_dim'].dropna().unique()):
        subset = df[df['hidden_dim'] == h_dim]
        if len(subset) > 0:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                hidden_data.append(vals.values)
                hidden_labels.append(f'{int(h_dim)}D')
    
    if hidden_data:
        bp = axes[1, 0].boxplot(hidden_data, labels=hidden_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], PALETTE[:len(hidden_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 0].set_ylabel('KEI-BO MSE')
        axes[1, 0].set_title('(C) Hidden Dimension', fontweight='bold')
    
    # Panel D: Orbital embedding dimension
    embed_data = []
    embed_labels = []
    for e_dim in sorted(df['orbital_embedding_dim'].dropna().unique()):
        subset = df[df['orbital_embedding_dim'] == e_dim]
        if len(subset) > 0:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                embed_data.append(vals.values)
                embed_labels.append(f'{int(e_dim)}D')
    
    if embed_data:
        bp = axes[1, 1].boxplot(embed_data, labels=embed_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], PALETTE[:len(embed_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 1].set_ylabel('KEI-BO MSE')
        axes[1, 1].set_title('(D) Orbital Embedding Dimension', fontweight='bold')
    
    plt.suptitle('Architecture Ablation Study', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_architecture_ablation.png')
    plt.savefig(output_dir / 'fig3_architecture_ablation.pdf')
    plt.close()
    print("Generated: fig3_architecture_ablation.png/pdf")


def fig4_feature_importance(df, output_dir):
    """Figure 4: Feature importance analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metric = 'overall/avg_val_kei_bo_mse'
    if metric not in df.columns or df[metric].isna().all():
        metric = 'random_split/avg_val_kei_bo_mse'
    
    features = [
        ('use_element_baselines', 'Element Energy Baselines', axes[0, 0]),
        ('use_rbf_distance', 'RBF Distance Encoding', axes[0, 1]),
        ('include_hybridization', 'Hybridization Prediction', axes[0, 2]),
        ('include_orbital_type', 'Orbital Type Features', axes[1, 0]),
        ('include_m_quantum', 'm-Quantum Number', axes[1, 1]),
        ('use_physics_constraints', 'Physics Constraints', axes[1, 2])
    ]
    
    for feat_col, feat_name, ax in features:
        if feat_col in df.columns:
            data = []
            labels = []
            for val in [True, False]:
                subset = df[df[feat_col] == val]
                if len(subset) > 0:
                    vals = subset[metric].dropna()
                    if len(vals) > 0:
                        data.append(vals.values)
                        labels.append('Enabled' if val else 'Disabled')
            
            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                colors = [COLORS['success'], COLORS['quaternary']] if len(data) == 2 else [PALETTE[0]]
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_ylabel('KEI-BO MSE')
                ax.set_title(feat_name, fontweight='bold')
                
                # Add statistical annotation
                if len(data) == 2 and len(data[0]) > 1 and len(data[1]) > 1:
                    from scipy import stats
                    try:
                        stat, p_val = stats.mannwhitneyu(data[0], data[1], alternative='two-sided')
                        significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                        ax.annotate(f'p={p_val:.3f} ({significance})', xy=(0.5, 0.95), 
                                   xycoords='axes fraction', ha='center', fontsize=9)
                    except:
                        pass
    
    plt.suptitle('Impact of Physics-Informed Features', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_feature_importance.png')
    plt.savefig(output_dir / 'fig4_feature_importance.pdf')
    plt.close()
    print("Generated: fig4_feature_importance.png/pdf")


def fig5_pooling_comparison(df, output_dir):
    """Figure 5: Pooling method comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    pooling_methods = ['sum', 'mean', 'attention']
    metrics = [
        ('random_split/avg_val_kei_bo_mse', 'KEI-BO MSE'),
        ('random_split/avg_val_occupation_mse', 'Occupation MSE'),
        ('random_split/avg_val_energy_mse', 'Energy MSE')
    ]
    
    for ax_idx, (metric, ylabel) in enumerate(metrics):
        if metric not in df.columns:
            continue
            
        data = []
        labels = []
        for method in pooling_methods:
            subset = df[df['pooling_method'] == method]
            if len(subset) > 0:
                vals = subset[metric].dropna()
                if len(vals) > 0:
                    data.append(vals.values)
                    labels.append(method.capitalize())
        
        if data:
            bp = axes[ax_idx].boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], PALETTE[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[ax_idx].set_ylabel(ylabel)
            axes[ax_idx].set_title(f'{ylabel} by Pooling Method', fontweight='bold')
            axes[ax_idx].set_yscale('log')
    
    plt.suptitle('Global Pooling Method Comparison', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_pooling_comparison.png')
    plt.savefig(output_dir / 'fig5_pooling_comparison.pdf')
    plt.close()
    print("Generated: fig5_pooling_comparison.png/pdf")


def fig6_normalization_impact(df, output_dir):
    """Figure 6: Impact of normalization strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    metric = 'overall/avg_val_kei_bo_mse'
    if metric not in df.columns or df[metric].isna().all():
        metric = 'random_split/avg_val_kei_bo_mse'
    
    # Panel A: Normalization enabled vs disabled
    data = []
    labels = []
    for enabled in [True, False]:
        subset = df[df['normalization_enabled'] == enabled]
        if len(subset) > 0:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                data.append(vals.values)
                labels.append('Enabled' if enabled else 'Disabled')
    
    if data:
        bp = axes[0].boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], [COLORS['success'], COLORS['quaternary']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_ylabel('KEI-BO MSE')
        axes[0].set_title('(A) Normalization Impact', fontweight='bold')
    
    # Panel B: Global vs Per-fold normalization (among normalized runs)
    norm_df = df[df['normalization_enabled'] == True]
    
    data = []
    labels = []
    for is_global in [True, False]:
        subset = norm_df[norm_df['normalization_global'] == is_global]
        if len(subset) > 0:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                data.append(vals.values)
                labels.append('Global' if is_global else 'Per-Fold')
    
    if data:
        bp = axes[1].boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], PALETTE[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_ylabel('KEI-BO MSE')
        axes[1].set_title('(B) Global vs Per-Fold Normalization', fontweight='bold')
    
    plt.suptitle('Data Normalization Strategy Analysis', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_normalization.png')
    plt.savefig(output_dir / 'fig6_normalization.pdf')
    plt.close()
    print("Generated: fig6_normalization.png/pdf")


def fig7_hyperparameter_heatmap(df, output_dir):
    """Figure 7: Hyperparameter sensitivity heatmap."""
    metric = 'overall/avg_val_kei_bo_mse'
    if metric not in df.columns or df[metric].isna().all():
        metric = 'random_split/avg_val_kei_bo_mse'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Hidden dim vs Num layers
    pivot_data = df.pivot_table(
        values=metric, 
        index='hidden_dim', 
        columns='num_layers', 
        aggfunc='mean'
    )
    
    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                    ax=axes[0], cbar_kws={'label': 'KEI-BO MSE'})
        axes[0].set_title('(A) Hidden Dim × Num Layers', fontweight='bold')
        axes[0].set_xlabel('Number of Layers')
        axes[0].set_ylabel('Hidden Dimension')
    
    # Panel B: Orbital embedding dim vs RBF num
    pivot_data2 = df.pivot_table(
        values=metric, 
        index='orbital_embedding_dim', 
        columns='num_rbf', 
        aggfunc='mean'
    )
    
    if not pivot_data2.empty:
        sns.heatmap(pivot_data2, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                    ax=axes[1], cbar_kws={'label': 'KEI-BO MSE'})
        axes[1].set_title('(B) Embedding Dim × RBF Basis Functions', fontweight='bold')
        axes[1].set_xlabel('Number of RBF Basis Functions')
        axes[1].set_ylabel('Orbital Embedding Dimension')
    
    plt.suptitle('Hyperparameter Sensitivity Analysis', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_hyperparameter_heatmap.png')
    plt.savefig(output_dir / 'fig7_hyperparameter_heatmap.pdf')
    plt.close()
    print("Generated: fig7_hyperparameter_heatmap.png/pdf")


def fig8_energy_weight_analysis(df, output_dir):
    """Figure 8: Energy weight impact on multi-task learning."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Group by energy weight
    energy_weights = sorted(df['energy_weight'].dropna().unique())
    
    metrics = [
        ('random_split/avg_val_kei_bo_mse', 'KEI-BO MSE'),
        ('random_split/avg_val_occupation_mse', 'Occupation MSE'),
        ('random_split/avg_val_energy_mse', 'Energy MSE')
    ]
    
    for ax_idx, (metric, ylabel) in enumerate(metrics):
        if metric not in df.columns:
            continue
            
        means = []
        stds = []
        valid_weights = []
        
        for weight in energy_weights:
            subset = df[df['energy_weight'] == weight]
            if len(subset) > 0:
                vals = subset[metric].dropna()
                if len(vals) > 0:
                    means.append(vals.mean())
                    stds.append(vals.std())
                    valid_weights.append(weight)
        
        if means:
            axes[ax_idx].errorbar(valid_weights, means, yerr=stds, 
                                  marker='o', capsize=5, linewidth=2, markersize=8,
                                  color=PALETTE[ax_idx])
            axes[ax_idx].set_xlabel('Energy Weight')
            axes[ax_idx].set_ylabel(ylabel)
            axes[ax_idx].set_title(ylabel, fontweight='bold')
            axes[ax_idx].set_xscale('log')
            axes[ax_idx].set_yscale('log')
            axes[ax_idx].grid(True, alpha=0.3)
    
    plt.suptitle('Impact of Energy Loss Weight on Multi-Task Performance', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig8_energy_weight.png')
    plt.savefig(output_dir / 'fig8_energy_weight.pdf')
    plt.close()
    print("Generated: fig8_energy_weight.png/pdf")


def fig9_rbf_analysis(df, output_dir):
    """Figure 9: RBF distance encoding analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    metric = 'overall/avg_val_kei_bo_mse'
    if metric not in df.columns or df[metric].isna().all():
        metric = 'random_split/avg_val_kei_bo_mse'
    
    # Panel A: RBF enabled vs disabled
    rbf_df = df[df['use_rbf_distance'].notna()]
    
    data = []
    labels = []
    for enabled in [True, False]:
        subset = rbf_df[rbf_df['use_rbf_distance'] == enabled]
        if len(subset) > 0:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                data.append(vals.values)
                labels.append('RBF Enabled' if enabled else 'Raw Distance')
    
    if data:
        bp = axes[0].boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], [COLORS['primary'], COLORS['secondary']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_ylabel('KEI-BO MSE')
        axes[0].set_title('(A) RBF vs Raw Distance', fontweight='bold')
    
    # Panel B: RBF cutoff analysis (among RBF-enabled runs)
    rbf_enabled = df[df['use_rbf_distance'] == True]
    
    cutoff_data = []
    cutoff_labels = []
    for cutoff in sorted(rbf_enabled['rbf_cutoff'].dropna().unique()):
        subset = rbf_enabled[rbf_enabled['rbf_cutoff'] == cutoff]
        if len(subset) > 0:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                cutoff_data.append(vals.values)
                cutoff_labels.append(f'{cutoff} Å')
    
    if cutoff_data:
        bp = axes[1].boxplot(cutoff_data, labels=cutoff_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], PALETTE[:len(cutoff_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_ylabel('KEI-BO MSE')
        axes[1].set_title('(B) RBF Cutoff Distance', fontweight='bold')
    
    plt.suptitle('Radial Basis Function Distance Encoding Analysis', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig9_rbf_analysis.png')
    plt.savefig(output_dir / 'fig9_rbf_analysis.pdf')
    plt.close()
    print("Generated: fig9_rbf_analysis.png/pdf")


def fig10_pareto_frontier(df, output_dir):
    """Figure 10: Pareto frontier of accuracy vs model complexity."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    metric = 'overall/avg_val_kei_bo_mse'
    if metric not in df.columns or df[metric].isna().all():
        metric = 'random_split/avg_val_kei_bo_mse'
    
    # Estimate model complexity (rough parameter count proxy)
    df_plot = df.copy()
    df_plot['complexity'] = (df_plot['hidden_dim'] * df_plot['num_layers'] * 
                             df_plot['orbital_embedding_dim'] / 1000)  # Normalized
    
    # Remove NaN
    df_plot = df_plot.dropna(subset=[metric, 'complexity'])
    
    if len(df_plot) > 0:
        # Color by loss balancing strategy
        strategies = df_plot['loss_balancing_strategy'].unique()
        
        for i, strat in enumerate(strategies):
            subset = df_plot[df_plot['loss_balancing_strategy'] == strat]
            ax.scatter(subset['complexity'], subset[metric], 
                      label=strat.replace('_', ' ').title(),
                      alpha=0.6, s=60, c=PALETTE[i % len(PALETTE)])
        
        # Find Pareto frontier
        points = df_plot[[metric, 'complexity']].values
        pareto_mask = np.ones(len(points), dtype=bool)
        for i, (err, comp) in enumerate(points):
            for j, (err2, comp2) in enumerate(points):
                if i != j and err2 <= err and comp2 <= comp and (err2 < err or comp2 < comp):
                    pareto_mask[i] = False
                    break
        
        pareto_points = df_plot[pareto_mask].sort_values('complexity')
        if len(pareto_points) > 1:
            ax.plot(pareto_points['complexity'], pareto_points[metric], 
                   'k--', linewidth=2, label='Pareto Frontier', alpha=0.7)
        
        ax.set_xlabel('Model Complexity (hidden × layers × embed / 1000)')
        ax.set_ylabel('KEI-BO MSE')
        ax.set_title('Accuracy-Complexity Trade-off', fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig10_pareto.png')
    plt.savefig(output_dir / 'fig10_pareto.pdf')
    plt.close()
    print("Generated: fig10_pareto.png/pdf")


def table1_best_results(df, output_dir):
    """Generate LaTeX table of best results."""
    
    # Find best run for each validation strategy
    results = []
    
    val_strategies = [
        ('random_split/avg_val_kei_bo_mse', 'random_split/avg_val_occupation_mse', 
         'random_split/avg_val_energy_mse', 'Random Split'),
        ('per_molecule/avg_val_kei_bo_mse', None, None, 'Leave-One-Molecule-Out'),
        ('per_element/avg_val_kei_bo_mse', None, None, 'Leave-One-Element-Out')
    ]
    
    for kei_col, occ_col, eng_col, name in val_strategies:
        if kei_col in df.columns:
            valid = df[df[kei_col].notna()]
            if len(valid) > 0:
                best_idx = valid[kei_col].idxmin()
                best_run = valid.loc[best_idx]
                
                result = {
                    'Strategy': name,
                    'KEI-BO MSE': f"{best_run[kei_col]:.4f}",
                    'Run Name': best_run['Name'],
                    'Loss Balancing': best_run['loss_balancing_strategy'],
                    'Hidden Dim': int(best_run['hidden_dim']),
                    'Layers': int(best_run['num_layers']),
                    'Attention': best_run['use_attention'],
                    'RBF': best_run['use_rbf_distance'],
                }
                
                if occ_col and occ_col in df.columns:
                    result['Occ MSE'] = f"{best_run[occ_col]:.4f}"
                if eng_col and eng_col in df.columns:
                    result['Energy MSE'] = f"{best_run[eng_col]:.2f}"
                
                results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Save as CSV
    results_df.to_csv(output_dir / 'table1_best_results.csv', index=False)
    
    # Generate LaTeX
    latex = results_df.to_latex(index=False, escape=False, 
                                 column_format='l' + 'c' * (len(results_df.columns) - 1))
    
    with open(output_dir / 'table1_best_results.tex', 'w') as f:
        f.write(latex)
    
    print("Generated: table1_best_results.csv/tex")
    print("\nBest Results Summary:")
    print(results_df.to_string())
    
    return results_df


def table2_ablation_summary(df, output_dir):
    """Generate ablation summary table."""
    
    metric = 'overall/avg_val_kei_bo_mse'
    if metric not in df.columns or df[metric].isna().all():
        metric = 'random_split/avg_val_kei_bo_mse'
    
    ablations = []
    
    # Loss balancing
    for strat in df['loss_balancing_strategy'].unique():
        subset = df[df['loss_balancing_strategy'] == strat]
        if len(subset) > 0:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                ablations.append({
                    'Category': 'Loss Balancing',
                    'Setting': strat.replace('_', ' ').title(),
                    'Mean MSE': f"{vals.mean():.4f}",
                    'Std MSE': f"{vals.std():.4f}",
                    'Best MSE': f"{vals.min():.4f}",
                    'N': len(vals)
                })
    
    # Pooling method
    for method in df['pooling_method'].unique():
        subset = df[df['pooling_method'] == method]
        if len(subset) > 0:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                ablations.append({
                    'Category': 'Pooling',
                    'Setting': method.capitalize(),
                    'Mean MSE': f"{vals.mean():.4f}",
                    'Std MSE': f"{vals.std():.4f}",
                    'Best MSE': f"{vals.min():.4f}",
                    'N': len(vals)
                })
    
    # Boolean features
    bool_features = [
        ('use_attention', 'Attention'),
        ('use_rbf_distance', 'RBF Distance'),
        ('use_element_baselines', 'Element Baselines'),
        ('include_hybridization', 'Hybridization'),
        ('normalization_enabled', 'Normalization'),
    ]
    
    for col, name in bool_features:
        if col in df.columns:
            for val in [True, False]:
                subset = df[df[col] == val]
                if len(subset) > 0:
                    vals = subset[metric].dropna()
                    if len(vals) > 0:
                        ablations.append({
                            'Category': name,
                            'Setting': 'Enabled' if val else 'Disabled',
                            'Mean MSE': f"{vals.mean():.4f}",
                            'Std MSE': f"{vals.std():.4f}",
                            'Best MSE': f"{vals.min():.4f}",
                            'N': len(vals)
                        })
    
    ablation_df = pd.DataFrame(ablations)
    ablation_df.to_csv(output_dir / 'table2_ablation_summary.csv', index=False)
    
    # LaTeX version
    latex = ablation_df.to_latex(index=False, escape=False)
    with open(output_dir / 'table2_ablation_summary.tex', 'w') as f:
        f.write(latex)
    
    print("Generated: table2_ablation_summary.csv/tex")
    
    return ablation_df


def supplementary_stats(df, output_dir):
    """Generate supplementary statistics."""
    
    stats = {
        'Total Experiments': len(df),
        'Experiments with 200 epochs': len(df[df['epoch'] == 200]),
    }
    
    # Best overall results
    metric = 'overall/avg_val_kei_bo_mse'
    if metric in df.columns:
        valid = df[df[metric].notna()]
        if len(valid) > 0:
            best_idx = valid[metric].idxmin()
            stats['Best Overall KEI-BO MSE'] = f"{valid.loc[best_idx, metric]:.4f}"
            stats['Best Overall Run'] = valid.loc[best_idx, 'Name']
    
    # By validation strategy
    for col_name, display_name in [
        ('random_split/avg_val_kei_bo_mse', 'Random Split'),
        ('per_molecule/avg_val_kei_bo_mse', 'Per-Molecule'),
        ('per_element/avg_val_kei_bo_mse', 'Per-Element')
    ]:
        if col_name in df.columns:
            valid = df[df[col_name].notna()]
            if len(valid) > 0:
                stats[f'{display_name} - Best'] = f"{valid[col_name].min():.4f}"
                stats[f'{display_name} - Mean'] = f"{valid[col_name].mean():.4f}"
                stats[f'{display_name} - Median'] = f"{valid[col_name].median():.4f}"
    
    # Save stats
    with open(output_dir / 'supplementary_stats.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SUPPLEMENTARY STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print("\nSupplementary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("Generated: supplementary_stats.txt")


def main():
    """Main analysis pipeline."""
    # Setup
    input_file = Path('/mnt/user-data/uploads/wandb_export_2026-01-20T11_06_05_099-06_00.csv')
    output_dir = Path('/home/claude/paper_figures')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("ORBITAL-CENTRIC GNN PAPER FIGURE GENERATION")
    print("=" * 60)
    
    # Load data
    df = load_and_clean_data(input_file)
    
    print(f"\nDataset Overview:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Loss balancing strategies: {df['loss_balancing_strategy'].unique().tolist()}")
    print(f"  Pooling methods: {df['pooling_method'].unique().tolist()}")
    print(f"  Hidden dims: {sorted(df['hidden_dim'].dropna().unique().tolist())}")
    print(f"  Num layers: {sorted(df['num_layers'].dropna().unique().tolist())}")
    
    print("\n" + "-" * 60)
    print("Generating Figures...")
    print("-" * 60)
    
    # Generate all figures
    try:
        fig1_loss_balancing_comparison(df, output_dir)
    except Exception as e:
        print(f"Error in fig1: {e}")
    
    try:
        fig2_validation_strategy_comparison(df, output_dir)
    except Exception as e:
        print(f"Error in fig2: {e}")
    
    try:
        fig3_architecture_ablation_simple(df, output_dir)
    except Exception as e:
        print(f"Error in fig3_simple: {e}")
    
    try:
        fig3_architecture_ablation(df, output_dir)
    except Exception as e:
        print(f"Error in fig3: {e}")
    
    try:
        fig4_feature_importance(df, output_dir)
    except Exception as e:
        print(f"Error in fig4: {e}")
    
    try:
        fig5_pooling_comparison(df, output_dir)
    except Exception as e:
        print(f"Error in fig5: {e}")
    
    try:
        fig6_normalization_impact(df, output_dir)
    except Exception as e:
        print(f"Error in fig6: {e}")
    
    try:
        fig7_hyperparameter_heatmap(df, output_dir)
    except Exception as e:
        print(f"Error in fig7: {e}")
    
    try:
        fig8_energy_weight_analysis(df, output_dir)
    except Exception as e:
        print(f"Error in fig8: {e}")
    
    try:
        fig9_rbf_analysis(df, output_dir)
    except Exception as e:
        print(f"Error in fig9: {e}")
    
    try:
        fig10_pareto_frontier(df, output_dir)
    except Exception as e:
        print(f"Error in fig10: {e}")
    
    print("\n" + "-" * 60)
    print("Generating Tables...")
    print("-" * 60)
    
    try:
        table1_best_results(df, output_dir)
    except Exception as e:
        print(f"Error in table1: {e}")
    
    try:
        table2_ablation_summary(df, output_dir)
    except Exception as e:
        print(f"Error in table2: {e}")
    
    try:
        supplementary_stats(df, output_dir)
    except Exception as e:
        print(f"Error in supplementary stats: {e}")
    
    print("\n" + "=" * 60)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
