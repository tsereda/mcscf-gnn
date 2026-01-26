#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Orbital-Centric GNN Paper
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from scipy import stats
from typing import Dict, List, Tuple, Any
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

# Configuration
plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif', 'axes.labelsize': 11,
    'axes.titlesize': 12, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
})

COLORS = {'blue': '#0077BB', 'orange': '#EE7733', 'teal': '#009988',
          'red': '#CC3311', 'purple': '#AA3377', 'gray': '#BBBBBB', 'green': '#228833'}
PALETTE = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#AA3377', '#BBBBBB']
FIG_DOUBLE = 7.0


def load_data(filepath: Path) -> pd.DataFrame:
    """Load and clean WandB export data."""
    df = pd.read_csv(filepath)
    bool_cols = ['use_physics_constraints', 'use_attention', 'include_hybridization',
                 'include_m_quantum', 'include_orbital_type', 'normalization_enabled',
                 'normalization_global', 'use_element_baselines', 'use_rbf_distance']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({'true': True, 'false': False, True: True, False: False})
    numeric_cols = ['energy_weight', 'hidden_dim', 'num_layers', 'num_rbf', 
                    'occupation_weight', 'orbital_embedding_dim', 'rbf_cutoff',
                    'epoch', 'overall/avg_val_kei_bo_mse', 'random_split/avg_val_kei_bo_mse',
                    'per_molecule/avg_val_kei_bo_mse', 'per_element/avg_val_kei_bo_mse',
                    'random_split/avg_val_occupation_mse', 'random_split/avg_val_energy_mse']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    original_len = len(df)
    df = df[df['epoch'] >= 50]
    if 'per_element/avg_val_kei_bo_mse' in df.columns:
        df = df[df['per_element/avg_val_kei_bo_mse'] < 10]
    print(f"Loaded {len(df)} valid experiments (filtered from {original_len})")
    return df


def mannwhitney_test(a, b):
    a, b = np.array(a), np.array(b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan
    try:
        stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        effect = 1 - (2 * stat) / (len(a) * len(b))
        return stat, p, effect
    except:
        return np.nan, np.nan, np.nan


def kruskal_test(groups):
    groups = [np.array(g)[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) < 2:
        return np.nan, np.nan
    try:
        stat, p = stats.kruskal(*groups)
        return stat, p
    except:
        return np.nan, np.nan


def significance_stars(p):
    if pd.isna(p): return ''
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return 'ns'


class MetricsCollector:
    def __init__(self):
        self.metrics = {}
        self.stat_tests = {}
    
    def add(self, key, value, description=""):
        self.metrics[key] = {'value': value, 'description': description}
    
    def add_test(self, name, stat, p, effect=None, group_a="", group_b=""):
        self.stat_tests[name] = {
            'statistic': stat, 'p_value': p, 'effect_size': effect,
            'significant': p < 0.05 if not pd.isna(p) else None,
            'stars': significance_stars(p), 'groups': f"{group_a} vs {group_b}"
        }
    
    def save(self, output_dir):
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)): return float(obj)
            elif isinstance(obj, (np.bool_, bool)): return bool(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            return obj
        with open(output_dir / 'paper_metrics.json', 'w') as f:
            json.dump(convert(self.metrics), f, indent=2)
        with open(output_dir / 'statistical_tests.json', 'w') as f:
            json.dump(convert(self.stat_tests), f, indent=2)


def fig2_validation_comparison(df, output_dir, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE, 3.5))
    strategies = [('random_split/avg_val_kei_bo_mse', 'Random\nSplit'),
                  ('per_molecule/avg_val_kei_bo_mse', 'Leave-One-\nMolecule'),
                  ('per_element/avg_val_kei_bo_mse', 'Leave-One-\nElement')]
    data, labels, stats_data = [], [], {}
    for col, label in strategies:
        if col in df.columns:
            vals = df[col].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(label)
                stats_data[label] = {'best': float(np.min(vals)), 'mean': float(np.mean(vals)),
                                     'median': float(np.median(vals)), 'std': float(np.std(vals)), 'n': len(vals)}
                clean_label = label.replace('\n', '_').replace('-', '_')
                metrics.add(f'{clean_label}_best_kei_bo', np.min(vals))
                metrics.add(f'{clean_label}_mean_kei_bo', np.mean(vals))
    
    bp = axes[0].boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], PALETTE[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel('KEI-BO MSE')
    axes[0].set_title('(A) Distribution Across All Runs', fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    x = np.arange(len(labels))
    width = 0.25
    bests = [stats_data[l]['best'] for l in labels]
    means = [stats_data[l]['mean'] for l in labels]
    medians = [stats_data[l]['median'] for l in labels]
    axes[1].bar(x - width, bests, width, label='Best', color=COLORS['blue'], alpha=0.8)
    axes[1].bar(x, medians, width, label='Median', color=COLORS['orange'], alpha=0.8)
    axes[1].bar(x + width, means, width, label='Mean', color=COLORS['teal'], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('KEI-BO MSE')
    axes[1].set_title('(B) Summary Statistics', fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, b in enumerate(bests):
        axes[1].annotate(f'{b:.3f}', (i - width, b), ha='center', va='bottom', fontsize=7, rotation=45)
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(output_dir / f'fig2_validation_comparison.{fmt}')
    plt.close()
    print("  Generated: fig2_validation_comparison")
    return stats_data


def fig3_loss_balancing(df, output_dir, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_DOUBLE + 2, 3.5))
    strategies = ['static', 'first_epoch', 'uncertainty_weighting', 'gradnorm']
    strategy_labels = ['Static', 'First-Epoch', 'Uncertainty', 'GradNorm']
    metric_configs = [('random_split/avg_val_kei_bo_mse', 'KEI-BO MSE', 'Random Split'),
                      ('random_split/avg_val_occupation_mse', 'Occupation MSE', 'Random Split'),
                      ('per_element/avg_val_kei_bo_mse', 'KEI-BO MSE', 'Leave-One-Element')]
    
    for ax_idx, (metric_col, ylabel, title) in enumerate(metric_configs):
        if metric_col not in df.columns: continue
        data, valid_labels = [], []
        for strat, label in zip(strategies, strategy_labels):
            vals = df[df['loss_balancing_strategy'] == strat][metric_col].dropna().values
            if len(vals) > 0:
                data.append(vals)
                valid_labels.append(label)
        if data:
            bp = axes[ax_idx].boxplot(data, labels=valid_labels, patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], PALETTE[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[ax_idx].set_ylabel(ylabel)
            axes[ax_idx].set_title(f'({chr(65+ax_idx)}) {title}', fontweight='bold')
            axes[ax_idx].tick_params(axis='x', rotation=30)
            axes[ax_idx].grid(True, alpha=0.3, axis='y')
            h_stat, p_val = kruskal_test(data)
            if not pd.isna(p_val):
                axes[ax_idx].annotate(f'H={h_stat:.1f}, p={p_val:.3f} ({significance_stars(p_val)})',
                                      xy=(0.5, 0.98), xycoords='axes fraction', ha='center', va='top', fontsize=8)
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(output_dir / f'fig3_loss_balancing.{fmt}')
    plt.close()
    print("  Generated: fig3_loss_balancing")


def fig4_architecture_ablation(df, output_dir, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_DOUBLE + 2, 3.5))
    metric = 'random_split/avg_val_kei_bo_mse'
    
    # Panel A: Attention
    attn_data, attn_labels = [], []
    for use_attn, label in [(True, 'With\nAttention'), (False, 'Without\nAttention')]:
        vals = df[df['use_attention'] == use_attn][metric].dropna().values
        if len(vals) > 0:
            attn_data.append(vals)
            attn_labels.append(label)
    if len(attn_data) == 2:
        bp = axes[0].boxplot(attn_data, labels=attn_labels, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(COLORS['green'])
        bp['boxes'][1].set_facecolor(COLORS['red'])
        for box in bp['boxes']: box.set_alpha(0.7)
        stat, p, effect = mannwhitney_test(attn_data[0], attn_data[1])
        axes[0].annotate(f'p={p:.3f} ({significance_stars(p)})', xy=(0.5, 0.98),
                        xycoords='axes fraction', ha='center', va='top', fontsize=9)
        metrics.add('attention_mean_mse', np.mean(attn_data[0]))
        metrics.add('no_attention_mean_mse', np.mean(attn_data[1]))
        metrics.add_test('attention_vs_no_attention', stat, p, effect, 'With Attention', 'Without Attention')
    axes[0].set_ylabel('KEI-BO MSE')
    axes[0].set_title('(A) Edge-Level Attention', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Panel B: Layers
    layer_data, layer_labels = [], []
    for n_layers in sorted(df['num_layers'].dropna().unique()):
        vals = df[df['num_layers'] == n_layers][metric].dropna().values
        if len(vals) > 0:
            layer_data.append(vals)
            layer_labels.append(f'{int(n_layers)}')
    if layer_data:
        bp = axes[1].boxplot(layer_data, labels=layer_labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], PALETTE[:len(layer_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        if len(layer_data) == 2:
            stat, p, effect = mannwhitney_test(layer_data[0], layer_data[1])
            axes[1].annotate(f'p={p:.3f} ({significance_stars(p)})', xy=(0.5, 0.98),
                            xycoords='axes fraction', ha='center', va='top', fontsize=9)
            metrics.add_test('layers_3_vs_5', stat, p, effect, '3 layers', '5 layers')
    axes[1].set_ylabel('KEI-BO MSE')
    axes[1].set_xlabel('Number of Layers')
    axes[1].set_title('(B) Network Depth', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Panel C: Hidden dim
    hidden_data, hidden_labels = [], []
    for h_dim in sorted(df['hidden_dim'].dropna().unique()):
        vals = df[df['hidden_dim'] == h_dim][metric].dropna().values
        if len(vals) > 0:
            hidden_data.append(vals)
            hidden_labels.append(f'{int(h_dim)}')
    if hidden_data:
        bp = axes[2].boxplot(hidden_data, labels=hidden_labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], PALETTE[:len(hidden_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        if len(hidden_data) == 2:
            stat, p, effect = mannwhitney_test(hidden_data[0], hidden_data[1])
            axes[2].annotate(f'p={p:.3f} ({significance_stars(p)})', xy=(0.5, 0.98),
                            xycoords='axes fraction', ha='center', va='top', fontsize=9)
            metrics.add_test('hidden_32_vs_64', stat, p, effect, '32D', '64D')
    axes[2].set_ylabel('KEI-BO MSE')
    axes[2].set_xlabel('Hidden Dimension')
    axes[2].set_title('(C) Hidden Dimension', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(output_dir / f'fig4_architecture_ablation.{fmt}')
    plt.close()
    print("  Generated: fig4_architecture_ablation")


def fig5_feature_importance(df, output_dir, metrics):
    fig, axes = plt.subplots(2, 3, figsize=(FIG_DOUBLE + 2, 6))
    axes = axes.flatten()
    metric = 'random_split/avg_val_kei_bo_mse'
    features = [('include_m_quantum', 'm-Quantum Number'), ('use_rbf_distance', 'RBF Distance'),
                ('use_element_baselines', 'Element Baselines'), ('include_hybridization', 'Hybridization'),
                ('include_orbital_type', 'Orbital Type'), ('use_physics_constraints', 'Physics Constraints')]
    
    for ax_idx, (col, name) in enumerate(features):
        if col not in df.columns:
            axes[ax_idx].set_visible(False)
            continue
        data, labels = [], []
        for val, label in [(True, 'Enabled'), (False, 'Disabled')]:
            vals = df[df[col] == val][metric].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(label)
        if len(data) == 2:
            bp = axes[ax_idx].boxplot(data, labels=labels, patch_artist=True, widths=0.5)
            bp['boxes'][0].set_facecolor(COLORS['green'])
            bp['boxes'][1].set_facecolor(COLORS['red'])
            for box in bp['boxes']: box.set_alpha(0.7)
            stat, p, effect = mannwhitney_test(data[0], data[1])
            axes[ax_idx].annotate(f'p={p:.3f} ({significance_stars(p)})', xy=(0.5, 0.98),
                                  xycoords='axes fraction', ha='center', va='top', fontsize=8)
            metrics.add_test(f'feature_{col}', stat, p, effect, 'Enabled', 'Disabled')
        axes[ax_idx].set_ylabel('KEI-BO MSE')
        axes[ax_idx].set_title(name, fontweight='bold')
        axes[ax_idx].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(output_dir / f'fig5_feature_importance.{fmt}')
    plt.close()
    print("  Generated: fig5_feature_importance")


def fig6_pareto_frontier(df, output_dir, metrics):
    fig, ax = plt.subplots(figsize=(5, 4))
    metric = 'random_split/avg_val_kei_bo_mse'
    df_plot = df.copy()
    df_plot['complexity'] = (df_plot['hidden_dim'] * df_plot['num_layers'] * df_plot['orbital_embedding_dim'] / 1000)
    df_plot = df_plot.dropna(subset=[metric, 'complexity'])
    
    for i, strat in enumerate(df_plot['loss_balancing_strategy'].unique()):
        subset = df_plot[df_plot['loss_balancing_strategy'] == strat]
        ax.scatter(subset['complexity'], subset[metric], label=strat.replace('_', ' ').title(),
                   alpha=0.6, s=40, c=PALETTE[i % len(PALETTE)], edgecolors='white', linewidth=0.5)
    
    points = df_plot[[metric, 'complexity']].values
    pareto_mask = np.ones(len(points), dtype=bool)
    for i, (err, comp) in enumerate(points):
        for j, (err2, comp2) in enumerate(points):
            if i != j and err2 <= err and comp2 <= comp and (err2 < err or comp2 < comp):
                pareto_mask[i] = False
                break
    pareto_points = df_plot[pareto_mask].sort_values('complexity')
    if len(pareto_points) > 1:
        ax.plot(pareto_points['complexity'], pareto_points[metric], 'k--', linewidth=2, label='Pareto Frontier', alpha=0.7)
        ax.scatter(pareto_points['complexity'], pareto_points[metric], c='black', s=80, marker='*', zorder=11)
    
    ax.set_xlabel('Model Complexity (hidden × layers × embed / 1000)')
    ax.set_ylabel('KEI-BO MSE')
    ax.set_title('Accuracy vs Complexity Trade-off', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    metrics.add('n_pareto_optimal', int(pareto_mask.sum()))
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(output_dir / f'fig6_pareto_frontier.{fmt}')
    plt.close()
    print("  Generated: fig6_pareto_frontier")


def figS1_pooling_comparison(df, output_dir, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_DOUBLE + 2, 3.5))
    pooling_methods = ['sum', 'mean', 'attention']
    metric_configs = [('random_split/avg_val_kei_bo_mse', 'KEI-BO MSE'),
                      ('random_split/avg_val_occupation_mse', 'Occupation MSE'),
                      ('random_split/avg_val_energy_mse', 'Energy MSE')]
    for ax_idx, (metric_col, ylabel) in enumerate(metric_configs):
        if metric_col not in df.columns: continue
        data, labels = [], []
        for method in pooling_methods:
            vals = df[df['pooling_method'] == method][metric_col].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(method.capitalize())
        if data:
            bp = axes[ax_idx].boxplot(data, labels=labels, patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], PALETTE[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            h_stat, p_val = kruskal_test(data)
            if not pd.isna(p_val):
                axes[ax_idx].annotate(f'H={h_stat:.1f}, p={p_val:.3f} ({significance_stars(p_val)})',
                                      xy=(0.5, 0.98), xycoords='axes fraction', ha='center', va='top', fontsize=8)
        axes[ax_idx].set_ylabel(ylabel)
        axes[ax_idx].set_title(f'({chr(65+ax_idx)}) {ylabel}', fontweight='bold')
        axes[ax_idx].grid(True, alpha=0.3, axis='y')
        if 'energy' in metric_col.lower(): axes[ax_idx].set_yscale('log')
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(output_dir / f'figS1_pooling_comparison.{fmt}')
    plt.close()
    print("  Generated: figS1_pooling_comparison")


def figS2_normalization_impact(df, output_dir, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE, 3.5))
    metric = 'random_split/avg_val_kei_bo_mse'
    data, labels = [], []
    for enabled, label in [(True, 'Enabled'), (False, 'Disabled')]:
        vals = df[df['normalization_enabled'] == enabled][metric].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(label)
    if len(data) == 2:
        bp = axes[0].boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(COLORS['green'])
        bp['boxes'][1].set_facecolor(COLORS['red'])
        for box in bp['boxes']: box.set_alpha(0.7)
        stat, p, _ = mannwhitney_test(data[0], data[1])
        axes[0].annotate(f'p={p:.3f} ({significance_stars(p)})', xy=(0.5, 0.98),
                        xycoords='axes fraction', ha='center', va='top', fontsize=9)
    axes[0].set_ylabel('KEI-BO MSE')
    axes[0].set_title('(A) Normalization Impact', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    norm_df = df[df['normalization_enabled'] == True]
    data, labels = [], []
    for is_global, label in [(True, 'Global'), (False, 'Per-Fold')]:
        vals = norm_df[norm_df['normalization_global'] == is_global][metric].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(label)
    if data:
        bp = axes[1].boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], PALETTE[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    axes[1].set_ylabel('KEI-BO MSE')
    axes[1].set_title('(B) Global vs Per-Fold', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(output_dir / f'figS2_normalization_impact.{fmt}')
    plt.close()
    print("  Generated: figS2_normalization_impact")


def figS3_hyperparameter_heatmaps(df, output_dir, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE, 4))
    metric = 'random_split/avg_val_kei_bo_mse'
    pivot1 = df.pivot_table(values=metric, index='hidden_dim', columns='num_layers', aggfunc='mean')
    if not pivot1.empty:
        sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[0], cbar_kws={'label': 'KEI-BO MSE'})
        axes[0].set_title('(A) Hidden Dim × Layers', fontweight='bold')
    pivot2 = df.pivot_table(values=metric, index='orbital_embedding_dim', columns='num_rbf', aggfunc='mean')
    if not pivot2.empty:
        sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[1], cbar_kws={'label': 'KEI-BO MSE'})
        axes[1].set_title('(B) Embedding Dim × RBF', fontweight='bold')
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(output_dir / f'figS3_hyperparameter_heatmaps.{fmt}')
    plt.close()
    print("  Generated: figS3_hyperparameter_heatmaps")


def figS4_energy_weight_analysis(df, output_dir, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_DOUBLE + 2, 3.5))
    energy_weights = sorted(df['energy_weight'].dropna().unique())
    metrics_config = [('random_split/avg_val_kei_bo_mse', 'KEI-BO MSE'),
                      ('random_split/avg_val_occupation_mse', 'Occupation MSE'),
                      ('random_split/avg_val_energy_mse', 'Energy MSE')]
    for ax_idx, (metric_col, ylabel) in enumerate(metrics_config):
        if metric_col not in df.columns: continue
        means, stds, valid_weights = [], [], []
        for weight in energy_weights:
            vals = df[df['energy_weight'] == weight][metric_col].dropna().values
            if len(vals) > 0:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                valid_weights.append(weight)
        if means:
            axes[ax_idx].errorbar(valid_weights, means, yerr=stds, marker='o', capsize=4, color=PALETTE[ax_idx])
            axes[ax_idx].set_xlabel('Energy Weight')
            axes[ax_idx].set_ylabel(ylabel)
            axes[ax_idx].set_title(f'({chr(65+ax_idx)}) {ylabel}', fontweight='bold')
            axes[ax_idx].set_xscale('log')
            if 'energy' in metric_col.lower(): axes[ax_idx].set_yscale('log')
            axes[ax_idx].grid(True, alpha=0.3)
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(output_dir / f'figS4_energy_weight_analysis.{fmt}')
    plt.close()
    print("  Generated: figS4_energy_weight_analysis")


def figS5_rbf_analysis(df, output_dir, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE, 3.5))
    metric = 'random_split/avg_val_kei_bo_mse'
    data, labels = [], []
    for enabled, label in [(True, 'RBF'), (False, 'Raw Distance')]:
        vals = df[df['use_rbf_distance'] == enabled][metric].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(label)
    if len(data) == 2:
        bp = axes[0].boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(COLORS['blue'])
        bp['boxes'][1].set_facecolor(COLORS['orange'])
        for box in bp['boxes']: box.set_alpha(0.7)
        stat, p, _ = mannwhitney_test(data[0], data[1])
        axes[0].annotate(f'p={p:.3f} ({significance_stars(p)})', xy=(0.5, 0.98),
                        xycoords='axes fraction', ha='center', va='top', fontsize=9)
    axes[0].set_ylabel('KEI-BO MSE')
    axes[0].set_title('(A) RBF vs Raw Distance', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    rbf_df = df[df['use_rbf_distance'] == True]
    data, labels = [], []
    for cutoff in sorted(rbf_df['rbf_cutoff'].dropna().unique()):
        vals = rbf_df[rbf_df['rbf_cutoff'] == cutoff][metric].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(f'{cutoff} Å')
    if data:
        bp = axes[1].boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], PALETTE[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    axes[1].set_ylabel('KEI-BO MSE')
    axes[1].set_title('(B) RBF Cutoff Distance', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(output_dir / f'figS5_rbf_analysis.{fmt}')
    plt.close()
    print("  Generated: figS5_rbf_analysis")


def figS6_mse_distributions(df, output_dir, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_DOUBLE + 2, 3.5))
    strategies = [('random_split/avg_val_kei_bo_mse', 'Random Split'),
                  ('per_molecule/avg_val_kei_bo_mse', 'Leave-Molecule'),
                  ('per_element/avg_val_kei_bo_mse', 'Leave-Element')]
    for ax_idx, (metric_col, title) in enumerate(strategies):
        if metric_col not in df.columns: continue
        vals = df[metric_col].dropna()
        axes[ax_idx].hist(vals, bins=30, color=PALETTE[ax_idx], alpha=0.7, edgecolor='black')
        axes[ax_idx].axvline(vals.min(), color='red', linestyle='--', linewidth=2, label=f'Best: {vals.min():.4f}')
        axes[ax_idx].axvline(vals.mean(), color='black', linestyle=':', linewidth=2, label=f'Mean: {vals.mean():.4f}')
        axes[ax_idx].set_xlabel('KEI-BO MSE')
        axes[ax_idx].set_ylabel('Count')
        axes[ax_idx].set_title(f'({chr(65+ax_idx)}) {title}', fontweight='bold')
        axes[ax_idx].legend(fontsize=8)
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        plt.savefig(output_dir / f'figS6_mse_distributions.{fmt}')
    plt.close()
    print("  Generated: figS6_mse_distributions")


# Tables
def table1_best_results(df, output_dir, metrics):
    results = []
    strategies = [('random_split/avg_val_kei_bo_mse', 'random_split/avg_val_occupation_mse', 'Random Split'),
                  ('per_molecule/avg_val_kei_bo_mse', None, 'Leave-One-Molecule-Out'),
                  ('per_element/avg_val_kei_bo_mse', None, 'Leave-One-Element-Out')]
    for kei_col, occ_col, name in strategies:
        if kei_col not in df.columns: continue
        valid = df[df[kei_col].notna()]
        if len(valid) == 0: continue
        best_idx = valid[kei_col].idxmin()
        best_run = valid.loc[best_idx]
        result = {'Validation Strategy': name, 'KEI-BO MSE': f"{best_run[kei_col]:.4f}",
                  'Run Name': best_run['Name'], 'Loss Strategy': best_run['loss_balancing_strategy'],
                  'Hidden': int(best_run['hidden_dim']), 'Layers': int(best_run['num_layers']),
                  'Attention': 'Yes' if best_run['use_attention'] else 'No'}
        if occ_col and occ_col in df.columns and pd.notna(best_run.get(occ_col)):
            result['Occ. MSE'] = f"{best_run[occ_col]:.4f}"
        results.append(result)
        metrics.add(f'best_{name.replace(" ", "_")}_kei_bo', best_run[kei_col])
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'table1_best_results.csv', index=False)
    results_df.to_latex(output_dir / 'table1_best_results.tex', index=False, escape=True)
    print("  Generated: table1_best_results")
    return results_df


def table2_loss_balancing(df, output_dir, metrics):
    results = []
    metric = 'random_split/avg_val_kei_bo_mse'
    for strat in ['static', 'first_epoch', 'uncertainty_weighting', 'gradnorm']:
        subset = df[df['loss_balancing_strategy'] == strat]
        vals = subset[metric].dropna()
        if len(vals) == 0: continue
        per_elem_col = 'per_element/avg_val_kei_bo_mse'
        elem_vals = subset[per_elem_col].dropna() if per_elem_col in subset.columns else pd.Series()
        success_rate = (elem_vals < 0.15).mean() * 100 if len(elem_vals) > 0 else np.nan
        failure_rate = (elem_vals > 0.5).mean() * 100 if len(elem_vals) > 0 else np.nan
        results.append({'Strategy': strat.replace('_', ' ').title(), 'Mean MSE': f"{vals.mean():.4f}",
                       'Std': f"{vals.std():.4f}", 'Best': f"{vals.min():.4f}", 'N': len(vals),
                       'Success Rate (%)': f"{success_rate:.1f}" if not pd.isna(success_rate) else '-'})
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'table2_loss_balancing.csv', index=False)
    results_df.to_latex(output_dir / 'table2_loss_balancing.tex', index=False, escape=True)
    print("  Generated: table2_loss_balancing")
    return results_df


def table3_ablation_summary(df, output_dir, metrics):
    results = []
    metric = 'random_split/avg_val_kei_bo_mse'
    for col, category in [('loss_balancing_strategy', 'Loss Balancing'), ('pooling_method', 'Pooling')]:
        for val in df[col].unique():
            vals = df[df[col] == val][metric].dropna()
            if len(vals) > 0:
                results.append({'Category': category, 'Setting': str(val).replace('_', ' ').title(),
                               'Mean': f"{vals.mean():.4f}", 'Std': f"{vals.std():.4f}",
                               'Best': f"{vals.min():.4f}", 'N': len(vals)})
    for col, category in [('use_attention', 'Attention'), ('use_rbf_distance', 'RBF Distance'),
                          ('include_m_quantum', 'm-Quantum'), ('normalization_enabled', 'Normalization')]:
        if col not in df.columns: continue
        for val in [True, False]:
            vals = df[df[col] == val][metric].dropna()
            if len(vals) > 0:
                results.append({'Category': category, 'Setting': 'Enabled' if val else 'Disabled',
                               'Mean': f"{vals.mean():.4f}", 'Std': f"{vals.std():.4f}",
                               'Best': f"{vals.min():.4f}", 'N': len(vals)})
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'table3_ablation_summary.csv', index=False)
    results_df.to_latex(output_dir / 'table3_ablation_summary.tex', index=False, escape=True)
    print("  Generated: table3_ablation_summary")
    return results_df


def tableS1_hyperparameter_space(df, output_dir):
    params = [('num_layers', 'Message Passing Layers'), ('hidden_dim', 'Hidden Dimension'),
              ('orbital_embedding_dim', 'Orbital Embedding Dim'), ('num_rbf', 'RBF Basis Functions'),
              ('rbf_cutoff', 'Cutoff Radius'), ('energy_weight', 'Energy Weight'),
              ('loss_balancing_strategy', 'Loss Balancing'), ('pooling_method', 'Pooling Method')]
    results = []
    for col, name in params:
        if col not in df.columns: continue
        unique_vals = df[col].dropna().unique()
        values = ', '.join(sorted([str(v) for v in unique_vals])) if len(unique_vals) <= 10 else f"{df[col].min():.4g} - {df[col].max():.4g}"
        results.append({'Parameter': name, 'Values': values})
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'tableS1_hyperparameter_space.csv', index=False)
    results_df.to_latex(output_dir / 'tableS1_hyperparameter_space.tex', index=False, escape=True)
    print("  Generated: tableS1_hyperparameter_space")


def tableS2_best_configurations(df, output_dir):
    metric = 'random_split/avg_val_kei_bo_mse'
    top10 = df.nsmallest(10, metric)
    cols = ['Name', metric, 'per_element/avg_val_kei_bo_mse', 'loss_balancing_strategy',
            'hidden_dim', 'num_layers', 'pooling_method', 'use_attention']
    cols = [c for c in cols if c in top10.columns]
    results_df = top10[cols].copy()
    results_df.columns = [c.split('/')[-1].replace('avg_val_', '').replace('_', ' ').title() for c in results_df.columns]
    results_df.to_csv(output_dir / 'tableS2_best_configurations.csv', index=False)
    print("  Generated: tableS2_best_configurations")


def generate_summary_report(df, metrics, output_dir):
    lines = ["=" * 70, "ORBITAL-CENTRIC GNN PAPER - ANALYSIS SUMMARY REPORT",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "=" * 70, "",
             "DATASET OVERVIEW", "-" * 40, f"Total experiments analyzed: {len(df)}",
             f"Loss balancing strategies: {df['loss_balancing_strategy'].unique().tolist()}",
             f"Pooling methods: {df['pooling_method'].unique().tolist()}", ""]
    
    lines.append("BEST RESULTS BY VALIDATION STRATEGY")
    lines.append("-" * 40)
    for col, name in [('random_split/avg_val_kei_bo_mse', 'Random Split'),
                      ('per_molecule/avg_val_kei_bo_mse', 'Leave-One-Molecule-Out'),
                      ('per_element/avg_val_kei_bo_mse', 'Leave-One-Element-Out')]:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                best_idx = vals.idxmin()
                best_run = df.loc[best_idx]
                lines.extend([f"\n{name}:", f"  Best KEI-BO MSE: {vals.min():.4f}",
                             f"  Mean KEI-BO MSE: {vals.mean():.4f}", f"  Best run: {best_run['Name']}",
                             f"    - Loss strategy: {best_run['loss_balancing_strategy']}",
                             f"    - Hidden dim: {int(best_run['hidden_dim'])}",
                             f"    - Layers: {int(best_run['num_layers'])}",
                             f"    - Attention: {best_run['use_attention']}"])
    
    lines.extend(["", "STATISTICAL TESTS SUMMARY", "-" * 40])
    for name, result in metrics.stat_tests.items():
        sig = "SIGNIFICANT" if result['significant'] else "not significant"
        lines.extend([f"\n{name}: {sig}", f"  p-value: {result['p_value']:.4f} ({result['stars']})"])
        if result['effect_size']: lines.append(f"  effect size: {result['effect_size']:.3f}")
    
    lines.extend(["", "=" * 70, "END OF REPORT", "=" * 70])
    report = "\n".join(lines)
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write(report)
    print("\n" + report)


def main(input_file):
    input_path = Path(input_file)
    output_base = Path('./paper_outputs')
    dirs = {
        'figures': output_base / 'figures',
        'figures_supp': output_base / 'figures_supplementary',
        'tables': output_base / 'tables',
        'metrics': output_base / 'metrics'
    }
    # Create all output directories
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ORBITAL-CENTRIC GNN PAPER - COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print(f"\nInput: {input_path}\nOutput: {output_base}\n")

    df = load_data(input_path)
    metrics = MetricsCollector()
    metrics.add('n_experiments', len(df))

    print("\n" + "-" * 40 + "\nGENERATING MAIN TEXT FIGURES\n" + "-" * 40)
    fig2_validation_comparison(df, dirs['figures'], metrics)
    fig3_loss_balancing(df, dirs['figures'], metrics)
    fig4_architecture_ablation(df, dirs['figures'], metrics)
    fig5_feature_importance(df, dirs['figures'], metrics)
    fig6_pareto_frontier(df, dirs['figures'], metrics)

    print("\n" + "-" * 40 + "\nGENERATING SUPPLEMENTARY FIGURES\n" + "-" * 40)
    figS1_pooling_comparison(df, dirs['figures_supp'], metrics)
    figS2_normalization_impact(df, dirs['figures_supp'], metrics)
    figS3_hyperparameter_heatmaps(df, dirs['figures_supp'], metrics)
    figS4_energy_weight_analysis(df, dirs['figures_supp'], metrics)
    figS5_rbf_analysis(df, dirs['figures_supp'], metrics)
    figS6_mse_distributions(df, dirs['figures_supp'], metrics)

    print("\n" + "-" * 40 + "\nGENERATING TABLES\n" + "-" * 40)
    table1_best_results(df, dirs['tables'], metrics)
    table2_loss_balancing(df, dirs['tables'], metrics)
    table3_ablation_summary(df, dirs['tables'], metrics)
    tableS1_hyperparameter_space(df, dirs['tables'])
    tableS2_best_configurations(df, dirs['tables'])

    print("\n" + "-" * 40 + "\nSAVING METRICS\n" + "-" * 40)
    metrics.save(dirs['metrics'])
    print("  Saved: paper_metrics.json\n  Saved: statistical_tests.json")

    print("\n" + "-" * 40 + "\nGENERATING SUMMARY REPORT\n" + "-" * 40)
    generate_summary_report(df, metrics, dirs['metrics'])

    print("\n" + "=" * 70 + "\nANALYSIS COMPLETE\n" + "=" * 70)
    print(f"\nAll outputs saved to: {output_base}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'wandb_export_2026-01-25T21_24_38.674-06_00.csv'
    main(input_file)