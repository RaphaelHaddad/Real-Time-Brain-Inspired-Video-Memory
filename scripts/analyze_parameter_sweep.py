#!/usr/bin/env python3
"""
Comprehensive Parameter Sweep Analysis for Video Memory KG Pipeline
====================================================================

This script performs in-depth statistical analysis and generates publication-quality
visualizations from parameter sweep experiments.

Usage:
    python scripts/analyze_parameter_sweep.py --input outputs/sweeps/sweep_20251202_230641/sweep_summary.json
    python scripts/analyze_parameter_sweep.py --input outputs/sweeps/sweep_20251202_230641/sweep_summary.json --output-dir outputs/analysis

Author: Generated for conference presentation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
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
    'light': '#E9ECEF',
}

PARAM_NAMES = {
    'max_connection_subgraph': 'Max Subgraph\nConnections',
    'max_new_triplets': 'Max New\nTriplets',
    'max_inter_chunk_relations': 'Max Inter-Chunk\nRelations',
    'max_merge_instructions': 'Max Merge\nInstructions',
    'max_prune_instructions': 'Max Prune\nInstructions',
}

PARAM_SHORT = {
    'max_connection_subgraph': 'Subgraph',
    'max_new_triplets': 'Triplets',
    'max_inter_chunk_relations': 'Inter-Chunk',
    'max_merge_instructions': 'Merge',
    'max_prune_instructions': 'Prune',
}

METRIC_NAMES = {
    'accuracy': 'Retrieval Accuracy',
    'avg_retrieval_time': 'Avg Retrieval Time (s)',
    'kg_build_time': 'KG Build Time (s)',
}


def load_sweep_data(filepath: str) -> pd.DataFrame:
    """Load sweep summary JSON and convert to DataFrame."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Flatten the data
    rows = []
    for r in results:
        if r['status'] != 'success':
            continue
        row = {
            'epoch': r['epoch'],
            'graph_uuid': r['graph_uuid'],
            'accuracy': r['accuracy'],
            'avg_retrieval_time': r['avg_retrieval_time'],
            'kg_build_time': r['kg_build_time'],
        }
        row.update(r['parameters'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def compute_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute descriptive statistics for all metrics and parameters."""
    params = ['max_connection_subgraph', 'max_new_triplets', 'max_inter_chunk_relations',
              'max_merge_instructions', 'max_prune_instructions']
    metrics = ['accuracy', 'avg_retrieval_time', 'kg_build_time']
    
    stats_dict = {
        'parameters': {},
        'metrics': {},
        'n_samples': len(df),
    }
    
    for col in params:
        stats_dict['parameters'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'unique_values': sorted(df[col].unique().tolist()),
        }
    
    for col in metrics:
        stats_dict['metrics'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median(),
            'q25': df[col].quantile(0.25),
            'q75': df[col].quantile(0.75),
        }
    
    return stats_dict


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix between parameters and metrics."""
    params = ['max_connection_subgraph', 'max_new_triplets', 'max_inter_chunk_relations',
              'max_merge_instructions', 'max_prune_instructions']
    metrics = ['accuracy', 'avg_retrieval_time', 'kg_build_time']
    
    cols = params + metrics
    corr_matrix = df[cols].corr(method='spearman')
    
    return corr_matrix


def compute_parameter_importance(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute feature importance for each metric using Random Forest."""
    params = ['max_connection_subgraph', 'max_new_triplets', 'max_inter_chunk_relations',
              'max_merge_instructions', 'max_prune_instructions']
    metrics = ['accuracy', 'avg_retrieval_time', 'kg_build_time']
    
    X = df[params].values
    importance_results = {}
    
    for metric in metrics:
        y = df[metric].values
        
        # Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Permutation importance for more robust estimates
        perm_imp = permutation_importance(rf, X, y, n_repeats=30, random_state=42)
        
        importance_df = pd.DataFrame({
            'parameter': params,
            'rf_importance': rf.feature_importances_,
            'perm_importance_mean': perm_imp.importances_mean,
            'perm_importance_std': perm_imp.importances_std,
        })
        importance_df = importance_df.sort_values('perm_importance_mean', ascending=False)
        importance_results[metric] = importance_df
    
    return importance_results


def perform_regression_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform multiple linear regression for each metric."""
    params = ['max_connection_subgraph', 'max_new_triplets', 'max_inter_chunk_relations',
              'max_merge_instructions', 'max_prune_instructions']
    metrics = ['accuracy', 'avg_retrieval_time', 'kg_build_time']
    
    results = {}
    
    for metric in metrics:
        X = df[params]
        X = sm.add_constant(X)
        y = df[metric]
        
        model = sm.OLS(y, X).fit()
        
        results[metric] = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'coefficients': model.params.to_dict(),
            'pvalues': model.pvalues.to_dict(),
            'std_errors': model.bse.to_dict(),
            'conf_int': model.conf_int().to_dict(),
        }
    
    return results


def find_optimal_configurations(df: pd.DataFrame) -> Dict[str, Any]:
    """Find optimal parameter configurations for different objectives."""
    params = ['max_connection_subgraph', 'max_new_triplets', 'max_inter_chunk_relations',
              'max_merge_instructions', 'max_prune_instructions']
    
    results = {}
    
    # Best accuracy
    best_acc_idx = df['accuracy'].idxmax()
    results['best_accuracy'] = {
        'epoch': int(df.loc[best_acc_idx, 'epoch']),
        'accuracy': df.loc[best_acc_idx, 'accuracy'],
        'avg_retrieval_time': df.loc[best_acc_idx, 'avg_retrieval_time'],
        'kg_build_time': df.loc[best_acc_idx, 'kg_build_time'],
        'parameters': {p: int(df.loc[best_acc_idx, p]) for p in params},
    }
    
    # Fastest retrieval
    best_ret_idx = df['avg_retrieval_time'].idxmin()
    results['fastest_retrieval'] = {
        'epoch': int(df.loc[best_ret_idx, 'epoch']),
        'accuracy': df.loc[best_ret_idx, 'accuracy'],
        'avg_retrieval_time': df.loc[best_ret_idx, 'avg_retrieval_time'],
        'kg_build_time': df.loc[best_ret_idx, 'kg_build_time'],
        'parameters': {p: int(df.loc[best_ret_idx, p]) for p in params},
    }
    
    # Fastest KG build
    best_build_idx = df['kg_build_time'].idxmin()
    results['fastest_build'] = {
        'epoch': int(df.loc[best_build_idx, 'epoch']),
        'accuracy': df.loc[best_build_idx, 'accuracy'],
        'avg_retrieval_time': df.loc[best_build_idx, 'avg_retrieval_time'],
        'kg_build_time': df.loc[best_build_idx, 'kg_build_time'],
        'parameters': {p: int(df.loc[best_build_idx, p]) for p in params},
    }
    
    # Pareto optimal: high accuracy + fast retrieval
    # Normalize metrics
    df_norm = df.copy()
    df_norm['acc_norm'] = (df['accuracy'] - df['accuracy'].min()) / (df['accuracy'].max() - df['accuracy'].min())
    df_norm['ret_norm'] = 1 - (df['avg_retrieval_time'] - df['avg_retrieval_time'].min()) / (df['avg_retrieval_time'].max() - df['avg_retrieval_time'].min())
    df_norm['build_norm'] = 1 - (df['kg_build_time'] - df['kg_build_time'].min()) / (df['kg_build_time'].max() - df['kg_build_time'].min())
    
    # Combined score (weighted)
    df_norm['combined_score'] = 0.5 * df_norm['acc_norm'] + 0.3 * df_norm['ret_norm'] + 0.2 * df_norm['build_norm']
    best_combined_idx = df_norm['combined_score'].idxmax()
    
    results['best_balanced'] = {
        'epoch': int(df.loc[best_combined_idx, 'epoch']),
        'accuracy': df.loc[best_combined_idx, 'accuracy'],
        'avg_retrieval_time': df.loc[best_combined_idx, 'avg_retrieval_time'],
        'kg_build_time': df.loc[best_combined_idx, 'kg_build_time'],
        'combined_score': df_norm.loc[best_combined_idx, 'combined_score'],
        'parameters': {p: int(df.loc[best_combined_idx, p]) for p in params},
    }
    
    # Top 5 by accuracy
    top5 = df.nlargest(5, 'accuracy')[['epoch', 'accuracy', 'avg_retrieval_time', 'kg_build_time'] + params]
    results['top5_accuracy'] = top5.to_dict('records')
    
    return results


def perform_anova_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform ANOVA/Kruskal-Wallis tests for categorical parameter effects."""
    params = ['max_connection_subgraph', 'max_new_triplets', 'max_inter_chunk_relations',
              'max_merge_instructions', 'max_prune_instructions']
    metrics = ['accuracy', 'avg_retrieval_time', 'kg_build_time']
    
    results = {}
    
    for metric in metrics:
        results[metric] = {}
        for param in params:
            # Create bins for continuous parameters
            unique_vals = df[param].nunique()
            
            if unique_vals <= 5:
                # Use actual values
                groups = [df[df[param] == v][metric].values for v in df[param].unique()]
            else:
                # Bin into quartiles
                df['_bin'] = pd.qcut(df[param], q=4, duplicates='drop')
                groups = [df[df['_bin'] == v][metric].values for v in df['_bin'].unique()]
                df.drop('_bin', axis=1, inplace=True)
            
            # Filter out empty groups
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                # Kruskal-Wallis (non-parametric)
                try:
                    h_stat, p_value = kruskal(*groups)
                    results[metric][param] = {
                        'h_statistic': h_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                    }
                except:
                    results[metric][param] = {'error': 'Could not compute'}
    
    return results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create correlation heatmap between parameters and metrics."""
    params = ['max_connection_subgraph', 'max_new_triplets', 'max_inter_chunk_relations',
              'max_merge_instructions', 'max_prune_instructions']
    metrics = ['accuracy', 'avg_retrieval_time', 'kg_build_time']
    
    cols = params + metrics
    corr = df[cols].corr(method='spearman')
    
    # Rename for display
    rename_dict = {**PARAM_SHORT, **{'accuracy': 'Accuracy', 'avg_retrieval_time': 'Ret. Time', 'kg_build_time': 'Build Time'}}
    corr = corr.rename(index=rename_dict, columns=rename_dict)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt='.2f',
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Spearman ρ'},
                ax=ax, vmin=-1, vmax=1)
    
    ax.set_title('Parameter-Metric Correlation Matrix\n(Spearman Correlation)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'correlation_heatmap.png', dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved correlation_heatmap.png")


def plot_parameter_importance(importance_results: Dict, output_dir: Path):
    """Plot parameter importance for each metric."""
    metrics = ['accuracy', 'avg_retrieval_time', 'kg_build_time']
    metric_titles = ['Retrieval Accuracy', 'Avg Retrieval Time', 'KG Build Time']
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for ax, metric, title, color in zip(axes, metrics, metric_titles, colors):
        imp_df = importance_results[metric]
        
        y_pos = np.arange(len(imp_df))
        bars = ax.barh(y_pos, imp_df['perm_importance_mean'], 
                       xerr=imp_df['perm_importance_std'],
                       color=color, alpha=0.8, capsize=3)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([PARAM_SHORT.get(p, p) for p in imp_df['parameter']])
        ax.set_xlabel('Permutation Importance')
        ax.set_title(title, fontweight='bold')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    fig.suptitle('Parameter Importance Analysis (Random Forest + Permutation)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'parameter_importance.png', dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved parameter_importance.png")


def plot_metric_distributions(df: pd.DataFrame, output_dir: Path):
    """Plot distributions of all metrics."""
    metrics = ['accuracy', 'avg_retrieval_time', 'kg_build_time']
    titles = ['Retrieval Accuracy', 'Avg Retrieval Time (s)', 'KG Build Time (s)']
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        data = df[metric]
        
        # Histogram with KDE
        ax.hist(data, bins=15, color=color, alpha=0.6, edgecolor='white', density=True)
        
        # KDE overlay
        kde_x = np.linspace(data.min(), data.max(), 100)
        kde = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), color=color, linewidth=2, label='KDE')
        
        # Statistics annotations
        mean_val = data.mean()
        std_val = data.std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.3f}')
        ax.axvline(data.median(), color='green', linestyle=':', linewidth=1.5, label=f'Median: {data.median():.3f}')
        
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.set_title(f'Distribution of {title}', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'metric_distributions.png', dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved metric_distributions.png")


def plot_parameter_effects(df: pd.DataFrame, output_dir: Path):
    """Plot effect of each parameter on accuracy with trend lines."""
    params = ['max_connection_subgraph', 'max_new_triplets', 'max_inter_chunk_relations',
              'max_merge_instructions', 'max_prune_instructions']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        ax = axes[i]
        
        x = df[param]
        y = df['accuracy']
        
        # Scatter plot
        scatter = ax.scatter(x, y, c=df['avg_retrieval_time'], cmap='viridis', 
                            alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
        
        # Trend line (LOWESS)
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            z = lowess(y, x, frac=0.6)
            ax.plot(z[:, 0], z[:, 1], color='red', linewidth=2, label='LOWESS trend')
        except:
            # Fallback to linear
            slope, intercept, r, p, se = stats.linregress(x, y)
            ax.plot(x, slope * x + intercept, 'r-', linewidth=2, 
                   label=f'Linear (r={r:.3f})')
        
        # Add correlation
        r, p = spearmanr(x, y)
        ax.set_title(f'{PARAM_SHORT[param]}\nρ = {r:.3f} (p={p:.3f})', fontweight='bold')
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Accuracy')
        ax.legend(fontsize=8)
    
    # Remove extra subplot
    axes[5].axis('off')
    
    # Add colorbar to last position
    cbar = fig.colorbar(scatter, ax=axes[5], orientation='vertical', fraction=0.5)
    cbar.set_label('Avg Retrieval Time (s)')
    
    fig.suptitle('Parameter Effects on Retrieval Accuracy\n(color = retrieval time)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'parameter_effects.png', dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved parameter_effects.png")


def plot_accuracy_vs_time_tradeoff(df: pd.DataFrame, optimal: Dict, output_dir: Path):
    """Plot accuracy vs retrieval time trade-off (Pareto frontier)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Main scatter
    scatter = ax.scatter(df['avg_retrieval_time'], df['accuracy'], 
                        c=df['kg_build_time'], cmap='plasma',
                        s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Highlight optimal configurations
    markers = {
        'best_accuracy': ('*', 'green', 'Best Accuracy'),
        'fastest_retrieval': ('D', 'blue', 'Fastest Retrieval'),
        'best_balanced': ('s', 'red', 'Best Balanced'),
    }
    
    for key, (marker, color, label) in markers.items():
        config = optimal[key]
        ax.scatter(config['avg_retrieval_time'], config['accuracy'], 
                  marker=marker, s=300, c=color, edgecolors='black', 
                  linewidth=2, label=label, zorder=10)
        
        # Annotate
        ax.annotate(f"Epoch {config['epoch']}", 
                   (config['avg_retrieval_time'], config['accuracy']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    # Pareto frontier
    pareto_mask = np.zeros(len(df), dtype=bool)
    sorted_df = df.sort_values('avg_retrieval_time')
    max_acc = -np.inf
    for idx in sorted_df.index:
        if df.loc[idx, 'accuracy'] > max_acc:
            pareto_mask[idx] = True
            max_acc = df.loc[idx, 'accuracy']
    
    pareto_df = df[pareto_mask].sort_values('avg_retrieval_time')
    ax.plot(pareto_df['avg_retrieval_time'], pareto_df['accuracy'], 
            'k--', linewidth=2, alpha=0.5, label='Pareto Frontier')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('KG Build Time (s)', fontsize=11)
    
    ax.set_xlabel('Average Retrieval Time (s)', fontsize=12)
    ax.set_ylabel('Retrieval Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Speed Trade-off Analysis\n(75 Parameter Configurations)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'accuracy_vs_time_tradeoff.png', dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved accuracy_vs_time_tradeoff.png")


def plot_pairwise_interactions(df: pd.DataFrame, output_dir: Path):
    """Plot pairwise parameter interactions for accuracy."""
    params = ['max_new_triplets', 'max_inter_chunk_relations', 
              'max_merge_instructions', 'max_prune_instructions']
    
    n_params = len(params)
    fig, axes = plt.subplots(n_params-1, n_params-1, figsize=(12, 12))
    
    for i in range(n_params - 1):
        for j in range(n_params - 1):
            ax = axes[i, j]
            
            if j <= i:
                # Lower triangle: scatter plots
                p1, p2 = params[j], params[i+1]
                scatter = ax.scatter(df[p1], df[p2], c=df['accuracy'], 
                                    cmap='RdYlGn', s=40, alpha=0.7,
                                    vmin=df['accuracy'].min(), 
                                    vmax=df['accuracy'].max())
                
                if i == n_params - 2:
                    ax.set_xlabel(PARAM_SHORT[p1], fontsize=9)
                if j == 0:
                    ax.set_ylabel(PARAM_SHORT[p2], fontsize=9)
            else:
                ax.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Accuracy', fontsize=11)
    
    fig.suptitle('Pairwise Parameter Interactions\n(color = accuracy)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    fig.savefig(output_dir / 'pairwise_interactions.png', dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved pairwise_interactions.png")


def plot_epoch_progression(df: pd.DataFrame, output_dir: Path):
    """Plot metrics over epochs to check for trends/drift."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    metrics = ['accuracy', 'avg_retrieval_time', 'kg_build_time']
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
    titles = ['Retrieval Accuracy', 'Avg Retrieval Time (s)', 'KG Build Time (s)']
    
    for ax, metric, color, title in zip(axes, metrics, colors, titles):
        x = df['epoch']
        y = df[metric]
        
        # Plot line with markers
        ax.plot(x, y, 'o-', color=color, alpha=0.7, markersize=4, linewidth=1)
        
        # Rolling average
        window = 10
        rolling_mean = y.rolling(window=window, center=True).mean()
        ax.plot(x, rolling_mean, color='red', linewidth=2, 
               label=f'{window}-epoch moving avg')
        
        # Overall mean
        ax.axhline(y.mean(), color='gray', linestyle='--', alpha=0.5, 
                  label=f'Overall mean: {y.mean():.3f}')
        
        ax.set_ylabel(title)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Epoch')
    axes[0].set_title('Metric Progression Across Epochs', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'epoch_progression.png', dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved epoch_progression.png")


def plot_boxplots_by_parameter(df: pd.DataFrame, output_dir: Path):
    """Create box plots showing accuracy distribution by parameter bins."""
    params = ['max_connection_subgraph', 'max_new_triplets', 'max_inter_chunk_relations',
              'max_merge_instructions', 'max_prune_instructions']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        ax = axes[i]
        
        # Bin parameter into categories
        unique_vals = df[param].nunique()
        if unique_vals <= 4:
            df['_cat'] = df[param].astype(str)
        else:
            df['_cat'] = pd.cut(df[param], bins=4, precision=0)
        
        # Box plot
        categories = sorted(df['_cat'].unique(), key=lambda x: float(str(x).split(',')[0].replace('(', '').replace('[', '')) if ',' in str(x) else float(x))
        
        box_data = [df[df['_cat'] == cat]['accuracy'].values for cat in categories]
        bp = ax.boxplot(box_data, patch_artist=True, labels=[str(c)[:10] for c in categories])
        
        # Color boxes
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['primary'])
            patch.set_alpha(0.6)
        
        ax.set_xlabel(PARAM_SHORT[param])
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy by {PARAM_SHORT[param]}', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        df.drop('_cat', axis=1, inplace=True)
    
    # Remove extra subplot
    axes[5].axis('off')
    
    fig.suptitle('Accuracy Distribution by Parameter Ranges', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'boxplots_by_parameter.png', dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved boxplots_by_parameter.png")


def plot_build_time_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze what affects KG build time."""
    params = ['max_connection_subgraph', 'max_new_triplets', 'max_inter_chunk_relations',
              'max_merge_instructions', 'max_prune_instructions']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        ax = axes[i]
        
        x = df[param]
        y = df['kg_build_time']
        
        ax.scatter(x, y, c=df['accuracy'], cmap='RdYlGn', s=50, alpha=0.7)
        
        # Trend line
        slope, intercept, r, p, se = stats.linregress(x, y)
        ax.plot(x, slope * x + intercept, 'r-', linewidth=2)
        
        ax.set_xlabel(PARAM_SHORT[param])
        ax.set_ylabel('KG Build Time (s)')
        ax.set_title(f'r = {r:.3f}, p = {p:.3f}', fontsize=10)
    
    axes[5].axis('off')
    
    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(df['accuracy'].min(), df['accuracy'].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[5], orientation='vertical', fraction=0.5)
    cbar.set_label('Accuracy')
    
    fig.suptitle('Parameters Affecting KG Build Time\n(color = accuracy)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'build_time_analysis.png', dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved build_time_analysis.png")


def create_conference_figure(df: pd.DataFrame, optimal: Dict, importance: Dict, 
                             regression: Dict, output_dir: Path):
    """Create the main conference-ready multi-panel figure."""
    
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel A: Accuracy vs Time Trade-off (large, top-left)
    # =========================================================================
    ax_tradeoff = fig.add_subplot(gs[0, :2])
    
    scatter = ax_tradeoff.scatter(df['avg_retrieval_time'], df['accuracy'], 
                                  c=df['kg_build_time'], cmap='plasma',
                                  s=100, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Highlight optimal
    best = optimal['best_accuracy']
    ax_tradeoff.scatter(best['avg_retrieval_time'], best['accuracy'], 
                       marker='*', s=400, c='lime', edgecolors='black', 
                       linewidth=2, label=f"Best Accuracy ({best['accuracy']:.1%})", zorder=10)
    
    balanced = optimal['best_balanced']
    ax_tradeoff.scatter(balanced['avg_retrieval_time'], balanced['accuracy'], 
                       marker='s', s=300, c='red', edgecolors='black', 
                       linewidth=2, label=f"Best Balanced", zorder=10)
    
    cbar = plt.colorbar(scatter, ax=ax_tradeoff, shrink=0.8)
    cbar.set_label('Build Time (s)')
    
    ax_tradeoff.set_xlabel('Average Retrieval Time (s)')
    ax_tradeoff.set_ylabel('Retrieval Accuracy')
    ax_tradeoff.set_title('A) Accuracy-Speed Trade-off Space (n=75)', fontweight='bold', fontsize=13)
    ax_tradeoff.legend(loc='lower left')
    ax_tradeoff.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel B: Parameter Importance (top-right)
    # =========================================================================
    ax_importance = fig.add_subplot(gs[0, 2:])
    
    imp_df = importance['accuracy']
    y_pos = np.arange(len(imp_df))
    bars = ax_importance.barh(y_pos, imp_df['perm_importance_mean'], 
                              xerr=imp_df['perm_importance_std'],
                              color=COLORS['primary'], alpha=0.8, capsize=4)
    
    ax_importance.set_yticks(y_pos)
    ax_importance.set_yticklabels([PARAM_SHORT.get(p, p) for p in imp_df['parameter']])
    ax_importance.set_xlabel('Permutation Importance')
    ax_importance.set_title('B) Parameter Importance for Accuracy', fontweight='bold', fontsize=13)
    ax_importance.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # =========================================================================
    # Panel C: Top 2 Parameter Effects (middle row)
    # =========================================================================
    top_params = importance['accuracy'].head(2)['parameter'].tolist()
    
    for idx, param in enumerate(top_params):
        ax = fig.add_subplot(gs[1, idx*2:(idx+1)*2])
        
        x = df[param]
        y = df['accuracy']
        
        scatter = ax.scatter(x, y, c=df['avg_retrieval_time'], cmap='viridis', 
                            s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        # LOWESS trend
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            z = lowess(y, x, frac=0.5)
            ax.plot(z[:, 0], z[:, 1], color='red', linewidth=3, label='LOWESS')
        except:
            pass
        
        r, p = spearmanr(x, y)
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Accuracy')
        ax.set_title(f'C{idx+1}) Effect of {PARAM_SHORT[param]} (ρ={r:.3f})', 
                    fontweight='bold', fontsize=13)
        
        if idx == 1:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Retrieval Time (s)')
    
    # =========================================================================
    # Panel D: Correlation Heatmap (bottom-left)
    # =========================================================================
    ax_corr = fig.add_subplot(gs[2, :2])
    
    params = ['max_connection_subgraph', 'max_new_triplets', 'max_inter_chunk_relations',
              'max_merge_instructions', 'max_prune_instructions']
    metrics = ['accuracy', 'avg_retrieval_time', 'kg_build_time']
    
    # Just parameter vs metrics correlations
    corr_data = []
    for metric in metrics:
        row = []
        for param in params:
            r, _ = spearmanr(df[param], df[metric])
            row.append(r)
        corr_data.append(row)
    
    corr_df = pd.DataFrame(corr_data, 
                          index=['Accuracy', 'Retrieval Time', 'Build Time'],
                          columns=[PARAM_SHORT[p] for p in params])
    
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax_corr, vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
    ax_corr.set_title('D) Parameter-Metric Correlations (Spearman ρ)', fontweight='bold', fontsize=13)
    
    # =========================================================================
    # Panel E: Distribution Summary (bottom-right)
    # =========================================================================
    ax_dist = fig.add_subplot(gs[2, 2:])
    
    # Violin plots for normalized metrics
    df_plot = df.copy()
    for metric in metrics:
        df_plot[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
    
    violin_data = [df_plot['accuracy_norm'], df_plot['avg_retrieval_time_norm'], df_plot['kg_build_time_norm']]
    parts = ax_dist.violinplot(violin_data, positions=[1, 2, 3], showmeans=True, showmedians=True)
    
    colors_violin = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_alpha(0.7)
    
    ax_dist.set_xticks([1, 2, 3])
    ax_dist.set_xticklabels(['Accuracy', 'Retrieval Time', 'Build Time'])
    ax_dist.set_ylabel('Normalized Value (0-1)')
    ax_dist.set_title('E) Metric Distributions (Normalized)', fontweight='bold', fontsize=13)
    
    # Add statistics box
    stats_text = f"""Best Accuracy: {optimal['best_accuracy']['accuracy']:.1%}
Best Config: Epoch {optimal['best_accuracy']['epoch']}
Mean Accuracy: {df['accuracy'].mean():.1%} ± {df['accuracy'].std():.1%}
Accuracy Range: [{df['accuracy'].min():.1%}, {df['accuracy'].max():.1%}]"""
    
    ax_dist.text(0.02, 0.98, stats_text, transform=ax_dist.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main title
    fig.suptitle('Video Memory Knowledge Graph: Parameter Sweep Analysis\n75 Configurations Evaluated',
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'conference_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'conference_figure.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved conference_figure.png/pdf")


def generate_summary_report(df: pd.DataFrame, stats: Dict, optimal: Dict, 
                           importance: Dict, regression: Dict, anova: Dict,
                           output_dir: Path):
    """Generate a comprehensive text/markdown summary report."""
    
    report = []
    report.append("=" * 80)
    report.append("PARAMETER SWEEP ANALYSIS REPORT")
    report.append("Video Memory Knowledge Graph Pipeline")
    report.append("=" * 80)
    report.append("")
    
    # Overview
    report.append("## 1. EXPERIMENT OVERVIEW")
    report.append("-" * 40)
    report.append(f"Total Configurations Tested: {stats['n_samples']}")
    report.append(f"Parameters Varied: 5")
    report.append("")
    
    # Parameter ranges
    report.append("### Parameter Ranges:")
    for param, pstats in stats['parameters'].items():
        report.append(f"  • {PARAM_SHORT.get(param, param)}: [{pstats['min']}, {pstats['max']}]")
    report.append("")
    
    # Metric statistics
    report.append("## 2. METRIC STATISTICS")
    report.append("-" * 40)
    for metric, mstats in stats['metrics'].items():
        report.append(f"\n### {METRIC_NAMES.get(metric, metric)}:")
        report.append(f"  Mean ± Std:  {mstats['mean']:.4f} ± {mstats['std']:.4f}")
        report.append(f"  Median:      {mstats['median']:.4f}")
        report.append(f"  Range:       [{mstats['min']:.4f}, {mstats['max']:.4f}]")
        report.append(f"  IQR:         [{mstats['q25']:.4f}, {mstats['q75']:.4f}]")
    report.append("")
    
    # Optimal configurations
    report.append("## 3. OPTIMAL CONFIGURATIONS")
    report.append("-" * 40)
    
    report.append("\n### Best Accuracy:")
    best = optimal['best_accuracy']
    report.append(f"  Epoch: {best['epoch']}")
    report.append(f"  Accuracy: {best['accuracy']:.2%}")
    report.append(f"  Retrieval Time: {best['avg_retrieval_time']:.2f}s")
    report.append(f"  Build Time: {best['kg_build_time']:.1f}s")
    report.append(f"  Parameters: {best['parameters']}")
    
    report.append("\n### Fastest Retrieval:")
    fast = optimal['fastest_retrieval']
    report.append(f"  Epoch: {fast['epoch']}")
    report.append(f"  Accuracy: {fast['accuracy']:.2%}")
    report.append(f"  Retrieval Time: {fast['avg_retrieval_time']:.2f}s")
    report.append(f"  Parameters: {fast['parameters']}")
    
    report.append("\n### Best Balanced (50% acc, 30% speed, 20% build):")
    bal = optimal['best_balanced']
    report.append(f"  Epoch: {bal['epoch']}")
    report.append(f"  Accuracy: {bal['accuracy']:.2%}")
    report.append(f"  Retrieval Time: {bal['avg_retrieval_time']:.2f}s")
    report.append(f"  Build Time: {bal['kg_build_time']:.1f}s")
    report.append(f"  Combined Score: {bal['combined_score']:.3f}")
    report.append(f"  Parameters: {bal['parameters']}")
    report.append("")
    
    # Parameter importance
    report.append("## 4. PARAMETER IMPORTANCE (for Accuracy)")
    report.append("-" * 40)
    imp_df = importance['accuracy']
    for _, row in imp_df.iterrows():
        sig = "***" if row['perm_importance_mean'] > 0.01 else ""
        report.append(f"  {PARAM_SHORT.get(row['parameter'], row['parameter']):15s}: "
                     f"{row['perm_importance_mean']:.4f} ± {row['perm_importance_std']:.4f} {sig}")
    report.append("")
    
    # Regression results
    report.append("## 5. REGRESSION ANALYSIS (Accuracy)")
    report.append("-" * 40)
    reg = regression['accuracy']
    report.append(f"  R² = {reg['r_squared']:.4f}, Adj R² = {reg['adj_r_squared']:.4f}")
    report.append(f"  F-statistic = {reg['f_statistic']:.2f}, p = {reg['f_pvalue']:.4e}")
    report.append("\n  Coefficients (significant at p<0.05 marked with *):")
    for param, coef in reg['coefficients'].items():
        if param == 'const':
            continue
        pval = reg['pvalues'][param]
        sig = "*" if pval < 0.05 else ""
        report.append(f"    {PARAM_SHORT.get(param, param):15s}: {coef:+.5f} (p={pval:.4f}) {sig}")
    report.append("")
    
    # Key findings
    report.append("## 6. KEY FINDINGS")
    report.append("-" * 40)
    
    # Find most impactful parameter
    top_param = importance['accuracy'].iloc[0]['parameter']
    top_corr, _ = spearmanr(df[top_param], df['accuracy'])
    
    report.append(f"\n1. Most influential parameter for accuracy: {PARAM_SHORT.get(top_param, top_param)}")
    report.append(f"   (Spearman ρ = {top_corr:.3f})")
    
    # Accuracy range
    acc_range = df['accuracy'].max() - df['accuracy'].min()
    report.append(f"\n2. Accuracy variation across configurations: {acc_range:.1%}")
    report.append(f"   This represents a {acc_range/df['accuracy'].mean()*100:.1f}% relative difference")
    
    # Trade-off insight
    acc_time_corr, _ = spearmanr(df['accuracy'], df['avg_retrieval_time'])
    report.append(f"\n3. Accuracy-Speed trade-off correlation: ρ = {acc_time_corr:.3f}")
    if abs(acc_time_corr) < 0.3:
        report.append("   → Weak correlation suggests independent optimization is possible")
    
    # Build time insight  
    build_param = importance['kg_build_time'].iloc[0]['parameter']
    report.append(f"\n4. Build time most affected by: {PARAM_SHORT.get(build_param, build_param)}")
    
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    # Save as text
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    # Save as markdown
    with open(output_dir / 'analysis_report.md', 'w') as f:
        f.write(report_text)
    
    print(f"  ✓ Saved analysis_report.txt/md")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Analyze parameter sweep results')
    parser.add_argument('--input', '-i', required=True, help='Path to sweep_summary.json')
    parser.add_argument('--output-dir', '-o', default=None, help='Output directory for plots and reports')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / 'analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PARAMETER SWEEP ANALYSIS")
    print("=" * 60)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    print("Loading data...")
    df = load_sweep_data(input_path)
    print(f"  Loaded {len(df)} successful configurations")
    print()
    
    # Statistical analysis
    print("Computing statistics...")
    stats = compute_descriptive_stats(df)
    print(f"  ✓ Descriptive statistics")
    
    importance = compute_parameter_importance(df)
    print(f"  ✓ Parameter importance (Random Forest)")
    
    regression = perform_regression_analysis(df)
    print(f"  ✓ Multiple regression")
    
    optimal = find_optimal_configurations(df)
    print(f"  ✓ Optimal configurations")
    
    anova = perform_anova_analysis(df)
    print(f"  ✓ ANOVA/Kruskal-Wallis tests")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_correlation_heatmap(df, output_dir)
    plot_parameter_importance(importance, output_dir)
    plot_metric_distributions(df, output_dir)
    plot_parameter_effects(df, output_dir)
    plot_accuracy_vs_time_tradeoff(df, optimal, output_dir)
    plot_pairwise_interactions(df, output_dir)
    plot_epoch_progression(df, output_dir)
    plot_boxplots_by_parameter(df, output_dir)
    plot_build_time_analysis(df, output_dir)
    print()
    
    # Create conference figure
    print("Creating conference figure...")
    create_conference_figure(df, optimal, importance, regression, output_dir)
    print()
    
    # Generate report
    print("Generating summary report...")
    report = generate_summary_report(df, stats, optimal, importance, regression, anova, output_dir)
    print()
    
    # Print summary to console
    print("=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"\nBest Accuracy: {optimal['best_accuracy']['accuracy']:.2%} (Epoch {optimal['best_accuracy']['epoch']})")
    print(f"  Parameters: {optimal['best_accuracy']['parameters']}")
    print(f"\nMean Accuracy: {df['accuracy'].mean():.2%} ± {df['accuracy'].std():.2%}")
    print(f"Accuracy Range: [{df['accuracy'].min():.2%}, {df['accuracy'].max():.2%}]")
    print(f"\nTop Parameter for Accuracy: {PARAM_SHORT.get(importance['accuracy'].iloc[0]['parameter'], importance['accuracy'].iloc[0]['parameter'])}")
    print(f"\nAll outputs saved to: {output_dir}")
    print()
    
    # Save raw results as JSON
    results_json = {
        'descriptive_stats': stats,
        'optimal_configurations': optimal,
        'parameter_importance': {k: v.to_dict('records') for k, v in importance.items()},
        'regression_results': regression,
        'anova_results': anova,
    }
    
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"  ✓ Saved analysis_results.json")


if __name__ == '__main__':
    main()
