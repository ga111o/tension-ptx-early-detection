import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Set publication-quality parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4

# Color scheme - publication friendly (grayscale compatible)
COLOR_XGB = '#2563eb'  # Blue
COLOR_LGB = '#dc2626'  # Red

# XGBoost Top 5 Features
xgboost_features = [
    'w120_RR_kurt',
    'all_SHOCK_INDEX_kurt',
    'all_RR_kurt',
    'w120_SpO2_mad',
    'w120_DBP_kurt'
]
xgboost_importance = [
    0.041717,
    0.030034,
    0.029135,
    0.026487,
    0.024531
]

# LightGBM Top 5 Features
lightgbm_features = [
    'all_DBP_mad',
    'all_SHOCK_INDEX_mad',
    'all_RR_kurt',
    'all_SHOCK_INDEX_kurt',
    'w120_MAP_rmssd'
]
lightgbm_importance = [
    1890.921264,
    1755.765925,
    1451.095946,
    1299.409242,
    1242.025727
]

# Feature name mapping for better readability
feature_labels = {
    'w120_RR_kurt': 'RR Kurtosis (2h)',
    'all_SHOCK_INDEX_kurt': 'Shock Index Kurtosis',
    'all_RR_kurt': 'RR Kurtosis (All)',
    'w120_SpO2_mad': 'SpO₂ MAD (2h)',
    'w120_DBP_kurt': 'DBP Kurtosis (2h)',
    'all_DBP_mad': 'DBP MAD (All)',
    'all_SHOCK_INDEX_mad': 'Shock Index MAD',
    'w120_MAP_rmssd': 'MAP RMSSD (2h)',
}

# ============================================================================
# Figure 1: Side-by-side comparison (A and B panels)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel A - XGBoost
y_labels_xgb = [feature_labels.get(f, f) for f in xgboost_features]
y_pos = np.arange(len(xgboost_features))

bars_xgb = axes[0].barh(y_pos, xgboost_importance, height=0.6, 
                        color=COLOR_XGB, edgecolor='black', linewidth=0.8)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(y_labels_xgb, fontsize=9)
axes[0].set_xlabel('Feature Importance', fontsize=10, fontweight='bold')
axes[0].set_title('(A) XGBoost', fontsize=11, fontweight='bold', loc='left', pad=10)
axes[0].invert_yaxis()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
axes[0].set_axisbelow(True)

# Add value labels
for i, (bar, v) in enumerate(zip(bars_xgb, xgboost_importance)):
    axes[0].text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8, 
                fontweight='normal')

# Panel B - LightGBM
y_labels_lgb = [feature_labels.get(f, f) for f in lightgbm_features]

bars_lgb = axes[1].barh(y_pos, lightgbm_importance, height=0.6,
                        color=COLOR_LGB, edgecolor='black', linewidth=0.8)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(y_labels_lgb, fontsize=9)
axes[1].set_xlabel('Feature Importance', fontsize=10, fontweight='bold')
axes[1].set_title('(B) LightGBM', fontsize=11, fontweight='bold', loc='left', pad=10)
axes[1].invert_yaxis()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
axes[1].set_axisbelow(True)

# Add value labels
for i, (bar, v) in enumerate(zip(bars_lgb, lightgbm_importance)):
    axes[1].text(v + 30, i, f'{v:.0f}', va='center', fontsize=8, 
                fontweight='normal')

plt.tight_layout()
plt.savefig('figures/feature_importance_top5.png', dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('figures/feature_importance_top5.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Saved: figures/feature_importance_top5.png (600 DPI)")
print("✓ Saved: figures/feature_importance_top5.pdf (vector)")
plt.close()

# ============================================================================
# Figure 2: Combined normalized comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Normalize importance scores
xgb_normalized = np.array(xgboost_importance) / max(xgboost_importance) * 100
lgb_normalized = np.array(lightgbm_importance) / max(lightgbm_importance) * 100

# Use XGBoost feature order and labels
combined_labels = y_labels_xgb
y_pos = np.arange(len(xgboost_features))
bar_height = 0.35

# Create grouped bars
bars1 = ax.barh(y_pos - bar_height/2, xgb_normalized, bar_height,
               label='XGBoost', color=COLOR_XGB, edgecolor='black', linewidth=0.8)
bars2 = ax.barh(y_pos + bar_height/2, lgb_normalized[:len(xgboost_features)], bar_height,
               label='LightGBM', color=COLOR_LGB, edgecolor='black', linewidth=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(combined_labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Normalized Importance (%)', fontsize=10, fontweight='bold')
ax.set_title('Feature Importance Comparison', fontsize=11, fontweight='bold', pad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
ax.set_xlim(0, 110)

# Add legend
legend = ax.legend(loc='lower right', frameon=True, edgecolor='black', 
                  fancybox=False, fontsize=9, framealpha=1)
legend.get_frame().set_linewidth(1.2)

plt.tight_layout()
plt.savefig('figures/feature_importance_comparison.png', dpi=600, bbox_inches='tight',
           facecolor='white', edgecolor='none')
plt.savefig('figures/feature_importance_comparison.pdf', bbox_inches='tight',
           facecolor='white', edgecolor='none')
print("✓ Saved: figures/feature_importance_comparison.png (600 DPI)")
print("✓ Saved: figures/feature_importance_comparison.pdf (vector)")
plt.close()

# Print summary
print("\n" + "="*70)
print("TOP 5 FEATURE IMPORTANCE SUMMARY")
print("="*70)
print("\nXGBoost:")
for i, (feat, imp) in enumerate(zip(xgboost_features, xgboost_importance), 1):
    print(f"  {i}. {feature_labels.get(feat, feat):30s} {imp:.6f}")

print("\nLightGBM:")
for i, (feat, imp) in enumerate(zip(lightgbm_features, lightgbm_importance), 1):
    print(f"  {i}. {feature_labels.get(feat, feat):30s} {imp:.2f}")
print("="*70)
print("\nNote: Figures saved in PNG (600 DPI) and PDF (vector) formats")
print("      for publication-quality submission.")

