# -*- coding: utf-8 -*-
"""
Enhanced Random Forest Classification Model
With comprehensive metrics, visualizations, and cross-validation
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    auc
)
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('dataset9000.csv')

print('='*80)
print('DATA OVERVIEW')
print('='*80)
print(f'\nDataset Shape: {df.shape}')
print(f'\nMissing Values:\n{df.isnull().sum()}')
print(f'\nDataframe Info:')
df.info()

# Visualize the distribution of job roles
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Role', order=df['Role'].value_counts().index, palette='viridis')
plt.title('Distribution of Job Roles', fontsize=14, fontweight='bold')
plt.xlabel('Role', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('role_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Prepare data
df_modeling = df.copy()

# Encode all categorical variables using LabelEncoder
le_dict = {}
for col in df_modeling.columns:
    if df_modeling[col].dtype == 'object':
        le = LabelEncoder()
        df_modeling[col] = le.fit_transform(df_modeling[col])
        le_dict[col] = le

# Separate features and target
X = df_modeling.drop('Role', axis=1)
y = df_modeling['Role']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f'\nTraining set size: {X_train.shape[0]}')
print(f'Test set size: {X_test.shape[0]}')

# Train Random Forest
print('\n' + '='*80)
print('RANDOM FOREST CLASSIFIER - TRAINING')
print('='*80)

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
clf.fit(X_train, y_train)

# Predictions
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Get prediction probabilities for ROC-AUC
y_test_pred_proba = clf.predict_proba(X_test)

# Basic Metrics
print('\n' + '='*80)
print('PERFORMANCE METRICS')
print('='*80)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
test_precision = precision_score(y_test, y_test_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f'\nTRAINING METRICS:')
print(f'  Accuracy:  {train_accuracy:.4f}')
print(f'  Precision: {train_precision:.4f}')
print(f'  Recall:    {train_recall:.4f}')
print(f'  F1-Score:  {train_f1:.4f}')

print(f'\nTEST METRICS:')
print(f'  Accuracy:  {test_accuracy:.4f}')
print(f'  Precision: {test_precision:.4f}')
print(f'  Recall:    {test_recall:.4f}')
print(f'  F1-Score:  {test_f1:.4f}')

# ROC-AUC Score
try:
    roc_auc = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='weighted')
    print(f'\nROC-AUC Score (OvR):     {roc_auc:.4f}')
except:
    print('\nROC-AUC Score: Not available for this dataset')

# Classification Report
print('\n' + '='*80)
print('DETAILED CLASSIFICATION REPORT')
print('='*80)
print(classification_report(y_test, y_test_pred))

# Cross-Validation with Multiple Folds
print('\n' + '='*80)
print('CROSS-VALIDATION ANALYSIS')
print('='*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

cv_results = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_train_score=True)

print('\nK-Fold Cross-Validation Results (k=5):')
print(f'  Accuracy:  {cv_results["test_accuracy"].mean():.4f} (+/- {cv_results["test_accuracy"].std():.4f})')
print(f'  Precision: {cv_results["test_precision_weighted"].mean():.4f} (+/- {cv_results["test_precision_weighted"].std():.4f})')
print(f'  Recall:    {cv_results["test_recall_weighted"].mean():.4f} (+/- {cv_results["test_recall_weighted"].std():.4f})')
print(f'  F1-Score:  {cv_results["test_f1_weighted"].mean():.4f} (+/- {cv_results["test_f1_weighted"].std():.4f})')

# Visualize Cross-Validation Scores
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    scores = cv_results[f'test_{metric}']
    ax.bar(range(1, 6), scores, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(y=scores.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.4f}')
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'{title} across Folds', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cross_validation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion Matrix
print('\n' + '='*80)
print('CONFUSION MATRIX')
print('='*80)

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=le_dict['Role'].classes_, yticklabels=le_dict['Role'].classes_)
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance
print('\n' + '='*80)
print('FEATURE IMPORTANCE')
print('='*80)

feature_importances = clf.feature_importances_
perm_importance = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)

print('\nTop 10 Most Important Features:')
print(perm_importance.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x=perm_importance.head(15).values, y=perm_importance.head(15).index, 
            palette='rocket', orient='h')
plt.title('Top 15 Feature Importances - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC Curve (One-vs-Rest for multiclass)
print('\n' + '='*80)
print('ROC CURVE ANALYSIS')
print('='*80)

n_classes = len(np.unique(y_test))
if n_classes > 2:
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_classes):
        y_test_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_test_bin, y_test_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{le_dict["Role"].classes_[i]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - One-vs-Rest Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# Model Comparison Summary
print('\n' + '='*80)
print('MODEL SUMMARY - RANDOM FOREST')
print('='*80)
print(f'\nModel: RandomForestClassifier')
print(f'Number of Trees: 100')
print(f'Train/Test Split: 80/20')
print(f'Cross-Validation Folds: 5')
print(f'\nBest Metrics:')
print(f'  Test Accuracy: {test_accuracy:.4f}')
print(f'  CV Mean Accuracy: {cv_results["test_accuracy"].mean():.4f}')
print(f'  Overfitting Check: Train-Test Difference = {train_accuracy - test_accuracy:.4f}')
if train_accuracy - test_accuracy < 0.1:
    print('  ✓ No significant overfitting detected')
else:
    print('  ⚠ Potential overfitting - consider regularization')

print('\n' + '='*80)
