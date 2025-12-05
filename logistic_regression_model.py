# -*- coding: utf-8 -*-
"""
Logistic Regression Classification Model
Baseline model with comprehensive evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
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
print('LOGISTIC REGRESSION CLASSIFIER')
print('='*80)
print(f'\nDataset Shape: {df.shape}')
print(f'Missing Values:\n{df.isnull().sum()}')

# Prepare data
df_modeling = df.copy()

le_dict = {}
for col in df_modeling.columns:
    if df_modeling[col].dtype == 'object':
        le = LabelEncoder()
        df_modeling[col] = le.fit_transform(df_modeling[col])
        le_dict[col] = le

X = df_modeling.drop('Role', axis=1)
y = df_modeling['Role']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f'\nTraining set size: {X_train.shape[0]}')
print(f'Test set size: {X_test.shape[0]}')

# Train Logistic Regression
print('\n' + '='*80)
print('LOGISTIC REGRESSION - TRAINING')
print('='*80)

lr_clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', n_jobs=-1)
lr_clf.fit(X_train, y_train)

# Predictions
y_train_pred = lr_clf.predict(X_train)
y_test_pred = lr_clf.predict(X_test)

# Get prediction probabilities for ROC-AUC
y_test_pred_proba = lr_clf.predict_proba(X_test)

# Basic Metrics
print('\n' + '='*80)
print('PERFORMANCE METRICS')
print('='*80)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

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

# Cross-Validation
print('\n' + '='*80)
print('CROSS-VALIDATION ANALYSIS')
print('='*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

cv_results = cross_validate(lr_clf, X_scaled, y, cv=skf, scoring=scoring, return_train_score=True)

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
    ax.bar(range(1, 6), scores, color='seagreen', alpha=0.7, edgecolor='black')
    ax.axhline(y=scores.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.4f}')
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'{title} across Folds', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lr_cross_validation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion Matrix
print('\n' + '='*80)
print('CONFUSION MATRIX')
print('='*80)

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True,
            xticklabels=le_dict['Role'].classes_, yticklabels=le_dict['Role'].classes_)
plt.title('Confusion Matrix - Logistic Regression Test Set', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('lr_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC Curve
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
    ax.set_title('ROC Curves (One-vs-Rest) - Logistic Regression', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lr_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# Model Summary
print('\n' + '='*80)
print('MODEL SUMMARY - LOGISTIC REGRESSION')
print('='*80)
print(f'\nModel: LogisticRegression')
print(f'Max Iterations: 1000')
print(f'Multi-class: multinomial')
print(f'Train/Test Split: 80/20')
print(f'Cross-Validation Folds: 5')
print(f'Feature Scaling: StandardScaler')
print(f'\nBest Metrics:')
print(f'  Test Accuracy: {test_accuracy:.4f}')
print(f'  CV Mean Accuracy: {cv_results["test_accuracy"].mean():.4f}')
print(f'  Overfitting Check: Train-Test Difference = {train_accuracy - test_accuracy:.4f}')
if train_accuracy - test_accuracy < 0.1:
    print('  ✓ No significant overfitting detected')
else:
    print('  ⚠ Potential overfitting detected')

print('\n' + '='*80)
