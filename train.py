"""
train.py — SensorGuard: Predictive Anomaly Detection on Time-Series Sensor Data

End-to-end ML pipeline on NASA CMAPSS industrial sensor data:
1. Data loading & label creation (RUL-based failure labeling)
2. Feature engineering (lag features + rolling statistics)
3. Exploratory data analysis with saved plots
4. Model training & benchmarking (Isolation Forest, Random Forest, SVM)
5. Cross-validation and classification reporting
6. Shadow-mode deployment simulation
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def main():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    # ========================================================
    # Step 1 — Load Data
    # ========================================================
    print("=" * 60)
    print("STEP 1: Loading NASA CMAPSS Turbofan Engine Dataset")
    print("=" * 60)

    cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    data_path = os.path.join('data', 'train_FD001.txt')

    if not os.path.exists(data_path):
        print(f"Dataset not found at '{data_path}'. Generating synthetic data...")
        from generate_data import generate_cmapss_data
        df = generate_cmapss_data(n_units=100)
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, sep=' ', header=False, index=False)
        print("Synthetic data generated.\n")

    train_df = pd.read_csv(data_path, sep=' ', header=None, names=cols)
    train_df = train_df.dropna(axis=1)
    print(f"Dataset shape: {train_df.shape}")
    print(f"Engine units:  {train_df['unit'].nunique()}")
    print(f"Columns:       {list(train_df.columns)}")
    print(train_df.head())
    print()

    # ========================================================
    # Step 2 — Create Labels (Remaining Useful Life)
    # ========================================================
    print("=" * 60)
    print("STEP 2: Creating Failure Labels (RUL-based)")
    print("=" * 60)

    max_cycle = train_df.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max_cycle']
    train_df = train_df.merge(max_cycle, on='unit')

    # RUL = max_cycle - current_cycle
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']

    # Binary label: 1 = failure imminent (RUL <= 30), 0 = normal
    train_df['failure'] = (train_df['RUL'] <= 30).astype(int)

    print(train_df['failure'].value_counts())
    print(f"Failure rate: {train_df['failure'].mean():.2%}")
    print()

    # ========================================================
    # Step 3 — Feature Engineering (Lag + Rolling)
    # ========================================================
    print("=" * 60)
    print("STEP 3: Feature Engineering (Rolling Stats + Lag Features)")
    print("=" * 60)

    sensor_cols = [f's{i}' for i in range(1, 22) if f's{i}' in train_df.columns]

    # Rolling mean and std (window=5) per unit
    for col in sensor_cols:
        train_df[f'{col}_roll_mean'] = train_df.groupby('unit')[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        train_df[f'{col}_roll_std'] = train_df.groupby('unit')[col].transform(
            lambda x: x.rolling(5, min_periods=1).std().fillna(0)
        )

    # Lag features (lag=1)
    for col in sensor_cols:
        train_df[f'{col}_lag1'] = train_df.groupby('unit')[col].shift(1).fillna(0)

    print(f"Features after engineering: {train_df.shape[1]} columns")
    print(f"  - {len(sensor_cols)} raw sensor features")
    print(f"  - {len(sensor_cols)} rolling mean features")
    print(f"  - {len(sensor_cols)} rolling std features")
    print(f"  - {len(sensor_cols)} lag-1 features")
    print()

    # ========================================================
    # Step 4 — EDA Plots
    # ========================================================
    print("=" * 60)
    print("STEP 4: Exploratory Data Analysis & Visualization")
    print("=" * 60)

    # Sensor trends for Unit 1
    unit1 = train_df[train_df['unit'] == 1]
    plot_sensors = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13']

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, col in enumerate(plot_sensors):
        axes[i // 3][i % 3].plot(unit1['cycle'], unit1[col], color='steelblue', linewidth=0.8)
        axes[i // 3][i % 3].set_title(f'Sensor {col} — Unit 1', fontsize=10)
        axes[i // 3][i % 3].set_xlabel('Cycle')
    plt.suptitle('Sensor Readings Over Time (Unit 1)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/sensor_trends.png', dpi=150)
    plt.close()
    print("Sensor trends plot saved: plots/sensor_trends.png")

    # Class distribution
    plt.figure(figsize=(6, 4))
    train_df['failure'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'], edgecolor='black')
    plt.title('Class Distribution (0=Normal, 1=Failure Imminent)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('plots/class_distribution.png', dpi=150)
    plt.close()
    print("Class distribution plot saved: plots/class_distribution.png")

    # RUL distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(train_df['RUL'], bins=50, kde=True, color='steelblue')
    plt.title('Remaining Useful Life (RUL) Distribution')
    plt.xlabel('RUL (cycles)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('plots/rul_distribution.png', dpi=150)
    plt.close()
    print("RUL distribution plot saved: plots/rul_distribution.png")

    # Sensor correlation heatmap (subset for readability)
    plt.figure(figsize=(12, 9))
    corr = train_df[sensor_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Sensor Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/sensor_correlation.png', dpi=150)
    plt.close()
    print("Sensor correlation heatmap saved: plots/sensor_correlation.png")
    print()

    # ========================================================
    # Step 5 — Prepare Features
    # ========================================================
    print("=" * 60)
    print("STEP 5: Preparing Feature Matrix & Scaling")
    print("=" * 60)

    feature_cols = (
        sensor_cols +
        [f'{c}_roll_mean' for c in sensor_cols] +
        [f'{c}_roll_std' for c in sensor_cols] +
        [f'{c}_lag1' for c in sensor_cols]
    )

    X = train_df[feature_cols]
    y = train_df['failure']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Feature matrix: {X.shape[1]} features")
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution:  {np.bincount(y_test)}")
    print()

    # ========================================================
    # Step 6 — Train & Compare Models
    # ========================================================
    print("=" * 60)
    print("STEP 6: Training & Benchmarking Models")
    print("=" * 60)

    results = {}

    # 1. Isolation Forest (unsupervised anomaly detection)
    print("\nTraining Isolation Forest...")
    iso = IsolationForest(contamination=0.15, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    iso_preds = iso.predict(X_test)
    iso_preds = np.where(iso_preds == -1, 1, 0)
    results['Isolation Forest'] = {
        'F1': f1_score(y_test, iso_preds),
        'Precision': precision_score(y_test, iso_preds),
        'Recall': recall_score(y_test, iso_preds)
    }
    print(f"  F1={results['Isolation Forest']['F1']:.4f}  "
          f"Precision={results['Isolation Forest']['Precision']:.4f}  "
          f"Recall={results['Isolation Forest']['Recall']:.4f}")

    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results['Random Forest'] = {
        'F1': f1_score(y_test, rf_preds),
        'Precision': precision_score(y_test, rf_preds),
        'Recall': recall_score(y_test, rf_preds)
    }
    print(f"  F1={results['Random Forest']['F1']:.4f}  "
          f"Precision={results['Random Forest']['Precision']:.4f}  "
          f"Recall={results['Random Forest']['Recall']:.4f}")

    # 3. SVM
    print("Training SVM (RBF kernel)...")
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    results['SVM'] = {
        'F1': f1_score(y_test, svm_preds),
        'Precision': precision_score(y_test, svm_preds),
        'Recall': recall_score(y_test, svm_preds)
    }
    print(f"  F1={results['SVM']['F1']:.4f}  "
          f"Precision={results['SVM']['Precision']:.4f}  "
          f"Recall={results['SVM']['Recall']:.4f}")

    # Results table
    results_df = pd.DataFrame(results).T
    print(f"\n{'=' * 50}")
    print("MODEL COMPARISON RESULTS")
    print('=' * 50)
    print(results_df.round(4).to_string())
    print()

    # ========================================================
    # Step 7 — Visualize Model Comparison & Classification Report
    # ========================================================
    print("=" * 60)
    print("STEP 7: Visualization & Classification Report")
    print("=" * 60)

    # Bar chart comparison
    results_df.plot(kind='bar', figsize=(10, 6), colormap='Set2', edgecolor='black')
    plt.title('Model Comparison — Precision, Recall, F1', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=150)
    plt.close()
    print("Model comparison plot saved: plots/model_comparison.png")

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (name, preds) in zip(axes, [
        ('Isolation Forest', iso_preds),
        ('Random Forest', rf_preds),
        ('SVM', svm_preds)
    ]):
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Failure']).plot(ax=ax, cmap='Blues')
        ax.set_title(name, fontsize=12, fontweight='bold')
    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrices.png', dpi=150)
    plt.close()
    print("Confusion matrices saved: plots/confusion_matrices.png")

    # Best model detailed report
    print("\n--- Best Model: Random Forest ---")
    print(classification_report(y_test, rf_preds, target_names=['Normal', 'Failure']))

    # Cross-validation on best model
    print("Running 5-Fold Cross-Validation on Random Forest...")
    cv = cross_val_score(rf, X_scaled, y, cv=5, scoring='f1')
    print(f"Cross-Val F1: {cv.mean():.4f} ± {cv.std():.4f}\n")

    # Save the best model
    model_path = 'saved_models/random_forest_model.pkl'
    joblib.dump(rf, model_path)
    scaler_path = 'saved_models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Best model saved: {model_path}")
    print(f"Scaler saved:     {scaler_path}")

    # ========================================================
    # Step 8 — Shadow Mode Simulation
    # ========================================================
    print("\n" + "=" * 60)
    print("STEP 8: Shadow-Mode Deployment Simulation")
    print("=" * 60)

    # Hold out last 20% of cycles per unit as "live" data
    train_df_sorted = train_df.sort_values(['unit', 'cycle'])
    cutoff = int(len(train_df_sorted) * 0.8)

    shadow_train = train_df_sorted.iloc[:cutoff]
    shadow_test = train_df_sorted.iloc[cutoff:]

    X_shadow_train = scaler.fit_transform(shadow_train[feature_cols])
    X_shadow_test = scaler.transform(shadow_test[feature_cols])
    y_shadow_test = shadow_test['failure']

    rf_shadow = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_shadow.fit(X_shadow_train, shadow_train['failure'])
    shadow_preds = rf_shadow.predict(X_shadow_test)

    print("\nShadow Mode Validation Results:")
    print(classification_report(y_shadow_test, shadow_preds, target_names=['Normal', 'Failure']))

    shadow_f1 = f1_score(y_shadow_test, shadow_preds)
    print(f"Shadow Mode F1 Score: {shadow_f1:.4f}")

    # Shadow mode confusion matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_shadow_test, shadow_preds)
    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Failure']).plot(cmap='Oranges')
    plt.title('Shadow Mode — Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/shadow_mode_confusion.png', dpi=150)
    plt.close()
    print("Shadow mode confusion matrix saved: plots/shadow_mode_confusion.png")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == '__main__':
    main()
