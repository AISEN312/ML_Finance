"""
Quick data audit script - identify issues and features to drop.
"""
import pandas as pd
import numpy as np
import sys

# Load data
data_dir = r"c:\Users\ADITYA\Downloads\Financial ML Track Project\intern_data"

nifty = pd.read_csv(f"{data_dir}/nifty50.csv", parse_dates=['date'])
starter_features = pd.read_csv(f"{data_dir}/starter_features.csv")

print("=" * 80)
print("COMPREHENSIVE DATA AUDIT REPORT")
print("=" * 80)
print()

# OHLCV Audit
print("OHLCV DATA QUALITY CHECK")
print("-" * 80)
print(f"NIFTY 50 Data: {len(nifty)} rows, {nifty.shape[1]} columns")
print(f"Date range: {nifty['date'].min().date()} to {nifty['date'].max().date()}")
nifty_vol_zero = (nifty['Volume'] == 0).sum()
if nifty_vol_zero > 0:
    print(f"WARNING: {nifty_vol_zero} rows with zero volume")
print()

# Feature Audit
print("STARTER FEATURES AUDIT")
print("-" * 80)
print(f"Total features: {starter_features.shape[1]}")
print(f"Total rows: {len(starter_features)}")
print()

# Problematic and suspicious features
problematic = []
suspicious = []

print("FEATURE ANALYSIS:")
print()

for col in starter_features.columns:
    valid_data = starter_features[col].dropna()
    missing_pct = (starter_features[col].isnull().sum() / len(starter_features)) * 100
    
    # Skip if non-numeric
    try:
        if not pd.api.types.is_numeric_dtype(starter_features[col]):
            print(f"{col:25} | Non-numeric (date/str) - SKIP or handle separately")
            continue
    except:
        pass
    
    if missing_pct > 0:
        print(f"{col:25} | Missing: {missing_pct:5.1f}% | Count: {len(valid_data):4} | Mean: {valid_data.mean():8.4f} | Std: {valid_data.std():8.4f}")
        if missing_pct > 50:
            problematic.append(col)
            print(f"  -> [CRITICAL] Too many missing values")
        elif missing_pct > 10:
            suspicious.append(col)
            print(f"  -> [CAUTION] {missing_pct:.1f}% missing")
    else:
        print(f"{col:25} | Complete    | Count: {len(valid_data):4} | Mean: {valid_data.mean():8.4f} | Std: {valid_data.std():8.4f}")
    
    # Check for zero variance
    if len(valid_data) > 0 and valid_data.std() == 0:
        if col not in problematic:
            problematic.append(col)
        print(f"  -> [CRITICAL] Zero variance")
    
    # Check for out-of-range values
    if 'rsi' in col.lower() and len(valid_data) > 0:
        if valid_data.min() < 0 or valid_data.max() > 100:
            suspicious.append(col)
            print(f"  -> [CAUTION] RSI out of range [{valid_data.min():.1f}, {valid_data.max():.1f}]")
    
    if 'corr' in col.lower() or 'ratio' in col.lower():
        if len(valid_data) > 0 and (valid_data.min() < -2 or valid_data.max() > 2):
            suspicious.append(col)
            print(f"  -> [CAUTION] Correlation/ratio out of range [{valid_data.min():.4f}, {valid_data.max():.4f}]")
    
    print()

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Problematic features (CRITICAL): {len(set(problematic))}")
if problematic:
    for feat in sorted(set(problematic)):
        print(f"  - {feat}")
print()

print(f"Suspicious features (REVIEW): {len(set(suspicious))}")
if suspicious:
    for feat in sorted(set(suspicious)):
        if feat not in problematic:
            print(f"  - {feat}")
print()

print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print()

print("FEATURES TO DROP IMMEDIATELY:")
print("-" * 80)
to_drop = set(problematic)
if to_drop:
    for feat in sorted(to_drop):
        print(f"  - {feat}")
else:
    print("  (None identified based on data quality)")
print()

print("FEATURES TO REVIEW FOR DOMAIN REASONS:")
print("-" * 80)
domain_concerns = {
    'dow': "Day-of-week dummy -> overfitting risk; calendar arbitrage red flag",
    'close_vs_252d_high': "Requires 252-day history; conflicts with 6-month walk-forward windows",
    'close_vs_252d_low': "Requires 252-day history; conflicts with 6-month walk-forward windows",
    'ma5_smooth_signal': "Undocumented smoothing method; reproducibility concern",
    'volume_ratio_20d': "Out-of-range values detected; data quality concern",
}

for feat in sorted(set(suspicious)):
    if feat not in problematic:
        reason = domain_concerns.get(feat, "Data quality or domain-specific concern")
        print(f"  - {feat}: {reason}")
print()

print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
