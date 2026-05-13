"""
Data Loading and Auditing Module for NIFTY 50 Direction Prediction

This module handles:
1. Loading OHLCV data (NIFTY 50, Bank Nifty, India VIX)
2. Loading pre-computed starter features
3. Comprehensive data quality auditing
4. Identifying problematic features for dropping
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and audit financial data for NIFTY 50 prediction model."""
    
    def __init__(self, data_dir: str):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.nifty50 = None
        self.banknifty = None
        self.indiavix = None
        self.starter_features = None
        self.merged_data = None
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required data files.
        
        Returns:
            Dictionary with keys: 'nifty50', 'banknifty', 'indiavix', 'starter_features'
        """
        logger.info("Loading data files...")
        
        # Load OHLCV data
        self.nifty50 = pd.read_csv(self.data_dir / 'nifty50.csv', parse_dates=['date'])
        self.banknifty = pd.read_csv(self.data_dir / 'banknifty.csv', parse_dates=['date'])
        self.indiavix = pd.read_csv(self.data_dir / 'indiavix.csv', parse_dates=['date'])
        
        # Load starter features
        self.starter_features = pd.read_csv(self.data_dir / 'starter_features.csv')
        # Try to parse date column if it exists
        if 'date' in self.starter_features.columns:
            self.starter_features['date'] = pd.to_datetime(self.starter_features['date'])
        
        logger.info(f"NIFTY 50: {len(self.nifty50)} rows, {self.nifty50.shape[1]} columns")
        logger.info(f"Bank Nifty: {len(self.banknifty)} rows, {self.banknifty.shape[1]} columns")
        logger.info(f"India VIX: {len(self.indiavix)} rows, {self.indiavix.shape[1]} columns")
        logger.info(f"Starter Features: {len(self.starter_features)} rows, {self.starter_features.shape[1]} columns")
        
        return {
            'nifty50': self.nifty50,
            'banknifty': self.banknifty,
            'indiavix': self.indiavix,
            'starter_features': self.starter_features
        }
    
    def audit_ohlcv_data(self) -> Dict[str, any]:
        """
        Audit OHLCV data for data quality issues.
        
        Returns:
            Dictionary with audit results
        """
        logger.info("\n" + "="*60)
        logger.info("AUDITING OHLCV DATA")
        logger.info("="*60)
        
        audit_results = {}
        
        for name, df in [('NIFTY 50', self.nifty50), 
                         ('Bank Nifty', self.banknifty), 
                         ('India VIX', self.indiavix)]:
            logger.info(f"\n--- {name} ---")
            
            issues = []
            
            # 1. Check date range
            date_min = df['date'].min()
            date_max = df['date'].max()
            logger.info(f"Date range: {date_min.date()} to {date_max.date()}")
            logger.info(f"Total trading days: {len(df)}")
            
            # 2. Check for duplicates
            dup_count = df.duplicated(subset=['date']).sum()
            if dup_count > 0:
                issues.append(f"Duplicate dates: {dup_count}")
                logger.warning(f"  [WARNING] Duplicate dates found: {dup_count}")
            
            # 3. Check for missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                logger.info(f"  [INFO] Missing values:")
                for col, count in missing[missing > 0].items():
                    logger.info(f"     {col}: {count}")
                    issues.append(f"{col}: {count} missing")
            
            # 4. Check OHLC logic: High >= Low, High >= Open/Close, Low <= Open/Close
            ohlc_issues = 0
            for idx, row in df.iterrows():
                if not (row['High'] >= row['Low']):
                    ohlc_issues += 1
                if not (row['High'] >= row['Open'] and row['High'] >= row['Close']):
                    ohlc_issues += 1
                if not (row['Low'] <= row['Open'] and row['Low'] <= row['Close']):
                    ohlc_issues += 1
            
            if ohlc_issues > 0:
                issues.append(f"OHLC logic violations: {ohlc_issues}")
                logger.warning(f"  [WARNING] OHLC logic violations: {ohlc_issues} rows")
            
            # 5. Check for negative/zero values in OHLC and Volume
            for col in ['Open', 'High', 'Low', 'Close']:
                neg_count = (df[col] <= 0).sum()
                if neg_count > 0:
                    issues.append(f"{col} <= 0: {neg_count}")
                    logger.warning(f"  [WARNING] {col} has {neg_count} non-positive values")
            
            vol_zero = (df['Volume'] == 0).sum()
            if vol_zero > 0:
                if name == 'India VIX':
                    logger.info(f"  [INFO] Volume = 0 for all rows (expected for index)")
                else:
                    logger.warning(f"  [WARNING] Volume = 0 for {vol_zero} rows")
                    issues.append(f"Volume = 0: {vol_zero}")
            
            # 6. Check for extreme price moves (>50% in a day)
            df['pct_change'] = df['Close'].pct_change().abs()
            extreme_moves = (df['pct_change'] > 0.5).sum()
            if extreme_moves > 0:
                logger.warning(f"  [WARNING] Extreme price moves (>50%): {extreme_moves} days")
                issues.append(f"Extreme moves (>50%): {extreme_moves}")
            
            # 7. Check Adj Close vs Close consistency
            adj_close_diff = (df['Adj Close'] - df['Close']).abs()
            if adj_close_diff.max() > 0.01:  # Allow small floating point differences
                logger.warning(f"  [WARNING] Adj Close vs Close discrepancy (max diff: {adj_close_diff.max():.4f})")
                issues.append(f"Adj Close vs Close diff: {adj_close_diff.max():.4f}")
            
            audit_results[name] = {
                'rows': len(df),
                'date_range': f"{date_min.date()} to {date_max.date()}",
                'issues': issues,
                'has_issues': len(issues) > 0
            }
            
            if not issues:
                logger.info(f"  [OK] No issues detected")
        
        return audit_results
    
    def audit_starter_features(self) -> Dict[str, any]:
        """
        Audit starter features for data quality and feature viability.
        
        Returns:
            Dictionary with audit results and feature recommendations
        """
        logger.info("\n" + "="*60)
        logger.info("AUDITING STARTER FEATURES")
        logger.info("="*60)
        
        df = self.starter_features
        audit_results = {
            'total_features': df.shape[1],
            'total_rows': len(df),
            'problematic_features': [],
            'suspicious_features': [],
            'recommendations': {}
        }
        
        logger.info(f"Total features: {df.shape[1]}")
        logger.info(f"Total rows: {len(df)}")
        
        # Analyze each feature
        for col in df.columns:
            if col == 'date':
                continue
                
            logger.info(f"\n--- {col} ---")
            
            # 1. Missingness
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            logger.info(f"  Missing: {missing_pct:.1f}%")
            
            if missing_pct > 50:
                audit_results['problematic_features'].append(col)
                logger.warning(f"  ⚠️ RECOMMENDATION: DROP - Too many missing values (>{50}%)")
                audit_results['recommendations'][col] = "DROP - >50% missing"
                continue
            
            if missing_pct > 10:
                audit_results['suspicious_features'].append(col)
                logger.warning(f"  ⚠️ CAUTION: {missing_pct:.1f}% missing")
            
            # 2. Summary statistics
            valid_data = df[col].dropna()
            if len(valid_data) == 0:
                audit_results['problematic_features'].append(col)
                logger.warning(f"  ⚠️ RECOMMENDATION: DROP - All values are NaN")
                audit_results['recommendations'][col] = "DROP - All NaN"
                continue
            
            logger.info(f"  Count: {len(valid_data)}, Mean: {valid_data.mean():.4f}, "
                       f"Std: {valid_data.std():.4f}, Min: {valid_data.min():.4f}, "
                       f"Max: {valid_data.max():.4f}")
            
            # 3. Check for constant values (no variance)
            if valid_data.std() == 0:
                audit_results['problematic_features'].append(col)
                logger.warning(f"  ⚠️ RECOMMENDATION: DROP - Zero variance (constant value)")
                audit_results['recommendations'][col] = "DROP - Zero variance"
                continue
            
            # 4. Check for extreme outliers or unusual ranges
            
            # For ratio/correlation features: should be -1 to 1
            if 'corr' in col.lower() or 'ratio' in col.lower():
                if valid_data.min() < -2 or valid_data.max() > 2:
                    logger.warning(f"  ⚠️ Out of expected range for correlation/ratio: "
                                 f"[{valid_data.min():.4f}, {valid_data.max():.4f}]")
                    audit_results['suspicious_features'].append(col)
            
            # For RSI: should be 0-100
            if 'rsi' in col.lower():
                if valid_data.min() < -5 or valid_data.max() > 105:
                    logger.warning(f"  ⚠️ Out of expected range for RSI [0-100]: "
                                 f"[{valid_data.min():.4f}, {valid_data.max():.4f}]")
                    audit_results['suspicious_features'].append(col)
                    audit_results['recommendations'][col] = "REVIEW - RSI out of range"
            
            # For volume/log_volume metrics
            if 'volume' in col.lower() and 'log' not in col.lower():
                if valid_data.min() < 0:
                    logger.warning(f"  ⚠️ Negative values in volume metric")
                    audit_results['problematic_features'].append(col)
                    audit_results['recommendations'][col] = "DROP - Negative volumes"
            
            # 5. Check for any inf or extremely large values
            if np.isinf(valid_data).any():
                audit_results['problematic_features'].append(col)
                logger.warning(f"  ⚠️ RECOMMENDATION: DROP - Contains infinite values")
                audit_results['recommendations'][col] = "DROP - Infinite values"
                continue
            
            if valid_data.abs().max() > 1e10:
                logger.warning(f"  ⚠️ CAUTION: Extremely large values (max abs: {valid_data.abs().max():.2e})")
                audit_results['suspicious_features'].append(col)
            
            # 6. Check for lookhead bias indicators
            lookhead_bias_keywords = ['future', 'next', 'ahead']
            if any(keyword in col.lower() for keyword in lookhead_bias_keywords):
                logger.warning(f"  ⚠️ CAUTION: Potential lookhead bias indicator")
                audit_results['suspicious_features'].append(col)
        
        logger.info(f"\n\nSUMMARY:")
        logger.info(f"  Problematic features (recommend DROP): {len(audit_results['problematic_features'])}")
        logger.info(f"  Suspicious features (review): {len(audit_results['suspicious_features'])}")
        
        return audit_results
    
    def get_feature_drop_recommendations(self) -> Tuple[List[str], str]:
        """
        Generate final recommendations for features to drop.
        
        Returns:
            Tuple of (list of features to drop, detailed explanation)
        """
        audit = self.audit_starter_features()
        
        features_to_drop = []
        explanation_lines = []
        
        explanation_lines.append("="*70)
        explanation_lines.append("FEATURE DROP RECOMMENDATIONS FOR STARTER_FEATURES.CSV")
        explanation_lines.append("="*70)
        explanation_lines.append("")
        
        # 1. High missingness
        explanation_lines.append("TIER 1: CRITICAL ISSUES (DROP IMMEDIATELY)")
        explanation_lines.append("-" * 70)
        explanation_lines.append("")
        
        for col in audit['problematic_features']:
            features_to_drop.append(col)
            reason = audit['recommendations'].get(col, "Problematic data quality")
            explanation_lines.append(f"  [DROP] {col}: {reason}")
        
        explanation_lines.append("")
        explanation_lines.append("TIER 2: FEATURES TO REVIEW (POTENTIALLY DROP)")
        explanation_lines.append("-" * 70)
        explanation_lines.append("")
        
        # 2. Domain-specific concerns
        domain_concerns = {
            'dow': "Day-of-week dummy variable may overfit; creates 5 separate features via one-hot encoding",
            'ma5_smooth_signal': "Undocumented smoothing method; could embed lookahead bias",
            'close_vs_252d_high': "Requires 252-day history; may conflict with rolling walk-forward splits",
            'close_vs_252d_low': "Requires 252-day history; may conflict with rolling walk-forward splits",
        }
        
        for col in audit['suspicious_features']:
            if col not in features_to_drop:
                if col in domain_concerns:
                    reason = domain_concerns[col]
                    explanation_lines.append(f"  [REVIEW] {col}: {reason}")
        
        explanation_lines.append("")
        explanation_lines.append("RATIONALE FOR DROPPING FEATURES")
        explanation_lines.append("-" * 70)
        explanation_lines.append("")
        explanation_lines.append("1. HIGH MISSINGNESS (>50%):")
        explanation_lines.append("   - Model cannot reliably train on severely sparse data")
        explanation_lines.append("   - Imputation introduces artificial patterns")
        explanation_lines.append("")
        explanation_lines.append("2. ZERO VARIANCE / CONSTANT VALUES:")
        explanation_lines.append("   - No predictive signal; wastes model capacity")
        explanation_lines.append("   - Breaks tree-based split selection")
        explanation_lines.append("")
        explanation_lines.append("3. LOOKHEAD BIAS / FORWARD-LOOKING:")
        explanation_lines.append("   - Features computed with future data violate timestamp safety")
        explanation_lines.append("   - Invalid out-of-sample predictions")
        explanation_lines.append("")
        explanation_lines.append("4. OUT-OF-RANGE VALUES:")
        explanation_lines.append("   - RSI outside [0, 100]: computation error or data quality issue")
        explanation_lines.append("   - Indicates undocumented data processing")
        explanation_lines.append("")
        explanation_lines.append("5. DOMAIN-SPECIFIC CONCERNS:")
        explanation_lines.append("   - Day-of-week dummies: overfitting risk + calendar arbitrage red flag")
        explanation_lines.append("   - 252-day features: conflict with ~6-month walk-forward windows")
        explanation_lines.append("   - Undocumented smoothing: reproducibility risk")
        explanation_lines.append("")
        
        explanation = "\n".join(explanation_lines)
        
        return features_to_drop, explanation
    
    def print_comprehensive_audit_report(self):
        """Print a comprehensive audit report to console."""
        
        # Load all data
        self.load_all_data()
        
        # Audit OHLCV
        ohlcv_audit = self.audit_ohlcv_data()
        
        # Audit features (will print detailed analysis)
        feature_audit = self.audit_starter_features()
        
        # Get drop recommendations
        drops, explanation = self.get_feature_drop_recommendations()
        
        # Print explanation
        print("\n" + explanation)
        print(f"\nFEATURES TO DROP: {len(drops)}")
        if drops:
            for feat in sorted(drops):
                print(f"  - {feat}")
        else:
            print("  (No features categorized as immediately problematic)")
        
        return ohlcv_audit, feature_audit, drops


if __name__ == "__main__":
    # Run the audit
    data_dir = r"c:\Users\ADITYA\Downloads\Financial ML Track Project\intern_data"
    
    loader = DataLoader(data_dir)
    ohlcv, features, drops = loader.print_comprehensive_audit_report()
    
    # Additional analysis for suspicious features
    audit = loader.audit_starter_features()
    if audit['suspicious_features']:
        print("\n" + "="*70)
        print("SUSPICIOUS FEATURES FLAGGED (NEED CLOSER REVIEW)")
        print("="*70)
        for feat in sorted(audit['suspicious_features']):
            print(f"\n  • {feat}")
            if feat in audit['recommendations']:
                print(f"    Reason: {audit['recommendations'][feat]}")
