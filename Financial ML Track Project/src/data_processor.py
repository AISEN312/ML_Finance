"""
Data Processing Pipeline for NIFTY 50 Direction Prediction Model

Handles:
1. Loading OHLCV and feature data
2. Merging on dates
3. Removing problematic features
4. Computing target variable (next-day direction)
5. Handling missing values
6. Preparing data for walk-forward validation
7. MLflow tracking integration
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings
from datetime import datetime

# Try to import MLflow, but make it optional
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Production-ready data processing pipeline for NIFTY 50 direction prediction."""
    
    # Features identified as problematic in data audit
    FEATURES_TO_DROP = [
        'close_vs_252d_high',      # 252-day requirement conflicts with walk-forward
        'close_vs_252d_low',       # 252-day requirement conflicts with walk-forward
        'volume_ratio_20d',        # 14.3% missing + out-of-range values
        'dow',                     # Day-of-week: calendar overfitting risk
        'ma5_smooth_signal',       # Undocumented smoothing; reproducibility risk
    ]
    
    def __init__(self, data_dir: str, run_name: Optional[str] = None):
        """
        Initialize DataProcessor.
        
        Args:
            data_dir: Path to directory containing CSV files
            run_name: Optional MLflow run name (auto-generated if None)
        """
        self.data_dir = Path(data_dir)
        self.run_name = run_name or f"data_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Data storage
        self.nifty50_raw = None
        self.banknifty_raw = None
        self.indiavix_raw = None
        self.features_raw = None
        self.data_merged = None
        self.data_processed = None
        
        # Tracking
        self.mlflow_run = None
        self.data_stats = {}
        
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw data files."""
        logger.info("="*70)
        logger.info("LOADING RAW DATA")
        logger.info("="*70)
        
        try:
            self.nifty50_raw = pd.read_csv(self.data_dir / 'nifty50.csv', parse_dates=['date'])
            self.banknifty_raw = pd.read_csv(self.data_dir / 'banknifty.csv', parse_dates=['date'])
            self.indiavix_raw = pd.read_csv(self.data_dir / 'indiavix.csv', parse_dates=['date'])
            self.features_raw = pd.read_csv(self.data_dir / 'starter_features.csv')
            
            # Try to parse date if exists
            if 'date' in self.features_raw.columns:
                self.features_raw['date'] = pd.to_datetime(self.features_raw['date'])
            
            logger.info(f"NIFTY 50:       {len(self.nifty50_raw):4d} rows, {self.nifty50_raw.shape[1]:2d} cols")
            logger.info(f"Bank Nifty:     {len(self.banknifty_raw):4d} rows, {self.banknifty_raw.shape[1]:2d} cols")
            logger.info(f"India VIX:      {len(self.indiavix_raw):4d} rows, {self.indiavix_raw.shape[1]:2d} cols")
            logger.info(f"Starter Features: {len(self.features_raw):4d} rows, {self.features_raw.shape[1]:2d} cols")
            
            return {
                'nifty50': self.nifty50_raw,
                'banknifty': self.banknifty_raw,
                'indiavix': self.indiavix_raw,
                'features': self.features_raw
            }
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
    
    def compute_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute target variable: next-day NIFTY 50 close direction.
        
        Target = 1 if tomorrow's close > today's close, else 0
        
        Args:
            df: DataFrame with NIFTY 50 Close prices
            
        Returns:
            DataFrame with new 'target' column
        """
        logger.info("\nComputing target variable (next-day direction)...")
        
        # Next day's close
        df['next_day_close'] = df['Close'].shift(-1)
        
        # Direction: 1 if up, 0 if down (NaN for last row)
        df['target'] = (df['next_day_close'] > df['Close']).astype(int)
        
        # Count labels
        n_up = (df['target'] == 1).sum()
        n_down = (df['target'] == 0).sum()
        n_na = df['target'].isna().sum()
        
        logger.info(f"Target distribution:")
        logger.info(f"  Up (1):   {n_up:4d} ({100*n_up/(n_up+n_down):5.1f}%)")
        logger.info(f"  Down (0): {n_down:4d} ({100*n_down/(n_up+n_down):5.1f}%)")
        logger.info(f"  NaN:      {n_na:4d} (last row - no next day)")
        
        # Store stats
        self.data_stats['target_up'] = n_up
        self.data_stats['target_down'] = n_down
        self.data_stats['target_balance'] = n_up / (n_up + n_down) if (n_up + n_down) > 0 else 0
        
        return df
    
    def merge_data(self) -> pd.DataFrame:
        """Merge all data sources on date."""
        logger.info("\n" + "="*70)
        logger.info("MERGING DATA")
        logger.info("="*70)
        
        # Start with NIFTY 50 and compute target
        df = self.nifty50_raw.copy()
        df = self.compute_target(df)
        
        # Rename NIFTY 50 columns to avoid confusion
        nifty_cols = {col: f'nifty_{col.lower()}' if col not in ['date', 'target', 'next_day_close'] 
                      else col for col in df.columns}
        df = df.rename(columns=nifty_cols)
        
        # Merge Bank Nifty
        bn_df = self.banknifty_raw.copy()
        bn_df = bn_df.rename(columns={col: f'bn_{col.lower()}' if col != 'date' else col 
                                       for col in bn_df.columns})
        df = pd.merge(df, bn_df, on='date', how='left')
        logger.info(f"After Bank Nifty merge: {len(df)} rows")
        
        # Merge India VIX
        vix_df = self.indiavix_raw.copy()
        vix_df = vix_df.rename(columns={col: f'vix_{col.lower()}' if col != 'date' else col 
                                         for col in vix_df.columns})
        df = pd.merge(df, vix_df, on='date', how='left')
        logger.info(f"After VIX merge: {len(df)} rows")
        
        # Merge starter features
        features_df = self.features_raw.copy()
        df = pd.merge(df, features_df, on='date', how='left')
        logger.info(f"After features merge: {len(df)} rows")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Final merged dataset: {len(df)} rows, {df.shape[1]} columns")
        
        self.data_merged = df
        return df
    
    def drop_problematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features identified as problematic in data audit."""
        logger.info("\n" + "="*70)
        logger.info("REMOVING PROBLEMATIC FEATURES")
        logger.info("="*70)
        
        available_to_drop = [f for f in self.FEATURES_TO_DROP if f in df.columns]
        logger.info(f"Dropping {len(available_to_drop)} features:")
        for feat in available_to_drop:
            logger.info(f"  - {feat}")
        
        df = df.drop(columns=available_to_drop, errors='ignore')
        
        logger.info(f"Remaining columns: {len(df.columns)}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values.
        
        Args:
            df: DataFrame with potential missing values
            method: 'forward_fill' or 'drop'
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("\n" + "="*70)
        logger.info("HANDLING MISSING VALUES")
        logger.info("="*70)
        
        # Identify columns with missing values
        missing_before = df.isnull().sum()
        missing_before = missing_before[missing_before > 0]
        
        if len(missing_before) > 0:
            logger.info(f"Columns with missing values ({len(missing_before)}):")
            for col, count in missing_before.items():
                pct = 100 * count / len(df)
                logger.info(f"  - {col:30s}: {count:4d} ({pct:5.2f}%)")
        
        
        # Handle missing values
        if method == 'forward_fill':
            logger.info(f"Using forward-fill method...")
            # First forward-fill, then backward-fill for initial NaNs
            df = df.ffill().bfill()
            
        elif method == 'drop':
            logger.info(f"Dropping rows with any missing values...")
            initial_rows = len(df)
            df = df.dropna()
            rows_dropped = initial_rows - len(df)
            logger.info(f"Dropped {rows_dropped} rows ({100*rows_dropped/initial_rows:.1f}%)")
        # Verify
        missing_after = df.isnull().sum()
        remaining_missing = (missing_after > 0).sum()
        
        if remaining_missing > 0:
            logger.warning(f"Still {remaining_missing} columns with missing values!")
            logger.warning(missing_after[missing_after > 0])
        else:
            logger.info("All missing values handled successfully")
        
        return df
    
    
    def split_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Extract features and target from dataset.
        
        Args:
            df: Processed dataframe
            
        Returns:
            Tuple of (features, target, dates)
        """
        logger.info("\n" + "="*70)
        logger.info("EXTRACTING FEATURES AND TARGET")
        logger.info("="*70)
        
        # Target (remove NaN from last row where we can't compute next day)
        target = df['target'].copy()
        dates = df['date'].copy()
        
        # Columns to exclude: metadata, OHLC data
        exclude_cols = {
            'date', 'next_day_close', 'target',
            # NIFTY OHLC
            'nifty_open', 'nifty_high', 'nifty_low', 'nifty_close', 
            'nifty_adj close', 'nifty_volume',
            # Bank Nifty OHLC
            'bn_open', 'bn_high', 'bn_low', 'bn_close', 
            'bn_adj close', 'bn_volume',
            # VIX OHLC
            'vix_open', 'vix_high', 'vix_low', 'vix_close', 
            'vix_adj close', 'vix_volume'
        }
        
        # Keep features: starter features + Bank Nifty returns + VIX metrics
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        features = df[feature_cols].copy()
        
        logger.info(f"Feature columns: {len(features.columns)}")
        logger.info(f"Features: {sorted(list(features.columns))}")
        logger.info(f"Target samples: {len(target)}")
        logger.info(f"Target type: {target.dtype}")
        
        self.data_stats['n_features'] = len(features.columns)
        self.data_stats['feature_list'] = sorted(list(features.columns))
        
        return features, target, dates
    def process_all(self, method: str = 'forward_fill') -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Run complete data processing pipeline.
        
        Args:
            method: Missing value handling method ('forward_fill' or 'drop')
            
        Returns:
            Tuple of (features, target, dates)
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("="*70 + "\n")
        
        # Load
        self.load_raw_data()
        
        # Merge
        df = self.merge_data()
        
        # Drop problematic features
        df = self.drop_problematic_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df, method=method)
        
        # Store processed dataset
        self.data_processed = df
        
        # Extract features and target
        X, y, dates = self.split_features_and_target(df)
        
        logger.info("\n" + "="*70)
        logger.info("DATA PROCESSING COMPLETE")
        logger.info("="*70)
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Date range: {dates.min().date()} to {dates.max().date()}")
        
        return X, y, dates
    
    def log_to_mlflow(self, metadata: Dict = None):
        """Log data processing metadata to MLflow."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available - skipping MLflow logging")
            return None
            
        try:
            with mlflow.start_run(run_name=self.run_name) as run:
                mlflow.set_tag("pipeline", "data_processing")
                mlflow.set_tag("dataset", "nifty50_prediction")
                
                # Log parameters
                mlflow.log_param("features_dropped_count", len(self.FEATURES_TO_DROP))
                mlflow.log_param("features_dropped_list", str(self.FEATURES_TO_DROP))
                
                # Log metrics
                for key, value in self.data_stats.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                # Log metadata
                if metadata:
                    mlflow.log_dict(metadata, "processing_metadata.json")
                
                self.mlflow_run = run
                logger.info(f"MLflow run logged: {run.info.run_id}")
                
                return run.info.run_id
        except Exception as e:
            logger.warning(f"Error logging to MLflow: {e}")
            return None
    
    def save_processed_data(self, output_dir: str, X: pd.DataFrame, y: pd.Series, dates: pd.Series):
        """Save processed data for later use."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving processed data to {output_path}...")
        
        # Save features
        X.to_csv(output_path / 'X_features.csv', index=False)
        logger.info(f"  - X_features.csv ({X.shape})")
        
        # Save target
        y.to_csv(output_path / 'y_target.csv', index=False, header=['target'])
        logger.info(f"  - y_target.csv ({y.shape})")
        
        # Save dates
        dates.to_csv(output_path / 'dates.csv', index=False, header=['date'])
        logger.info(f"  - dates.csv ({dates.shape})")
        
        # Save feature names
        with open(output_path / 'feature_names.txt', 'w') as f:
            for feat in X.columns:
                f.write(f"{feat}\n")
        logger.info(f"  - feature_names.txt")
        
        # Save stats
        import json
        with open(output_path / 'processing_stats.json', 'w') as f:
            json.dump(self.data_stats, f, indent=2, default=str)
        logger.info(f"  - processing_stats.json")
        
        logger.info("Data saved successfully")



if __name__ == "__main__":
    # Example usage
    data_dir = r"c:\Users\ADITYA\Downloads\Financial ML Track Project\intern_data"
    output_dir = r"c:\Users\ADITYA\Downloads\Financial ML Track Project\data\processed"
    
    # Set up MLflow if available
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("nifty50_direction_prediction")
    
    # Process data
    processor = DataProcessor(data_dir, run_name="initial_data_processing")
    X, y, dates = processor.process_all(method='forward_fill')
    
    # Log to MLflow
    processor.log_to_mlflow(metadata={
        'data_dir': str(data_dir),
        'processing_date': datetime.now().isoformat(),
        'shape': {'features': X.shape, 'target': y.shape}
    })
    
    # Save processed data
    processor.save_processed_data(output_dir, X, y, dates)
    
    logger.info("\nData processing pipeline completed successfully!")
