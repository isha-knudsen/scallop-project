"""
Behavioral Classifier
Uses machine learning to classify vessel behavior as fishing vs transiting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class BehavioralClassifier:
    """Classifies vessel behavior using speed and movement patterns"""
    
    # Speed thresholds
    DREDGING_SPEED_MIN = 3.5
    DREDGING_SPEED_MAX = 4.5
    STEAMING_SPEED_MIN = 6.0
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def classify_behavior(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify behavior for all records
        
        Args:
            data: DataFrame with vessel AIS data
        
        Returns:
            DataFrame with added 'behavior' and 'confidence' columns
        """
        logger.info(f"Classifying behavior for {len(data)} records")
        
        df = data.copy()
        
        # Calculate features
        df = self._calculate_features(df)
        
        # Rule-based classification (primary method)
        df['behavior'] = 'unknown'
        df['confidence'] = 0.0
        
        # Active dredging: speed in range, low variance
        dredging_mask = (
            (df['SOG'] >= self.DREDGING_SPEED_MIN) &
            (df['SOG'] <= self.DREDGING_SPEED_MAX) &
            (df['speed_variance'] < 0.5)
        )
        df.loc[dredging_mask, 'behavior'] = 'fishing'
        df.loc[dredging_mask, 'confidence'] = 0.9
        
        # Steaming: high speed, relatively straight
        steaming_mask = (
            (df['SOG'] >= self.STEAMING_SPEED_MIN) &
            (df['heading_variance'] < 30)
        )
        df.loc[steaming_mask, 'behavior'] = 'steaming'
        df.loc[steaming_mask, 'confidence'] = 0.8
        
        # Maneuvering: low speed, high heading variance
        maneuvering_mask = (
            (df['SOG'] < self.DREDGING_SPEED_MIN) &
            (df['heading_variance'] > 30)
        )
        df.loc[maneuvering_mask, 'behavior'] = 'maneuvering'
        df.loc[maneuvering_mask, 'confidence'] = 0.7
        
        # If we have a trained model, use it to refine classifications
        if self.is_trained:
            df = self._apply_ml_classification(df)
        
        logger.info(f"Classification complete:")
        logger.info(f"  Fishing: {(df['behavior'] == 'fishing').sum()}")
        logger.info(f"  Steaming: {(df['behavior'] == 'steaming').sum()}")
        logger.info(f"  Maneuvering: {(df['behavior'] == 'maneuvering').sum()}")
        logger.info(f"  Unknown: {(df['behavior'] == 'unknown').sum()}")
        
        return df
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate behavioral features for classification"""
        
        # Calculate rolling statistics (over 5 points)
        window = 5
        
        for mmsi, group in df.groupby('MMSI'):
            group = group.sort_values('timestamp')
            
            # Speed variance
            df.loc[group.index, 'speed_variance'] = (
                group['SOG'].rolling(window, min_periods=1).std()
            )
            
            # Heading variance
            if 'COG' in group.columns:
                df.loc[group.index, 'heading_variance'] = (
                    group['COG'].rolling(window, min_periods=1).std()
                )
            else:
                df.loc[group.index, 'heading_variance'] = 0
            
            # Rate of turn (degrees per minute)
            if 'COG' in group.columns and 'time_diff' in group.columns:
                heading_diff = group['COG'].diff()
                # Adjust for wraparound
                heading_diff = np.where(heading_diff > 180, heading_diff - 360, heading_diff)
                heading_diff = np.where(heading_diff < -180, heading_diff + 360, heading_diff)
                
                rate_of_turn = np.abs(heading_diff) / (group['time_diff'] / 60)
                df.loc[group.index, 'rate_of_turn'] = rate_of_turn
            else:
                df.loc[group.index, 'rate_of_turn'] = 0
            
            # Tortuosity (path straightness)
            if 'distance' in group.columns:
                total_distance = group['distance'].rolling(window, min_periods=1).sum()
                straight_line_distance = np.sqrt(
                    (group['LAT'].diff(window)**2 + group['LON'].diff(window)**2)
                ) * 111000  # Rough conversion to meters
                
                tortuosity = total_distance / (straight_line_distance + 1)
                df.loc[group.index, 'tortuosity'] = tortuosity
            else:
                df.loc[group.index, 'tortuosity'] = 1.0
        
        # Fill NaN values
        df['speed_variance'] = df['speed_variance'].fillna(0)
        df['heading_variance'] = df['heading_variance'].fillna(0)
        df['rate_of_turn'] = df['rate_of_turn'].fillna(0)
        df['tortuosity'] = df['tortuosity'].fillna(1)
        
        return df
    
    def train_model(self, training_data: pd.DataFrame, labels: pd.Series):
        """
        Train machine learning model on labeled data
        
        Args:
            training_data: DataFrame with features
            labels: Series with behavior labels ('fishing', 'steaming', etc.)
        """
        logger.info("Training behavioral classification model")
        
        # Ensure features are calculated
        if 'speed_variance' not in training_data.columns:
            training_data = self._calculate_features(training_data)
        
        # Select features
        feature_cols = [
            'SOG', 'speed_variance', 'heading_variance',
            'rate_of_turn', 'tortuosity'
        ]
        
        X = training_data[feature_cols].values
        y = labels.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train random forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        
        # Feature importance
        importance = dict(zip(feature_cols, self.model.feature_importances_))
        logger.info(f"Feature importance: {importance}")
        
    def _apply_ml_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply trained ML model to refine classifications"""
        
        feature_cols = [
            'SOG', 'speed_variance', 'heading_variance',
            'rate_of_turn', 'tortuosity'
        ]
        
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Update behavior where confidence is low
        low_confidence_mask = df['confidence'] < 0.7
        df.loc[low_confidence_mask, 'behavior'] = predictions[low_confidence_mask]
        df.loc[low_confidence_mask, 'confidence'] = probabilities[low_confidence_mask].max(axis=1)
        
        return df
    
    def save_model(self, path: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
    
    def get_vessel_behavior(self, mmsi: str) -> Dict:
        """Get behavior summary for a specific vessel"""
        # This would query the classified data
        # Implementation depends on data storage
        raise NotImplementedError("Implement based on data storage")
    
    def calculate_catchability(
        self,
        start_date: str,
        end_date: str,
        landings_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Calculate catchability scores for fishing locations
        
        Catchability is the probability of successful catch given:
        - Environmental conditions
        - Historical catch data
        - Fishing effort
        """
        logger.info("Calculating catchability scores")
        
        # This is a simplified version
        # Full implementation would integrate:
        # - Bottom temperature (eMOLT data)
        # - Ocean currents
        # - Historical landings
        # - Bathymetry
        
        # For POC, we'll use fishing intensity as proxy
        catchability_scores = {}
        
        # Placeholder implementation
        return {
            'method': 'fishing_intensity_proxy',
            'description': 'Catchability estimated from fishing effort density',
            'scores': catchability_scores
        }
