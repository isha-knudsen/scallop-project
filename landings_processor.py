"""
Landings Data Processor
Processes Portland Fish Exchange landings data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class LandingsProcessor:
    """Processes scallop landings data"""
    
    def __init__(self):
        self.data = None
        self.is_loaded = False
    
    def load_data(self, file_path: str):
        """Load landings data from file"""
        logger.info(f"Loading landings data from {file_path}")
        
        # Try to load as different formats
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.pdf'):
            # Extract from PDF (the uploaded file is PDF)
            df = self._extract_from_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Process data
        self.data = self._process_landings(df)
        self.is_loaded = True
        
        logger.info(f"Loaded {len(self.data)} landings records")
    
    def _extract_from_pdf(self, file_path: str) -> pd.DataFrame:
        """Extract table data from PDF"""
        import tabula
        
        # Extract tables from PDF
        tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
        
        # Combine all tables
        df = pd.concat(tables, ignore_index=True)
        
        return df
    
    def _process_landings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean landings data"""
        
        # Filter to scallop records
        scallop_mask = df['FishDesc'].str.contains('Scallop', case=False, na=False)
        df = df[scallop_mask].copy()
        
        # Create date column
        df['Date'] = pd.to_datetime(
            df['YearNum'].astype(str) + '-' + df['MonthNum'].astype(str) + '-01'
        )
        
        # Ensure numeric columns
        numeric_cols = ['Sold', 'AvgPrice', 'LowPrice', 'HighPrice']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate total value
        df['TotalValue'] = df['Sold'] * df['AvgPrice']
        
        # Sort by date
        df = df.sort_values('Date')
        
        return df
    
    def get_record_count(self) -> int:
        """Get number of landings records"""
        return len(self.data) if self.data is not None else 0
    
    def get_date_range(self) -> Optional[Tuple[str, str]]:
        """Get date range of landings data"""
        if self.data is None or len(self.data) == 0:
            return None
        
        return (
            self.data['Date'].min().strftime('%Y-%m-%d'),
            self.data['Date'].max().strftime('%Y-%m-%d')
        )
    
    def get_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """Get summary statistics for landings"""
        
        if self.data is None:
            return {"error": "No data loaded"}
        
        df = self.data.copy()
        
        # Apply date filters
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        if len(df) == 0:
            return {"error": "No data in date range"}
        
        summary = {
            "total_records": len(df),
            "date_range": {
                "start": df['Date'].min().strftime('%Y-%m-%d'),
                "end": df['Date'].max().strftime('%Y-%m-%d')
            },
            "total_pounds": float(df['Sold'].sum()),
            "total_value": float(df['TotalValue'].sum()),
            "average_price": float(df['AvgPrice'].mean()),
            "price_range": {
                "min": float(df['LowPrice'].min()),
                "max": float(df['HighPrice'].max())
            },
            "by_month": df.groupby(df['Date'].dt.to_period('M')).agg({
                'Sold': 'sum',
                'TotalValue': 'sum',
                'AvgPrice': 'mean'
            }).to_dict('index')
        }
        
        return summary
    
    def get_landings_for_period(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get landings data for specific period"""
        
        if self.data is None:
            return pd.DataFrame()
        
        df = self.data.copy()
        df = df[
            (df['Date'] >= pd.to_datetime(start_date)) &
            (df['Date'] <= pd.to_datetime(end_date))
        ]
        
        return df
    
    def get_biomass_estimates(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Generate biomass estimates based on landings
        
        This is a simplified model for POC
        In production, would integrate:
        - Stock assessments
        - Survey data
        - Recruitment models
        """
        
        # For now, create a simple spatial distribution based on landings volume
        landings = self.get_landings_for_period(start_date, end_date)
        
        if len(landings) == 0:
            return pd.DataFrame()
        
        # Create hypothetical biomass grid
        # Georges Bank region
        lat_min, lat_max = 39.5, 42.5
        lon_min, lon_max = -71.0, -66.0
        
        # Create grid
        grid_size = 0.1
        lats = np.arange(lat_min, lat_max, grid_size)
        lons = np.arange(lon_min, lon_max, grid_size)
        
        # Create grid points
        grid_points = []
        for lat in lats:
            for lon in lons:
                # Assign density based on proximity to known productive areas
                # For POC, use random distribution weighted by total landings
                density = np.random.beta(2, 5) * (landings['Sold'].sum() / 100000)
                
                grid_points.append({
                    'lat': lat,
                    'lon': lon,
                    'density': density
                })
        
        biomass_df = pd.DataFrame(grid_points)
        
        return biomass_df
    
    def correlate_with_ais(
        self,
        fishing_activity: pd.DataFrame,
        landings: pd.DataFrame
    ) -> Dict:
        """
        Correlate AIS fishing activity with landings data
        
        Args:
            fishing_activity: DataFrame with date and fishing hours
            landings: DataFrame with landings data
        
        Returns:
            Correlation statistics
        """
        
        # Aggregate by month
        fishing_monthly = fishing_activity.groupby(
            pd.Grouper(key='date', freq='M')
        )['fishing_hours'].sum()
        
        landings_monthly = landings.groupby(
            landings['Date'].dt.to_period('M')
        )['Sold'].sum()
        
        # Align indices
        common_months = fishing_monthly.index.intersection(
            landings_monthly.index.to_timestamp()
        )
        
        if len(common_months) < 2:
            return {
                "correlation": None,
                "message": "Insufficient overlapping data"
            }
        
        fishing_values = fishing_monthly.loc[common_months].values
        landings_values = landings_monthly.loc[
            common_months.to_period('M')
        ].values
        
        # Calculate correlation
        correlation = np.corrcoef(fishing_values, landings_values)[0, 1]
        
        return {
            "correlation": float(correlation),
            "n_months": len(common_months),
            "interpretation": self._interpret_correlation(correlation),
            "fishing_activity": {
                "mean": float(fishing_values.mean()),
                "total": float(fishing_values.sum())
            },
            "landings": {
                "mean": float(landings_values.mean()),
                "total": float(landings_values.sum())
            }
        }
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient"""
        if abs(r) < 0.3:
            return "weak"
        elif abs(r) < 0.7:
            return "moderate"
        else:
            return "strong"
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the full landings dataset"""
        return self.data
