"""
AIS Data Processor
Processes raw AIS data, identifies scallop vessels, and manages trajectory data
"""

import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
import geopandas as gpd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

class AISProcessor:
    """Processes AIS data and manages vessel trajectories"""
    
    # Scallop dredging speed range (knots)
    SCALLOP_SPEED_MIN = 3.5
    SCALLOP_SPEED_MAX = 4.5
    SCALLOP_SPEED_TYPICAL_MIN = 3.0
    SCALLOP_SPEED_TYPICAL_MAX = 5.0
    
    # Speed variance threshold for active dredging
    MAX_SPEED_VARIANCE = 1.5
    
    # Minimum pings per vessel to be considered
    MIN_PINGS_PER_VESSEL = 50
    
    def __init__(self):
        self.data = None
        self.scallop_vessels = None
        self.vessel_metadata = {}
        
    def process_raw_ais(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw AIS data
        
        Args:
            raw_data: Raw AIS DataFrame from Marine Cadastre
        
        Returns:
            Processed DataFrame with standardized columns
        """
        logger.info(f"Processing {len(raw_data)} raw AIS records")
        
        df = raw_data.copy()
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Parse timestamp
        if 'BaseDateTime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['BaseDateTime'])
        elif 'timestamp' not in df.columns:
            raise ValueError("No timestamp column found")
        
        # Ensure required columns exist
        required_cols = ['MMSI', 'LAT', 'LON', 'SOG', 'COG', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter invalid positions
        df = df[
            (df['LAT'].notna()) & 
            (df['LON'].notna()) &
            (df['LAT'] >= -90) & 
            (df['LAT'] <= 90) &
            (df['LON'] >= -180) & 
            (df['LON'] <= 180)
        ].copy()
        
        # Filter invalid speeds (negative or unrealistic)
        df = df[(df['SOG'] >= 0) & (df['SOG'] < 50)].copy()
        
        # Sort by vessel and time
        df = df.sort_values(['MMSI', 'timestamp']).reset_index(drop=True)
        
        # Calculate time differences
        df['time_diff'] = df.groupby('MMSI')['timestamp'].diff().dt.total_seconds()
        
        # Calculate distances between consecutive points
        df['distance'] = self._calculate_distances(df)
        
        # Filter out unrealistic jumps (likely data errors)
        # Max speed ~ 50 knots = 25.7 m/s
        max_distance_per_second = 30  # meters
        df['max_expected_distance'] = df['time_diff'] * max_distance_per_second
        df = df[
            (df['time_diff'].isna()) |  # Keep first point of each vessel
            (df['distance'] <= df['max_expected_distance'])
        ].copy()
        
        # Create geometry column
        df['geometry'] = df.apply(lambda row: Point(row['LON'], row['LAT']), axis=1)
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        
        logger.info(f"Processed {len(gdf)} valid AIS records")
        
        self.data = gdf
        return gdf
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different data sources"""
        column_mapping = {
            'lat': 'LAT',
            'latitude': 'LAT',
            'lon': 'LON',
            'longitude': 'LON',
            'mmsi': 'MMSI',
            'sog': 'SOG',
            'speed': 'SOG',
            'cog': 'COG',
            'course': 'COG',
            'heading': 'Heading',
            'vessel_name': 'VesselName',
            'vesselname': 'VesselName',
            'vessel_type': 'VesselType',
            'vesseltype': 'VesselType',
            'status': 'Status',
            'basedatetime': 'BaseDateTime',
            'datetime': 'BaseDateTime'
        }
        
        df = df.rename(columns=column_mapping)
        return df
    
    def _calculate_distances(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distances between consecutive points using Haversine formula"""
        
        def haversine(lat1, lon1, lat2, lon2):
            """Calculate distance in meters"""
            R = 6371000  # Earth radius in meters
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        
        distances = []
        for mmsi, group in df.groupby('MMSI'):
            group = group.sort_values('timestamp')
            lats = group['LAT'].values
            lons = group['LON'].values
            
            dists = [np.nan]  # First point has no previous point
            for i in range(1, len(group)):
                dist = haversine(lats[i-1], lons[i-1], lats[i], lons[i])
                dists.append(dist)
            
            distances.extend(dists)
        
        return pd.Series(distances, index=df.index)
    
    def identify_scallop_vessels(
        self, 
        data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Identify scallop dredging vessels based on speed patterns
        
        Args:
            data: Processed AIS data (uses self.data if None)
        
        Returns:
            DataFrame filtered to scallop vessels
        """
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data available. Run process_raw_ais first.")
        
        logger.info("Identifying scallop vessels...")
        
        # Filter to fishing vessels (VesselType = 30 if available)
        if 'VesselType' in data.columns:
            fishing_vessels = data[data['VesselType'] == 30].copy()
        else:
            fishing_vessels = data.copy()
        
        # Calculate speed statistics per vessel
        vessel_stats = fishing_vessels.groupby('MMSI').agg({
            'SOG': ['mean', 'std', 'median', 'count'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        vessel_stats.columns = ['MMSI', 'mean_speed', 'std_speed', 'median_speed', 
                                'ping_count', 'first_seen', 'last_seen']
        
        # Identify scallop vessels
        # Criteria:
        # 1. Average speed in typical scallop range
        # 2. Low speed variance (consistent dredging)
        # 3. Sufficient data points
        scallop_mask = (
            (vessel_stats['mean_speed'] >= self.SCALLOP_SPEED_TYPICAL_MIN) &
            (vessel_stats['mean_speed'] <= self.SCALLOP_SPEED_TYPICAL_MAX) &
            (vessel_stats['std_speed'] <= self.MAX_SPEED_VARIANCE) &
            (vessel_stats['ping_count'] >= self.MIN_PINGS_PER_VESSEL)
        )
        
        scallop_mmsis = vessel_stats[scallop_mask]['MMSI'].tolist()
        
        logger.info(f"Identified {len(scallop_mmsis)} potential scallop vessels")
        
        # Filter data to scallop vessels
        scallop_data = fishing_vessels[fishing_vessels['MMSI'].isin(scallop_mmsis)].copy()
        
        # Store metadata
        self.vessel_metadata = vessel_stats[scallop_mask].set_index('MMSI').to_dict('index')
        
        self.scallop_vessels = scallop_data
        return scallop_data
    
    def get_record_count(self) -> int:
        """Get total number of AIS records"""
        return len(self.data) if self.data is not None else 0
    
    def get_vessel_list(
        self, 
        date_filter: Optional[str] = None,
        behavior: Optional[str] = None
    ) -> List[Dict]:
        """Get list of vessels with metadata"""
        if self.scallop_vessels is None:
            return []
        
        data = self.scallop_vessels.copy()
        
        # Apply date filter
        if date_filter:
            filter_date = pd.to_datetime(date_filter)
            data = data[data['timestamp'].dt.date == filter_date.date()]
        
        # Apply behavior filter
        if behavior and 'behavior' in data.columns:
            data = data[data['behavior'] == behavior]
        
        # Get unique vessels
        vessels = []
        for mmsi in data['MMSI'].unique():
            vessel_data = data[data['MMSI'] == mmsi]
            
            vessel_info = {
                'mmsi': str(mmsi),
                'vessel_name': vessel_data['VesselName'].iloc[0] if 'VesselName' in vessel_data else None,
                'record_count': len(vessel_data),
                'first_seen': vessel_data['timestamp'].min().isoformat(),
                'last_seen': vessel_data['timestamp'].max().isoformat(),
                'mean_speed': float(vessel_data['SOG'].mean()),
                'median_speed': float(vessel_data['SOG'].median())
            }
            
            if mmsi in self.vessel_metadata:
                vessel_info.update(self.vessel_metadata[mmsi])
            
            vessels.append(vessel_info)
        
        return vessels
    
    def get_date_range(self) -> Optional[Tuple[str, str]]:
        """Get date range of loaded data"""
        if self.data is None or len(self.data) == 0:
            return None
        
        return (
            self.data['timestamp'].min().strftime('%Y-%m-%d'),
            self.data['timestamp'].max().strftime('%Y-%m-%d')
        )
    
    def get_vessel_trajectory(
        self,
        mmsi: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """Get trajectory data for specific vessel"""
        if self.scallop_vessels is None:
            raise ValueError("No scallop vessel data available")
        
        # Filter to vessel
        vessel_data = self.scallop_vessels[self.scallop_vessels['MMSI'] == int(mmsi)].copy()
        
        if len(vessel_data) == 0:
            raise ValueError(f"Vessel {mmsi} not found")
        
        # Apply date filters
        if start_date:
            vessel_data = vessel_data[vessel_data['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            vessel_data = vessel_data[vessel_data['timestamp'] <= pd.to_datetime(end_date)]
        
        # Sort by time
        vessel_data = vessel_data.sort_values('timestamp')
        
        # Extract positions
        positions = []
        for _, row in vessel_data.iterrows():
            positions.append({
                'timestamp': row['timestamp'].isoformat(),
                'lat': float(row['LAT']),
                'lon': float(row['LON']),
                'speed': float(row['SOG']),
                'course': float(row['COG']) if pd.notna(row['COG']) else None,
                'behavior': row.get('behavior', 'unknown')
            })
        
        # Identify fishing events (clusters of low-speed, consistent heading)
        fishing_events = self._identify_fishing_events(vessel_data)
        
        # Calculate statistics
        statistics = {
            'total_distance_nm': float(vessel_data['distance'].sum() / 1852),  # Convert m to nm
            'total_time_hours': (vessel_data['timestamp'].max() - vessel_data['timestamp'].min()).total_seconds() / 3600,
            'mean_speed_knots': float(vessel_data['SOG'].mean()),
            'max_speed_knots': float(vessel_data['SOG'].max()),
            'num_fishing_events': len(fishing_events)
        }
        
        return {
            'vessel_name': vessel_data['VesselName'].iloc[0] if 'VesselName' in vessel_data else None,
            'date_range': (
                vessel_data['timestamp'].min().isoformat(),
                vessel_data['timestamp'].max().isoformat()
            ),
            'positions': positions,
            'fishing_events': fishing_events,
            'statistics': statistics
        }
    
    def _identify_fishing_events(self, vessel_data: pd.DataFrame) -> List[Dict]:
        """Identify fishing events from vessel trajectory"""
        events = []
        
        # Mark fishing activity (speed in dredging range)
        vessel_data['is_fishing'] = (
            (vessel_data['SOG'] >= self.SCALLOP_SPEED_MIN) &
            (vessel_data['SOG'] <= self.SCALLOP_SPEED_MAX)
        )
        
        # Find continuous fishing periods
        vessel_data['fishing_group'] = (
            vessel_data['is_fishing'] != vessel_data['is_fishing'].shift()
        ).cumsum()
        
        for group_id, group in vessel_data[vessel_data['is_fishing']].groupby('fishing_group'):
            if len(group) < 3:  # Need at least 3 points
                continue
            
            event = {
                'start_time': group['timestamp'].min().isoformat(),
                'end_time': group['timestamp'].max().isoformat(),
                'duration_hours': (group['timestamp'].max() - group['timestamp'].min()).total_seconds() / 3600,
                'start_position': {
                    'lat': float(group.iloc[0]['LAT']),
                    'lon': float(group.iloc[0]['LON'])
                },
                'end_position': {
                    'lat': float(group.iloc[-1]['LAT']),
                    'lon': float(group.iloc[-1]['LON'])
                },
                'mean_speed': float(group['SOG'].mean()),
                'distance_nm': float(group['distance'].sum() / 1852)
            }
            events.append(event)
        
        return events
    
    def generate_fishing_heatmap(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resolution: float = 0.01
    ) -> Dict:
        """Generate fishing activity heatmap"""
        data = self.scallop_vessels.copy()
        
        if start_date:
            data = data[data['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['timestamp'] <= pd.to_datetime(end_date)]
        
        # Filter to fishing behavior
        if 'behavior' in data.columns:
            fishing_data = data[data['behavior'] == 'fishing']
        else:
            # Use speed as proxy
            fishing_data = data[
                (data['SOG'] >= self.SCALLOP_SPEED_MIN) &
                (data['SOG'] <= self.SCALLOP_SPEED_MAX)
            ]
        
        # Create grid
        min_lat, max_lat = fishing_data['LAT'].min(), fishing_data['LAT'].max()
        min_lon, max_lon = fishing_data['LON'].min(), fishing_data['LON'].max()
        
        lat_bins = np.arange(min_lat, max_lat + resolution, resolution)
        lon_bins = np.arange(min_lon, max_lon + resolution, resolution)
        
        # Count fishing activity in each cell
        H, _, _ = np.histogram2d(
            fishing_data['LAT'],
            fishing_data['LON'],
            bins=[lat_bins, lon_bins]
        )
        
        return {
            'bounds': {
                'min_lat': float(min_lat),
                'max_lat': float(max_lat),
                'min_lon': float(min_lon),
                'max_lon': float(max_lon)
            },
            'grid': {
                'lat_bins': lat_bins.tolist(),
                'lon_bins': lon_bins.tolist()
            },
            'intensity': H.tolist()
        }
    
    def get_data_for_period(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get all data for a specific period"""
        if self.scallop_vessels is None:
            raise ValueError("No data available")
        
        data = self.scallop_vessels.copy()
        data = data[
            (data['timestamp'] >= pd.to_datetime(start_date)) &
            (data['timestamp'] <= pd.to_datetime(end_date))
        ]
        
        return data
    
    def to_geojson(self, data: pd.DataFrame) -> Dict:
        """Convert data to GeoJSON format"""
        if not isinstance(data, gpd.GeoDataFrame):
            data = gpd.GeoDataFrame(
                data,
                geometry=gpd.points_from_xy(data['LON'], data['LAT']),
                crs='EPSG:4326'
            )
        
        return json.loads(data.to_json())
