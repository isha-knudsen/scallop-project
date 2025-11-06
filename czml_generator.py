"""
CZML Generator
Generates Cesium Markup Language files for 3D visualization
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

class CZMLGenerator:
    """Generates CZML files for Cesium visualization"""
    
    # Color schemes
    COLORS = {
        'fishing': [255, 0, 0, 200],      # Red
        'steaming': [0, 255, 0, 150],     # Green
        'maneuvering': [255, 255, 0, 150], # Yellow
        'unknown': [128, 128, 128, 100],   # Gray
        'biomass_high': [255, 0, 0, 180],
        'biomass_medium': [255, 165, 0, 150],
        'biomass_low': [255, 255, 0, 120]
    }
    
    def __init__(self):
        pass
    
    def generate_czml(
        self,
        vessel_data: Optional[pd.DataFrame] = None,
        biomass_data: Optional[pd.DataFrame] = None,
        rotational_areas: Optional[List[Dict]] = None,
        start_time: str = None,
        end_time: str = None
    ) -> List[Dict]:
        """
        Generate complete CZML document
        
        Args:
            vessel_data: DataFrame with vessel trajectories
            biomass_data: DataFrame with biomass estimates
            rotational_areas: List of rotational area polygons
            start_time: Start time for animation
            end_time: End time for animation
        
        Returns:
            List of CZML entities
        """
        logger.info("Generating CZML document")
        
        czml = []
        
        # Document header
        document = self._create_document_header(start_time, end_time)
        czml.append(document)
        
        # Add vessel trajectories
        if vessel_data is not None:
            vessel_entities = self._create_vessel_entities(vessel_data)
            czml.extend(vessel_entities)
        
        # Add biomass visualization
        if biomass_data is not None:
            biomass_entities = self._create_biomass_entities(biomass_data)
            czml.extend(biomass_entities)
        
        # Add rotational areas
        if rotational_areas is not None:
            area_entities = self._create_rotational_area_entities(rotational_areas)
            czml.extend(area_entities)
        
        logger.info(f"Generated CZML with {len(czml)} entities")
        
        return czml
    
    def _create_document_header(
        self,
        start_time: Optional[str],
        end_time: Optional[str]
    ) -> Dict:
        """Create CZML document header with clock settings"""
        
        if start_time is None:
            start_time = datetime.now().isoformat()
        if end_time is None:
            end_time = (datetime.now() + timedelta(days=30)).isoformat()
        
        return {
            "id": "document",
            "name": "Scallop Dredge Behavior Visualization",
            "version": "1.0",
            "clock": {
                "interval": f"{start_time}/{end_time}",
                "currentTime": start_time,
                "multiplier": 3600,  # Speed up time
                "range": "LOOP_STOP",
                "step": "SYSTEM_CLOCK_MULTIPLIER"
            }
        }
    
    def _create_vessel_entities(self, vessel_data: pd.DataFrame) -> List[Dict]:
        """Create CZML entities for vessel trajectories"""
        
        entities = []
        
        # Group by vessel
        for mmsi, group in vessel_data.groupby('MMSI'):
            group = group.sort_values('timestamp')
            
            # Get vessel color based on behavior
            primary_behavior = group['behavior'].mode()[0] if 'behavior' in group else 'unknown'
            color = self.COLORS.get(primary_behavior, self.COLORS['unknown'])
            
            # Create position cartographicDegrees array
            # Format: [time1, lon1, lat1, height1, time2, lon2, lat2, height2, ...]
            positions = []
            for _, row in group.iterrows():
                positions.extend([
                    row['timestamp'].isoformat(),
                    float(row['LON']),
                    float(row['LAT']),
                    0  # Height above sea level
                ])
            
            # Create vessel entity
            vessel_name = group['VesselName'].iloc[0] if 'VesselName' in group else f"Vessel {mmsi}"
            
            entity = {
                "id": f"vessel_{mmsi}",
                "name": vessel_name,
                "availability": f"{group['timestamp'].min().isoformat()}/{group['timestamp'].max().isoformat()}",
                "description": f"<p>MMSI: {mmsi}</p><p>Behavior: {primary_behavior}</p>",
                
                # Path (trail)
                "path": {
                    "show": True,
                    "width": 2,
                    "material": {
                        "solidColor": {
                            "color": {
                                "rgba": color
                            }
                        }
                    },
                    "resolution": 120,
                    "leadTime": 0,
                    "trailTime": 3600  # 1 hour trail
                },
                
                # Position over time
                "position": {
                    "epoch": group['timestamp'].min().isoformat(),
                    "cartographicDegrees": positions
                },
                
                # Point marker
                "point": {
                    "show": True,
                    "pixelSize": 8,
                    "color": {
                        "rgba": color
                    },
                    "outlineColor": {
                        "rgba": [255, 255, 255, 255]
                    },
                    "outlineWidth": 2
                },
                
                # Label
                "label": {
                    "show": True,
                    "text": vessel_name,
                    "font": "12pt sans-serif",
                    "fillColor": {
                        "rgba": [255, 255, 255, 255]
                    },
                    "outlineColor": {
                        "rgba": [0, 0, 0, 255]
                    },
                    "outlineWidth": 2,
                    "horizontalOrigin": "LEFT",
                    "verticalOrigin": "BOTTOM",
                    "pixelOffset": {
                        "cartesian2": [10, 0]
                    },
                    "distanceDisplayCondition": {
                        "distanceDisplayCondition": [0, 50000]  # Only show when zoomed in
                    }
                }
            }
            
            entities.append(entity)
            
            # Add fishing events as separate entities
            if 'behavior' in group.columns:
                fishing_events = self._create_fishing_event_entities(mmsi, group)
                entities.extend(fishing_events)
        
        return entities
    
    def _create_fishing_event_entities(
        self,
        mmsi: str,
        vessel_data: pd.DataFrame
    ) -> List[Dict]:
        """Create entities for fishing events (circles/zones)"""
        
        entities = []
        
        # Find continuous fishing periods
        fishing_data = vessel_data[vessel_data['behavior'] == 'fishing'].copy()
        
        if len(fishing_data) == 0:
            return entities
        
        # Group consecutive fishing points
        fishing_data['fishing_group'] = (
            fishing_data['timestamp'].diff() > pd.Timedelta(hours=1)
        ).cumsum()
        
        for group_id, group in fishing_data.groupby('fishing_group'):
            if len(group) < 3:
                continue
            
            # Calculate center point
            center_lat = group['LAT'].mean()
            center_lon = group['LON'].mean()
            
            # Calculate radius based on extent
            lat_range = group['LAT'].max() - group['LAT'].min()
            lon_range = group['LON'].max() - group['LON'].min()
            radius = max(lat_range, lon_range) * 111000 / 2  # Convert to meters
            radius = max(radius, 100)  # Minimum 100m
            
            entity = {
                "id": f"fishing_event_{mmsi}_{group_id}",
                "name": f"Fishing Event",
                "availability": f"{group['timestamp'].min().isoformat()}/{group['timestamp'].max().isoformat()}",
                "position": {
                    "cartographicDegrees": [center_lon, center_lat, 0]
                },
                "ellipse": {
                    "show": True,
                    "semiMinorAxis": radius,
                    "semiMajorAxis": radius,
                    "height": 0,
                    "material": {
                        "solidColor": {
                            "color": {
                                "rgba": [255, 0, 0, 80]  # Semi-transparent red
                            }
                        }
                    },
                    "outline": True,
                    "outlineColor": {
                        "rgba": [255, 0, 0, 150]
                    },
                    "outlineWidth": 2
                }
            }
            
            entities.append(entity)
        
        return entities
    
    def _create_biomass_entities(self, biomass_data: pd.DataFrame) -> List[Dict]:
        """
        Create 3D volumetric entities for scallop biomass distribution
        
        Scallops are benthic, so we place them on the seafloor with
        size/color varying by density
        """
        
        entities = []
        
        # Grid the region
        grid_size = 0.02  # degrees (~2km)
        
        min_lat, max_lat = biomass_data['lat'].min(), biomass_data['lat'].max()
        min_lon, max_lon = biomass_data['lon'].min(), biomass_data['lon'].max()
        
        lat_bins = np.arange(min_lat, max_lat, grid_size)
        lon_bins = np.arange(min_lon, max_lon, grid_size)
        
        # Create grid points
        for i, lat in enumerate(lat_bins):
            for j, lon in enumerate(lon_bins):
                # Get biomass estimate for this cell
                cell_data = biomass_data[
                    (biomass_data['lat'] >= lat) &
                    (biomass_data['lat'] < lat + grid_size) &
                    (biomass_data['lon'] >= lon) &
                    (biomass_data['lon'] < lon + grid_size)
                ]
                
                if len(cell_data) == 0:
                    continue
                
                density = cell_data['density'].mean()
                
                # Determine color and size based on density
                if density > 0.7:
                    color = self.COLORS['biomass_high']
                    pixel_size = 15
                elif density > 0.4:
                    color = self.COLORS['biomass_medium']
                    pixel_size = 10
                else:
                    color = self.COLORS['biomass_low']
                    pixel_size = 5
                
                entity = {
                    "id": f"biomass_{i}_{j}",
                    "name": f"Scallop Biomass (Density: {density:.2f})",
                    "position": {
                        "cartographicDegrees": [
                            lon + grid_size/2,
                            lat + grid_size/2,
                            -50  # Place at seafloor (negative = below sea level)
                        ]
                    },
                    "point": {
                        "show": True,
                        "pixelSize": pixel_size,
                        "color": {
                            "rgba": color
                        },
                        "heightReference": "CLAMP_TO_GROUND"
                    }
                }
                
                entities.append(entity)
        
        logger.info(f"Created {len(entities)} biomass grid points")
        
        return entities
    
    def _create_rotational_area_entities(
        self,
        rotational_areas: List[Dict]
    ) -> List[Dict]:
        """Create entities for scallop rotational closure areas"""
        
        entities = []
        
        for area in rotational_areas:
            # Get area status (open/closed)
            is_open = area.get('status') == 'open'
            color = [0, 255, 0, 100] if is_open else [255, 0, 0, 100]
            
            # Get polygon coordinates
            coords = area['geometry']['coordinates'][0]  # Exterior ring
            
            # Flatten coordinates for CZML
            positions = []
            for lon, lat in coords:
                positions.extend([lon, lat])
            
            entity = {
                "id": f"rotational_area_{area['id']}",
                "name": area['name'],
                "description": f"<p>Status: {'Open' if is_open else 'Closed'}</p><p>{area.get('description', '')}</p>",
                "polygon": {
                    "show": True,
                    "positions": {
                        "cartographicDegrees": positions
                    },
                    "material": {
                        "solidColor": {
                            "color": {
                                "rgba": color
                            }
                        }
                    },
                    "height": 0,
                    "outline": True,
                    "outlineColor": {
                        "rgba": [255, 255, 255, 200]
                    },
                    "outlineWidth": 2
                }
            }
            
            entities.append(entity)
        
        return entities
    
    def generate_heatmap_czml(
        self,
        heatmap_data: Dict,
        min_value: float,
        max_value: float
    ) -> List[Dict]:
        """Generate CZML for a heatmap overlay"""
        
        entities = []
        
        grid = heatmap_data['grid']
        intensity = np.array(heatmap_data['intensity'])
        
        lat_bins = grid['lat_bins']
        lon_bins = grid['lon_bins']
        
        # Normalize intensity
        intensity_norm = (intensity - min_value) / (max_value - min_value)
        intensity_norm = np.clip(intensity_norm, 0, 1)
        
        # Create colored grid cells
        for i in range(len(lat_bins) - 1):
            for j in range(len(lon_bins) - 1):
                value = intensity_norm[i, j]
                
                if value < 0.1:  # Skip low-intensity cells
                    continue
                
                # Color from blue (low) to red (high)
                r = int(255 * value)
                g = int(255 * (1 - value))
                b = 0
                a = int(150 * value)
                
                entity = {
                    "id": f"heatmap_{i}_{j}",
                    "polygon": {
                        "show": True,
                        "positions": {
                            "cartographicDegrees": [
                                lon_bins[j], lat_bins[i],
                                lon_bins[j+1], lat_bins[i],
                                lon_bins[j+1], lat_bins[i+1],
                                lon_bins[j], lat_bins[i+1]
                            ]
                        },
                        "material": {
                            "solidColor": {
                                "color": {
                                    "rgba": [r, g, b, a]
                                }
                            }
                        },
                        "height": 0
                    }
                }
                
                entities.append(entity)
        
        return entities
