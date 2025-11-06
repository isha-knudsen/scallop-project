"""
Marine Cadastre AIS Data Fetcher
Handles downloading and parsing AIS data from NOAA Marine Cadastre
"""

import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import zipfile
import io
import pyarrow.parquet as pq
from typing import Optional, Dict, Tuple
import asyncio
import aiohttp
from shapely.geometry import box

logger = logging.getLogger(__name__)

class MarineCadastreFetcher:
    """Fetches AIS data from Marine Cadastre"""
    
    # Base URLs for different data sources
    BASE_URL_CSV = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"
    BASE_URL_PARQUET = "https://marinecadastre.gov/downloads/ais2024"
    AZURE_URL = "https://marinecadastre.blob.core.windows.net/ais"
    
    # Georges Bank region bounds
    GEORGES_BANK = {
        'min_lat': 39.5,
        'max_lat': 42.5,
        'min_lon': -71.0,
        'max_lon': -66.0,
        'utm_zone': 18
    }
    
    # Mid-Atlantic region bounds
    MID_ATLANTIC = {
        'min_lat': 35.0,
        'max_lat': 41.0,
        'min_lon': -76.0,
        'max_lon': -70.0,
        'utm_zone': 18
    }
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_region_bounds(self, region: str) -> Dict:
        """Get bounding box for specified region"""
        if region == "georges_bank":
            return self.GEORGES_BANK
        elif region == "mid_atlantic":
            return self.MID_ATLANTIC
        else:
            raise ValueError(f"Unknown region: {region}")
    
    async def fetch_ais_data(
        self, 
        start_date: str, 
        end_date: str, 
        region: str = "georges_bank",
        data_format: str = "auto"
    ) -> pd.DataFrame:
        """
        Fetch AIS data for specified date range and region
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            region: Region name (georges_bank or mid_atlantic)
            data_format: 'csv', 'parquet', or 'auto'
        
        Returns:
            DataFrame with AIS data
        """
        logger.info(f"Fetching AIS data from {start_date} to {end_date} for {region}")
        
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Determine format based on year
        if data_format == "auto":
            if start.year >= 2025:
                data_format = "parquet"
            else:
                data_format = "csv"
        
        # Get region bounds
        bounds = self.get_region_bounds(region)
        
        # Fetch data based on format
        if data_format == "parquet":
            data = await self._fetch_parquet_data(start, end, bounds)
        else:
            data = await self._fetch_csv_data(start, end, bounds)
        
        logger.info(f"Fetched {len(data)} AIS records")
        
        return data
    
    async def _fetch_parquet_data(
        self, 
        start_date: datetime, 
        end_date: datetime,
        bounds: Dict
    ) -> pd.DataFrame:
        """Fetch data in GeoParquet format (2024+ data)"""
        
        all_data = []
        current_date = start_date
        
        async with aiohttp.ClientSession() as session:
            while current_date <= end_date:
                # Construct URL for daily file
                date_str = current_date.strftime("%Y_%m_%d")
                url = f"{self.BASE_URL_PARQUET}/AIS_{date_str}.parquet"
                
                cache_file = self.cache_dir / f"AIS_{date_str}.parquet"
                
                # Check cache
                if cache_file.exists():
                    logger.info(f"Loading from cache: {cache_file}")
                    df = pd.read_parquet(cache_file)
                else:
                    try:
                        logger.info(f"Downloading: {url}")
                        async with session.get(url) as response:
                            if response.status == 200:
                                content = await response.read()
                                df = pd.read_parquet(io.BytesIO(content))
                                
                                # Cache the file
                                df.to_parquet(cache_file)
                            else:
                                logger.warning(f"Failed to download {url}: {response.status}")
                                current_date += timedelta(days=1)
                                continue
                    except Exception as e:
                        logger.error(f"Error downloading {url}: {str(e)}")
                        current_date += timedelta(days=1)
                        continue
                
                # Filter to region
                df_filtered = self._filter_to_region(df, bounds)
                all_data.append(df_filtered)
                
                current_date += timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        return combined_data
    
    async def _fetch_csv_data(
        self,
        start_date: datetime,
        end_date: datetime,
        bounds: Dict
    ) -> pd.DataFrame:
        """Fetch data in CSV format (pre-2024 data)"""
        
        all_data = []
        
        async with aiohttp.ClientSession() as session:
            # Check if we need daily files (2024+) or monthly zone files (pre-2024)
            if start_date.year >= 2024:
                # Use daily files for 2024+
                current_date = start_date
                while current_date <= end_date:
                    date_str = current_date.strftime('%Y_%m_%d')
                    filename = f'AIS_{date_str}.zip'
                    url = f'{self.BASE_URL_CSV}/{current_date.year}/{filename}'
                    
                    cache_file = self.cache_dir / filename
                    
                    # Check cache
                    if cache_file.exists():
                        logger.info(f'Loading from cache: {cache_file}')
                        df = self._read_zipped_csv(cache_file)
                    else:
                        try:
                            logger.info(f'Downloading: {url}')
                            async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as response:
                                if response.status == 200:
                                    content = await response.read()
                                    with open(cache_file, 'wb') as f:
                                        f.write(content)
                                    df = self._read_zipped_csv(cache_file)
                                else:
                                    logger.warning(f'Failed to download {url}: {response.status}')
                                    current_date += timedelta(days=1)
                                    continue
                        except Exception as e:
                            logger.error(f'Error downloading {url}: {str(e)}')
                            current_date += timedelta(days=1)
                            continue
                    
                    # Filter to region
                    df_filtered = self._filter_to_region(df, bounds)
                    all_data.append(df_filtered)
                    
                    current_date += timedelta(days=1)
            else:
                # Use monthly zone files for pre-2024
                current_month = start_date.replace(day=1)
                end_month = end_date.replace(day=1)
                utm_zone = bounds['utm_zone']
                
                while current_month <= end_month:
                    year = current_month.year
                    month = current_month.strftime("%m")
                    filename = f"AIS_{year}_{month}_Zone{utm_zone}.zip"
                    url = f"{self.BASE_URL_CSV}/{year}/{filename}"
                    
                    cache_file = self.cache_dir / filename
                    
                    if cache_file.exists():
                        logger.info(f"Loading from cache: {cache_file}")
                        df = self._read_zipped_csv(cache_file)
                    else:
                        try:
                            logger.info(f"Downloading: {url}")
                            async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as response:
                                if response.status == 200:
                                    content = await response.read()
                                    with open(cache_file, 'wb') as f:
                                        f.write(content)
                                    df = self._read_zipped_csv(cache_file)
                                else:
                                    logger.warning(f"Failed to download {url}: {response.status}")
                                    current_month += timedelta(days=32)
                                    current_month = current_month.replace(day=1)
                                    continue
                        except Exception as e:
                            logger.error(f"Error downloading {url}: {str(e)}")
                            current_month += timedelta(days=32)
                            current_month = current_month.replace(day=1)
                            continue
                    
                    df_filtered = self._filter_to_region(df, bounds)
                    df_filtered = df_filtered[
                        (pd.to_datetime(df_filtered['BaseDateTime']) >= start_date) &
                        (pd.to_datetime(df_filtered['BaseDateTime']) <= end_date)
                    ]
                    all_data.append(df_filtered)
                    
                    current_month += timedelta(days=32)
                    current_month = current_month.replace(day=1)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        return combined_data
    
    def _read_zipped_csv(self, zip_path: Path) -> pd.DataFrame:
        """Read CSV from zip file"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get the CSV filename (should be only one)
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No CSV file found in {zip_path}")
            
            csv_filename = csv_files[0]
            
            # Read CSV
            with zip_ref.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file, low_memory=False)
        
        return df
    
    def _filter_to_region(self, df: pd.DataFrame, bounds: Dict) -> pd.DataFrame:
        """Filter dataframe to geographic region"""
        
        # Ensure LAT and LON columns exist
        lat_col = 'LAT' if 'LAT' in df.columns else 'lat'
        lon_col = 'LON' if 'LON' in df.columns else 'lon'
        
        if lat_col not in df.columns or lon_col not in df.columns:
            logger.warning("No lat/lon columns found in data")
            return df
        
        # Filter to bounds
        mask = (
            (df[lat_col] >= bounds['min_lat']) &
            (df[lat_col] <= bounds['max_lat']) &
            (df[lon_col] >= bounds['min_lon']) &
            (df[lon_col] <= bounds['max_lon'])
        )
        
        return df[mask].copy()
    
    def get_available_dates(self, year: int) -> Dict[str, list]:
        """Get list of available data files for a given year"""
        try:
            if year >= 2024:
                # Check parquet files
                url = f"{self.BASE_URL_PARQUET}/"
                response = requests.get(url)
                # Parse HTML to find available files
                # This is simplified - actual implementation would parse the directory listing
                return {"format": "parquet", "files": []}
            else:
                # Check CSV files
                url = f"{self.BASE_URL_CSV}/{year}/"
                response = requests.get(url)
                return {"format": "csv", "files": []}
        except Exception as e:
            logger.error(f"Error checking available dates: {str(e)}")
            return {"format": "unknown", "files": []}
    
    def estimate_data_size(self, start_date: str, end_date: str, region: str) -> Dict:
        """Estimate the size of data to be downloaded"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        num_days = (end - start).days + 1
        
        # Rough estimates based on typical file sizes
        if start.year >= 2025:
            # Parquet files are ~100-200 MB per day for full coverage
            # But filtered to region might be ~10-20 MB per day
            estimated_mb = num_days * 15
        else:
            # CSV files are larger, ~1-3 GB per month
            # Filtered might be ~50-100 MB per day
            estimated_mb = num_days * 75
        
        return {
            "estimated_size_mb": estimated_mb,
            "estimated_size_gb": round(estimated_mb / 1024, 2),
            "num_days": num_days,
            "recommendation": "Sample smaller date range for testing" if estimated_mb > 1000 else "Proceed with full download"
        }
