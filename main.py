"""
Scallop Dredge Behavior Analysis Backend API
Handles Marine Cadastre AIS data fetching, processing, and CZML generation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
import json
import os
import logging
from pathlib import Path

# Import custom modules
from data_fetcher import MarineCadastreFetcher
from ais_processor import AISProcessor
from behavioral_classifier import BehavioralClassifier
from czml_generator import CZMLGenerator
from landings_processor import LandingsProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Scallop Dredge Behavior API",
    description="Backend API for ground-truthing scallop dredge behavior using Marine Cadastre AIS data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_fetcher = MarineCadastreFetcher()
ais_processor = AISProcessor()
behavioral_classifier = BehavioralClassifier()
czml_generator = CZMLGenerator()
landings_processor = LandingsProcessor()

# Data storage paths
DATA_DIR = Path("../data")
DATA_DIR.mkdir(exist_ok=True)

# Pydantic models
class DateRangeRequest(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    region: str = "georges_bank"

class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str

class VesselTrajectory(BaseModel):
    mmsi: str
    vessel_name: Optional[str]
    positions: List[Dict[str, Any]]
    fishing_events: List[Dict[str, Any]]

# Global task storage (in production, use Redis or database)
processing_tasks = {}

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Scallop Dredge Behavior Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "status": "/api/status",
            "data": "/api/data/*",
            "vessels": "/api/vessels/*",
            "czml": "/api/czml/*",
            "landings": "/api/landings/*"
        }
    }

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "status": "operational",
        "data_sources": {
            "marine_cadastre": "connected",
            "landings_data": "loaded" if landings_processor.is_loaded else "not_loaded"
        },
        "cached_data": {
            "ais_records": ais_processor.get_record_count(),
            "vessels": len(ais_processor.get_vessel_list()),
            "date_range": ais_processor.get_date_range()
        }
    }

@app.post("/api/data/fetch")
async def fetch_ais_data(
    request: DateRangeRequest,
    background_tasks: BackgroundTasks
):
    """
    Fetch AIS data from Marine Cadastre for specified date range
    This runs as a background task
    """
    task_id = f"fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize task
    processing_tasks[task_id] = {
        "status": "initiated",
        "progress": 0.0,
        "message": "Starting data fetch...",
        "start_time": datetime.now().isoformat()
    }
    
    # Add background task
    background_tasks.add_task(
        fetch_and_process_data,
        task_id,
        request.start_date,
        request.end_date,
        request.region
    )
    
    return {
        "task_id": task_id,
        "message": "Data fetch initiated",
        "status_endpoint": f"/api/data/status/{task_id}"
    }

async def fetch_and_process_data(task_id: str, start_date: str, end_date: str, region: str):
    """Background task to fetch and process AIS data"""
    try:
        # Update status
        processing_tasks[task_id]["status"] = "fetching"
        processing_tasks[task_id]["progress"] = 0.1
        processing_tasks[task_id]["message"] = "Fetching data from Marine Cadastre..."
        
        # Fetch data
        logger.info(f"Fetching AIS data for {start_date} to {end_date}")
        raw_data = await data_fetcher.fetch_ais_data(start_date, end_date, region)
        
        processing_tasks[task_id]["progress"] = 0.4
        processing_tasks[task_id]["message"] = "Processing AIS data..."
        
        # Process data
        processed_data = ais_processor.process_raw_ais(raw_data)
        
        processing_tasks[task_id]["progress"] = 0.6
        processing_tasks[task_id]["message"] = "Identifying scallop vessels..."
        
        # Identify scallop vessels
        scallop_vessels = ais_processor.identify_scallop_vessels(processed_data)
        
        processing_tasks[task_id]["progress"] = 0.8
        processing_tasks[task_id]["message"] = "Classifying fishing behavior..."
        
        # Classify behavior
        classified_data = behavioral_classifier.classify_behavior(scallop_vessels)
        
        processing_tasks[task_id]["progress"] = 0.9
        processing_tasks[task_id]["message"] = "Saving processed data..."
        
        # Save to disk
        output_path = DATA_DIR / f"processed_{task_id}.parquet"
        classified_data.to_parquet(output_path)
        
        # Update final status
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["progress"] = 1.0
        processing_tasks[task_id]["message"] = "Data processing complete"
        processing_tasks[task_id]["output_file"] = str(output_path)
        processing_tasks[task_id]["end_time"] = datetime.now().isoformat()
        processing_tasks[task_id]["records_processed"] = len(classified_data)
        processing_tasks[task_id]["vessels_found"] = classified_data['MMSI'].nunique()
        
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in task {task_id}: {str(e)}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["message"] = f"Error: {str(e)}"

@app.get("/api/data/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a data processing task"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return processing_tasks[task_id]

@app.get("/api/vessels/list")
async def get_vessels(
    date_filter: Optional[str] = Query(None, description="YYYY-MM-DD"),
    behavior: Optional[str] = Query(None, description="fishing or steaming")
):
    """Get list of vessels with optional filters"""
    vessels = ais_processor.get_vessel_list(date_filter=date_filter, behavior=behavior)
    return {
        "count": len(vessels),
        "vessels": vessels
    }

@app.get("/api/vessels/{mmsi}/trajectory")
async def get_vessel_trajectory(
    mmsi: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get trajectory data for a specific vessel"""
    try:
        trajectory = ais_processor.get_vessel_trajectory(
            mmsi, 
            start_date=start_date, 
            end_date=end_date
        )
        
        return {
            "mmsi": mmsi,
            "vessel_name": trajectory.get('vessel_name'),
            "record_count": len(trajectory['positions']),
            "date_range": trajectory['date_range'],
            "positions": trajectory['positions'],
            "fishing_events": trajectory['fishing_events'],
            "statistics": trajectory['statistics']
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/vessels/{mmsi}/behavior")
async def get_vessel_behavior(mmsi: str):
    """Get behavioral classification for a vessel"""
    try:
        behavior_data = behavioral_classifier.get_vessel_behavior(mmsi)
        return behavior_data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/fishing/heatmap")
async def get_fishing_heatmap(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    resolution: float = 0.01  # degrees
):
    """Generate fishing activity heatmap"""
    heatmap_data = ais_processor.generate_fishing_heatmap(
        start_date=start_date,
        end_date=end_date,
        resolution=resolution
    )
    
    return {
        "type": "heatmap",
        "resolution": resolution,
        "bounds": heatmap_data['bounds'],
        "grid": heatmap_data['grid'],
        "intensity": heatmap_data['intensity']
    }

@app.get("/api/czml/generate")
async def generate_czml(
    start_date: str,
    end_date: str,
    include_vessels: bool = True,
    include_biomass: bool = True,
    include_rotational_areas: bool = True
):
    """Generate CZML file for Cesium visualization"""
    try:
        # Get processed data
        data = ais_processor.get_data_for_period(start_date, end_date)
        
        # Generate CZML
        czml_data = czml_generator.generate_czml(
            vessel_data=data if include_vessels else None,
            biomass_data=landings_processor.get_biomass_estimates(start_date, end_date) if include_biomass else None,
            rotational_areas=ais_processor.get_rotational_areas() if include_rotational_areas else None,
            start_time=start_date,
            end_time=end_date
        )
        
        # Save to file
        czml_path = DATA_DIR / f"scallop_viz_{start_date}_{end_date}.czml"
        with open(czml_path, 'w') as f:
            json.dump(czml_data, f)
        
        return FileResponse(
            path=czml_path,
            filename=f"scallop_viz_{start_date}_{end_date}.czml",
            media_type="application/json"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating CZML: {str(e)}")

@app.post("/api/landings/upload")
async def upload_landings_data(file_path: str):
    """Load landings data from file"""
    try:
        landings_processor.load_data(file_path)
        return {
            "status": "success",
            "records": landings_processor.get_record_count(),
            "date_range": landings_processor.get_date_range()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading landings data: {str(e)}")

@app.get("/api/landings/summary")
async def get_landings_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get summary statistics for landings data"""
    try:
        summary = landings_processor.get_summary(start_date=start_date, end_date=end_date)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/landings/correlation")
async def get_ais_landings_correlation(
    start_date: str,
    end_date: str
):
    """Correlate AIS fishing activity with landings data"""
    try:
        # Get fishing activity
        fishing_activity = ais_processor.get_fishing_activity_summary(start_date, end_date)
        
        # Get landings
        landings = landings_processor.get_landings_for_period(start_date, end_date)
        
        # Calculate correlation
        correlation = landings_processor.correlate_with_ais(fishing_activity, landings)
        
        return correlation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/compliance")
async def get_compliance_analysis(
    start_date: str,
    end_date: str
):
    """Analyze regulatory compliance (fishing in open vs closed areas)"""
    try:
        compliance_data = ais_processor.analyze_compliance(start_date, end_date)
        return compliance_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/catchability")
async def get_catchability_model(
    start_date: str,
    end_date: str
):
    """Get catchability scores for fishing locations"""
    try:
        catchability = behavioral_classifier.calculate_catchability(
            start_date=start_date,
            end_date=end_date,
            landings_data=landings_processor.get_data()
        )
        return catchability
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/geojson")
async def export_geojson(
    start_date: str,
    end_date: str,
    behavior_filter: Optional[str] = None
):
    """Export data as GeoJSON"""
    try:
        data = ais_processor.get_data_for_period(start_date, end_date)
        
        if behavior_filter:
            data = data[data['behavior'] == behavior_filter]
        
        geojson = ais_processor.to_geojson(data)
        
        return JSONResponse(content=geojson)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
