import subprocess
from pathlib import Path
import numpy as np
import os

from configs import settings
from data_preparation.data_configs.data_settings import METER_NAD_CRS

def generate_tile_terrain(
    tile_id, 
    vrt_path, 
    out_dem_dir, 
    out_slope_dir, 
    out_aspect_dir, 
    base_res=90,
    grid_size=256
):
    """Crops perfectly aligned topography directly from the VRT based on the mathematical Tile ID.
    Calculates Slope, Aspect, and averages the DEM for the coarse channel.

    Args:
      tile_id: the tile identifier on the map.
      vrt_path: the path to the virtual DEM file.
      out_dem_dir: the output directory of the elevation tiles.
      out_slope_dir: the output directory of the slope tiles.
      out_aspect_dir: the output directory of the aspect tiles.
      base_res:  the target resolution in meters (Default value = 90)
      grid_size: the target grid size (Default value = 256)

    """
    # Create directories
    for directory in [out_dem_dir, out_slope_dir, out_aspect_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    out_dem_path = str(Path(out_dem_dir) / f"{tile_id}_dem_{base_res}m.tif")
    out_slope_path = str(Path(out_slope_dir) / f"{tile_id}_slope_{base_res}m.tif")
    out_aspect_path = str(Path(out_aspect_dir) / f"{tile_id}_aspect_{base_res}m.tif")

    # If the aspect already exists, this tile is fully processed, Skip it.
    if os.path.exists(out_aspect_path):
        print(f"[{tile_id}] Terrain already exists. Skipping.")
        return

    # THE GRID MATH
    # Parse the col and row from "tile_305_62"
    parts = tile_id.split('_')
    t_col = int(parts[1])
    t_row = int(parts[2])
    
    tile_span_m = grid_size * base_res # 256 * 90 = 23,040 meters
    
    # Calculate exact EPSG:3347 bounds for this 256x256 grid
    min_x_m = t_col * tile_span_m
    max_x_m = min_x_m + tile_span_m
    min_y_m = t_row * tile_span_m
    max_y_m = min_y_m + tile_span_m
    # -----------------------------

    try:
        # STEP 1: Crop and project standard DEM from VRT (Perfectly snapped)
        print(f"[{tile_id}] 1/3: Cropping DEM to exact grid limits...")
        subprocess.run([
            "gdalwarp", 
            "-t_srs", f"EPSG:{METER_NAD_CRS}",
            "-te", str(min_x_m), str(min_y_m), str(max_x_m), str(max_y_m), # Exact Bounds
            "-tr", str(base_res), str(base_res),
            "-tap", # Forces alignment to the resolution
            "-r", "cubicspline",
            "-srcnodata", "-9999", 
            "-dstnodata", "-9999",
            "-overwrite",
            str(vrt_path), out_dem_path   
        ], check=True, capture_output=True, text=True)

        # STEP 2: Calculate Slope
        print(f"[{tile_id}] 2/3: Calculating Slope...")
        subprocess.run([
            "gdaldem", "slope", 
            out_dem_path, out_slope_path,     
            "-alg", "Horn",
            "-compute_edges"              
        ], check=True, capture_output=True, text=True)

        # STEP 3: Calculate Aspect
        print(f"[{tile_id}] 3/3: Calculating Aspect...")
        subprocess.run([
            "gdaldem", "aspect", 
            out_dem_path, out_aspect_path,
            "-alg", "Horn",
            "-compute_edges",
            "-zero_for_flat"
        ], check=True, capture_output=True, text=True)

    except subprocess.CalledProcessError as e:
        print(f"[{tile_id}] ERROR: GDAL FAILED. Here is the exact error message:")
        print(e.stderr)

def process_fire(target_fire_id, fire_growth_pts):
    """Core logic to find all unique tiles for a single fire and generate their terrain.

    Args:
      target_fire_id: the target fire's ID
      fire_growth_pts: the CFSD dataframe

    """
    single_fire_df = fire_growth_pts[fire_growth_pts['ID'] == target_fire_id].copy()

    tile_span_m = settings.GRID_SIZE * settings.PIXEL_SIZE
    single_fire_df['tile_col'] = np.floor(single_fire_df[settings.X_COL] / tile_span_m).astype(int)
    single_fire_df['tile_row'] = np.floor(single_fire_df[settings.Y_COL] / tile_span_m).astype(int)

    unique_tiles = single_fire_df[['tile_col', 'tile_row']].drop_duplicates()
    
    print(f"  Found {len(unique_tiles)} tiles for Fire {target_fire_id}")

    for _, row in unique_tiles.iterrows():
        tile_id = f"tile_{row['tile_col']}_{row['tile_row']}"
        generate_tile_terrain(
            tile_id=tile_id, 
            vrt_path=f"{settings.DEM_FOLDER}/dem_mosaic.vrt", 
            out_dem_dir=settings.ELEVATION_FOLDER, 
            out_slope_dir=settings.SLOPE_FOLDER, 
            out_aspect_dir=settings.ASPECT_FOLDER,
            base_res=settings.PIXEL_SIZE,
            grid_size=settings.GRID_SIZE
        )