import os
import pandas as pd
import numpy as np
from pyproj import Transformer
from timezonefinder import TimezoneFinder
import pytz
from datetime import datetime, timedelta
import h5py
from data_preparation.data_configs.data_settings import DEG_CRS, DEG_NAD_CRS, METER_NAD_CRS

def local_to_utc(row, tf, lon_col, lat_col):
    """

    Args:
      row: the dataframe row
      tf: the timezone finder
      lon_col: the name of the longitude column
      lat_col: the name of the latitude column

    Returns: the new dataframe with the dates columns
    """
    
    if pd.isna(row[lat_col]) or pd.isna(row[lon_col]) or pd.isna(row['DOB']):
        return pd.Series([None, None, None, None])
    
    tz_name = tf.timezone_at(lat=row[lat_col], lng=row[lon_col])

    if tz_name is None:
        tz_name = "UTC"

    tz = pytz.timezone(tz_name)

    date = datetime(int(row['year']), 1, 1) + timedelta(days=int(row['DOB'])-1)
    start_local = tz.localize(datetime(date.year, date.month, date.day))
    end_local = start_local + timedelta(days=1)
    noon_local = tz.localize(datetime(date.year, date.month, date.day, 12))

    return pd.Series([
        int(date.month),
        start_local.astimezone(pytz.utc).isoformat()[:19],
        end_local.astimezone(pytz.utc).isoformat()[:19],
        noon_local.astimezone(pytz.utc).isoformat()[:19]
    ])

def append_tile_coordinates(df, grid_size=256, pixel_size=90, x_col='easting', y_col='northing'):
    """Calculates the global tile IDs and local 0-255 pixel coordinates
    for every row in the dataframe, appending them as new columns.

    Args:
      df: dataframe
      grid_size:  the grid size (Default value = 256)
      pixel_size:  the pixel size (Default value = 90)
      x_col:  the name of the X column (Default value = 'easting')
      y_col:  the name of the Y column (Default value = 'northing')

    Returns: the new dataframe with the columns related to the tiles

    """
    df_mapped = df.copy()
    tile_span_m = grid_size * pixel_size
    
    # Calculate Tile Columns and Rows
    df_mapped['tile_col'] = np.floor(df_mapped[x_col] / tile_span_m).astype(int)
    df_mapped['tile_row'] = np.floor(df_mapped[y_col] / tile_span_m).astype(int)
    df_mapped['tile_id'] = 'tile_' + df_mapped['tile_col'].astype(str) + '_' + df_mapped['tile_row'].astype(str)
    
    # Calculate Bounding boxes for each pixel's respective tile
    df_mapped['tile_x_min'] = df_mapped['tile_col'] * tile_span_m
    df_mapped['tile_y_max'] = (df_mapped['tile_row'] + 1) * tile_span_m
    
    # Calculate Local Pixel Coordinates (0 to 255)
    df_mapped['pixel_x'] = np.floor((df_mapped[x_col] - df_mapped['tile_x_min']) / pixel_size).astype(int).clip(0, grid_size - 1)
    df_mapped['pixel_y'] = np.floor((df_mapped['tile_y_max'] - df_mapped[y_col]) / pixel_size).astype(int).clip(0, grid_size - 1)

    return df_mapped

def generate_fire_h5(df, fire_id, output_folder, grid_size=256, pixel_size=90, 
                     x_col='easting', y_col='northing', lat_col='lat', lon_col='lon', 
                     fill_value=-9999):
    """Generates an HDF5 file for a specific fire using the permanent global tile architecture.
    Structure: /tile_ID / Coords | days / day_XXX / features

    Args:
      df: the CFSD dataframe
      fire_id: the id of the fire
      output_folder: the destination folder of the H5 files
      grid_size:  the grid size (Default value = 256)
      pixel_size:  the pixel size (Default value = 90)
      x_col: the name of the X column (Default value = 'easting')
      y_col: the name of the Y column (Default value = 'northing')
      lat_col: the name of the longitude column (Default value = 'lat')
      lon_col: the name of the latitude column (Default value = 'lon')
      fill_value:  the filling value for missing data in the firemask (Default value = -9999)

    Returns: the path of the fire's H5 folder

    """
    fire_df = df[df["ID"] == fire_id].copy()
    
    if fire_df.empty:
        print(f"No data for Fire ID {fire_id}. Skipping.")
        return None

    # Apply mapping math using the helper function
    fire_df = append_tile_coordinates(fire_df, grid_size, pixel_size, x_col, y_col)
    tile_span_m = grid_size * pixel_size

    # Setup Transformers & Tools
    # transformer = Transformer.from_crs("EPSG:3347", "EPSG:4269", always_xy=True)
    transformer = Transformer.from_crs(f"EPSG:{METER_NAD_CRS}", f"EPSG:{DEG_NAD_CRS}", always_xy=True)
    tf = TimezoneFinder()
    
    file_path = os.path.join(output_folder, f"fire_{fire_id}.h5")
    env_cols = ['firearea', 'cumuarea']
    meta_keys = ["DOB"]
    
    # Build the HDF5 File
    with h5py.File(file_path, "w") as f:
        # File-level metadata
        f.attrs["fire_id"] = fire_id
        f.attrs["year"] = int(fire_df["year"].unique()[0])
        f.attrs["pixel_size"] = pixel_size
        f.attrs["meter_crs"] = f"EPSG:{METER_NAD_CRS}"
        f.attrs["deg_crs"] = f"EPSG:{DEG_NAD_CRS}"

        # Group by Unique Tiles
        for (t_col, t_row), tile_df in fire_df.groupby(['tile_col', 'tile_row']):
            tile_id = f"tile_{t_col}_{t_row}"
            tile_grp = f.create_group(tile_id)
            
            # Tile Metadata
            tile_grp.attrs["tile_col"] = t_col
            tile_grp.attrs["tile_row"] = t_row
            tile_x_min = t_col * tile_span_m
            tile_y_max = (t_row + 1) * tile_span_m
            
            # --- TILE COORDS (Theoretical Geometry) ---
            coords_grp = tile_grp.create_group("coords")
            res_rows, res_cols = np.indices((grid_size, grid_size))
            
            # X goes left to right, Y goes top to bottom
            theoretical_x = tile_x_min + (res_cols * pixel_size)
            theoretical_y = tile_y_max - (res_rows * pixel_size)
            theoretical_lon, theoretical_lat = transformer.transform(theoretical_x, theoretical_y)
            
            coords_grp.create_dataset("theoretical_easting", data=theoretical_x, compression="gzip")
            coords_grp.create_dataset("theoretical_northing", data=theoretical_y, compression="gzip")
            coords_grp.create_dataset("theoretical_lon", data=theoretical_lon, compression="gzip")
            coords_grp.create_dataset("theoretical_lat", data=theoretical_lat, compression="gzip")

            # --- TILE DAYS ---
            days_grp = tile_grp.create_group("days")
            
            for fireday, day_df in tile_df.groupby('fireday'):
                day_grp = days_grp.create_group(f"day_{int(fireday):03d}")
                day_grp.attrs["fireday"] = fireday
                
                # Metadata & UTC Time
                first_row = day_df.iloc[0]
                month, start_utc, end_utc, noon_utc = local_to_utc(first_row, tf, lon_col, lat_col)
                day_grp.attrs["month"] = month if month else "N/A"
                day_grp.attrs["start_utc"] = start_utc if start_utc else "N/A"
                day_grp.attrs["end_utc"] = end_utc if end_utc else "N/A"
                day_grp.attrs["noon_utc"] = noon_utc if noon_utc else "N/A"
                
                for k in meta_keys:
                    val = first_row.get(k)
                    day_grp.attrs[k] = val if pd.notnull(val) else "N/A"

                # Extract local pixel indices for this day
                px_x = day_df['pixel_x'].values
                px_y = day_df['pixel_y'].values

                # --- FEATURES ---
                feat_grp = day_grp.create_group("features")
                binary_cols = ['firearea', 'cumuarea'] 
                
                for col in env_cols:
                    if col in day_df.columns:
                        if col in binary_cols:
                            # Initialize an empty boolean array (Defaults to False)
                            grid = np.zeros((grid_size, grid_size), dtype=bool)
                            
                            # Binarize: Any value > 0 becomes True, everything else is False
                            grid[px_y, px_x] = (day_df[col].values > 0)
                            
                            # Save as boolean type
                            feat_grp.create_dataset(col, data=grid, compression="gzip")
                        else:
                            # Fallback for standard float32 features
                            grid = np.full((grid_size, grid_size), fill_value, dtype=np.float32)
                            grid[px_y, px_x] = day_df[col].values
                            feat_grp.create_dataset(col, data=grid, compression="gzip")
                
                # --- OBSERVED COORDS ---
                obs_grp = day_grp.create_group("observed_coords")
                obs_map = {
                    "easting": x_col, "northing": y_col, 
                    "lat": lat_col, "lon": lon_col
                }
                for out_name, in_name in obs_map.items():
                    if in_name in day_df.columns:
                        grid = np.full((grid_size, grid_size), fill_value, dtype=np.float32)
                        grid[px_y, px_x] = day_df[in_name].values
                        obs_grp.create_dataset(out_name, data=grid, compression="gzip")

    print(f"Successfully generated HDF5 for {fire_id}: {file_path}")
    return file_path