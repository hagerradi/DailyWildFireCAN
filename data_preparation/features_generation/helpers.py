from pyproj import Transformer, CRS
from data_preparation.data_configs.data_settings import DEG_NAD_CRS, METER_NAD_CRS

def lonlat_to_canada_lambert(df, lon_col='lon', lat_col='lat'):
    """Convert lon/lat in EPSG:4269 (NAD83) to EPSG:3347 (NAD83 / Canada Lambert) in meters.
    Returns df with 'easting' and 'northing' columns and the target CRS object.

    Args:
      df: the CFSD dataframe
      lon_col:  the name of the longitude column (Default value = 'lon')
      lat_col:  the name of the latitude column (Default value = 'lat')

    Returns: the df with 'easting' and 'northing' columns and the target CRS object

    """
    # Safety: require lon/lat columns
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns '{lon_col}' and '{lat_col}'")

    source_crs = CRS.from_epsg(int(DEG_NAD_CRS))   # NAD83 geographic (degrees)
    target_crs = CRS.from_epsg(int(METER_NAD_CRS))   # NAD83 / Canada Lambert (meters)

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    lons = df[lon_col].to_numpy(dtype=float)
    lats = df[lat_col].to_numpy(dtype=float)

    # transform (vectorized)
    eastings, northings = transformer.transform(lons, lats)

    df['easting'] = eastings
    df['northing'] = northings
    df.attrs['target_crs'] = target_crs.to_string()

    return df, target_crs