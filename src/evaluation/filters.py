import pandas as pd

def get_baseline_indices(df: pd.DataFrame) -> list[int]:
    """Returns all indices for the baseline evaluation."""
    return df['sample_index'].tolist()

def get_region_indices(df: pd.DataFrame, region: str) -> list[int]:
    """Filters samples by Eastern (>= -100 lon) or Western (< -100 lon) Canada."""
    if region.lower() == "east":
        return df[df['center_lon'] >= -100.0]['sample_index'].tolist()
    elif region.lower() == "west":
        return df[df['center_lon'] < -100.0]['sample_index'].tolist()
    else:
        raise ValueError("Region must be 'East' or 'West'")

def get_size_indices(df: pd.DataFrame, min_ha: float = 0.0, max_ha: float = float('inf')) -> list[int]:
    """
    Filters based on the LARGEST fire present in the tile.
    """
    # Create a mask where the max fire size falls within the requested range
    mask = (df['max_fire_size'] >= min_ha) & (df['max_fire_size'] < max_ha)
    return df[mask]['sample_index'].tolist()

def get_year_indices(df: pd.DataFrame, target_year: int) -> list[int]:
    """Filters samples by a specific year."""
    return df[df['year'] == target_year]['sample_index'].tolist()

def get_month_indices(df: pd.DataFrame, target_month: int) -> list[int]:
    """Filters samples by a specific month (1-12)."""
    return df[df['parsed_date'].dt.month == target_month]['sample_index'].tolist()

def get_season_indices(df: pd.DataFrame, season: str) -> list[int]:
    """Filters samples by fire season (Spring, Summer, Fall)."""
    month = df['parsed_date'].dt.month
    
    if season.lower() == "spring":
        mask = month.isin([3, 4, 5])
    elif season.lower() == "summer":
        mask = month.isin([6, 7, 8])
    elif season.lower() == "fall" or season.lower() == "autumn":
        mask = month.isin([9, 10, 11])
    else:
        raise ValueError("Season must be 'spring', 'summer', or 'fall'")
        
    return df[mask]['sample_index'].tolist()


def get_province_indices(df: pd.DataFrame, province: str) -> list[int]:
    """
    Filters samples by Canadian province using strict lat/lon bounding boxes.
    """
    prov = province.upper()
    
    # Northern Territories
    if prov == 'YT': # Yukon
        mask = (df['center_lat'] >= 60.0) & (df['center_lat'] <= 70.0) & (df['center_lon'] <= -124.0)
    elif prov == 'NWT': # Northwest Territories
        mask = (df['center_lat'] >= 60.0) & (df['center_lat'] <= 78.0) & (df['center_lon'] > -124.0) & (df['center_lon'] <= -102.0)
        
    
    # Western Provinces
    elif prov == 'BC': # British Columbia
        mask = (df['center_lat'] >= 49.0) & (df['center_lat'] < 60.0) & (df['center_lon'] <= -120.0)
    elif prov == 'AB': # Alberta (Border is exactly 120W to 110W)
        mask = (df['center_lat'] >= 49.0) & (df['center_lat'] < 60.0) & (df['center_lon'] > -120.0) & (df['center_lon'] <= -110.0)
    elif prov == 'SK': # Saskatchewan (Border is exactly 110W to 102W)
        mask = (df['center_lat'] >= 49.0) & (df['center_lat'] < 60.0) & (df['center_lon'] > -110.0) & (df['center_lon'] <= -102.0)
    elif prov == 'MB': # Manitoba 
        mask = (df['center_lat'] >= 49.0) & (df['center_lat'] < 60.0) & (df['center_lon'] > -102.0) & (df['center_lon'] <= -90.0)
        
    
    # Eastern Provinces
    elif prov == 'ON': # Ontario (Dips south to ~41.6 in the Great Lakes)
        mask = (df['center_lat'] >= 41.5) & (df['center_lat'] < 57.0) & (df['center_lon'] > -95.0) & (df['center_lon'] <= -74.0)
    elif prov == 'QC': # Quebec (US border sits near 45.0)
        mask = (df['center_lat'] >= 45.0) & (df['center_lat'] < 63.0) & (df['center_lon'] > -79.5) & (df['center_lon'] <= -57.0)
        
    else:
        raise ValueError(f"Province '{province}' boundary not defined. Try: BC, AB, SK, MB, ON, QC, NWT, or YT.")
        
    return df[mask]['sample_index'].tolist()

def get_duration_indices(df: pd.DataFrame, min_days: int = 1, max_days: int = 999) -> list[int]:
    """
    Filters samples based on the fire age
    """
    if 'fireday' in df.columns:
        df['fireday'] = pd.to_numeric(df['fireday'], errors='coerce').fillna(1)
    else:
        print("[!] WARNING: 'fireday' not found in metadata! Defaulting to 1.")
        df['fireday'] = 1

    # Filter based strictly on the sample's fireday
    mask = (df['fireday'] >= min_days) & (df['fireday'] <= max_days)
    return df[mask]['sample_index'].tolist()