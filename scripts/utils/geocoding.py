from geopy.geocoders import Nominatim
from time import sleep
import pandas as pd

geolocator = Nominatim(user_agent="tableau_geo")

def geocode(city : str, state: str, country: str) -> tuple[float | None, float | None]:
    """Geocode a location using city, state, and country."""

    try:
        # Build query with state if available
        if pd.notna(state) and state:
            query = f"{city}, {state}, {country}"
        else:
            query = f"{city}, {country}"
        
        location = geolocator.geocode(query, timeout=10)
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass

    return None, None

def geocoding(df : pd.DataFrame) -> None:
    """Geocode locations in the dataframe and add LATITUDE and LONGITUDE columns."""

    # Check if STATE column exists
    has_state = "STATE" in df.columns
    
    # Get unique city/state/country combinations
    if has_state:
        unique_locations = df[["CITY", "STATE", "COUNTRY"]].drop_duplicates()
    else:
        unique_locations = df[["CITY", "COUNTRY"]].drop_duplicates()
        unique_locations["STATE"] = None # Ensures countries without states are handled uniformly
    
    geo_cache = {}

    print(f"Geocoding {len(unique_locations)} unique locations...")
    for idx, row in enumerate(unique_locations.itertuples(index=False), 1):
        if has_state:
            city, state, country = row
            # Normalize NaN to None
            state = state if pd.notna(state) else None
        else:
            city, country = row
            state = None
        
        display = f"{city}, {state}, {country}" if state else f"{city}, {country}"
        print(f"  {idx}/{len(unique_locations)}: {display}")
        latitude, longitude = geocode(city, state, country)
        geo_cache[(city, state, country)] = (latitude, longitude)
        sleep(1)  # Required to avoid being blocked from api

    # Map coordinates back to original dataframe
    def get_coordinates(row: pd.Series, coord_idx: int) -> float | None:
        state = row.get("STATE") if has_state else None
        state = state if pd.notna(state) else None
        return geo_cache.get((row["CITY"], state, row["COUNTRY"]), (None, None))[coord_idx]
    
    df["LATITUDE"] = df.apply(lambda row: get_coordinates(row, 0), axis=1)
    df["LONGITUDE"] = df.apply(lambda row: get_coordinates(row, 1), axis=1)