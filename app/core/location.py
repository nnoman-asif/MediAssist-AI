from typing import Optional

import googlemaps
import requests

from app import config
from app.utils.logger import get_logger

log = get_logger(__name__)

_gmaps: Optional[googlemaps.Client] = None


def _get_client() -> googlemaps.Client:
    global _gmaps
    if _gmaps is None:
        if not config.GOOGLE_MAPS_API_KEY:
            raise RuntimeError("GOOGLE_MAPS_API_KEY not set in .env")
        _gmaps = googlemaps.Client(key=config.GOOGLE_MAPS_API_KEY)
    return _gmaps


def geocode(location_name: str) -> dict:
    """Convert a place name to lat/lng.

    Returns dict with keys: lat, lng, formatted_address.
    """
    client = _get_client()
    results = client.geocode(location_name)
    if not results:
        log.warning("Geocode returned no results for '%s'", location_name)
        return {"lat": 0.0, "lng": 0.0, "formatted_address": ""}

    geo = results[0]["geometry"]["location"]
    address = results[0].get("formatted_address", location_name)
    log.info("Geocoded '%s' -> (%.4f, %.4f)", location_name, geo["lat"], geo["lng"])
    return {"lat": geo["lat"], "lng": geo["lng"], "formatted_address": address}


def find_hospitals(
    lat: float,
    lng: float,
    radius: int = 5000,
    max_results: int = 15,
) -> list[dict]:
    """Find nearby hospitals using Google Places API.

    Returns list of dicts with keys: name, address, maps_url.
    """
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "type": "hospital",
        "key": config.GOOGLE_MAPS_API_KEY,
    }

    resp = requests.get(base_url, params=params, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    hospitals = []
    for r in results[:max_results]:
        place_id = r.get("place_id", "")
        hospitals.append(
            {
                "name": r.get("name", "Unknown"),
                "address": r.get("vicinity", ""),
                "maps_url": f"https://www.google.com/maps/place/?q=place_id:{place_id}",
            }
        )

    log.info("Found %d hospitals near (%.4f, %.4f)", len(hospitals), lat, lng)
    return hospitals


def get_hospitals_for_location(location_name: str) -> list[dict]:
    """Convenience: geocode a place name and return nearby hospitals."""
    geo = geocode(location_name)
    if geo["lat"] == 0.0 and geo["lng"] == 0.0:
        return []
    return find_hospitals(geo["lat"], geo["lng"])
