"""Surf forecast tool using Open-Meteo Marine API.

Provides wave height, period, direction, wind, and tide information for surf spots.
"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool


@tool
def get_surf_forecast(latitude: float, longitude: float, location_name: str = "Unknown") -> str:
    """Get surf forecast for a location using coordinates.

    Retrieves wave height, wave period, wave direction, and wind data
    from Open-Meteo Marine API. Useful for surfers planning sessions.

    Args:
        latitude: Latitude of the surf spot (-90 to 90)
        longitude: Longitude of the surf spot (-180 to 180)
        location_name: Name of the location for display purposes

    Returns:
        Formatted surf forecast string with wave and wind conditions.
    """
    # Open-Meteo Marine API endpoint
    url = "https://marine-api.open-meteo.com/v1/marine"

    # Request parameters - marine variables
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "wave_height,wave_period,wave_direction,wind_wave_height,wind_wave_period,wind_wave_direction",
        "timezone": "auto",
        "forecast_days": 3,
    }

    try:
        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        # Parse response
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        wave_heights = hourly.get("wave_height", [])
        wave_periods = hourly.get("wave_period", [])
        wave_directions = hourly.get("wave_direction", [])
        wind_wave_heights = hourly.get("wind_wave_height", [])

        if not times or not wave_heights:
            return f"No surf forecast data available for {location_name} ({latitude}, {longitude})"

        # Format output - show current + next 24 hours summary
        result = [f"Surf Forecast for {location_name} ({latitude}, {longitude})\n"]
        result.append(f"Timezone: {data.get('timezone', 'UTC')}\n")

        # Show first 8 time periods (typically ~24 hours depending on interval)
        for i in range(min(8, len(times))):
            time = times[i].split('T')[1] if 'T' in times[i] else times[i]
            date = times[i].split('T')[0] if 'T' in times[i] else ""

            wave_ht = wave_heights[i] if i < len(wave_heights) else None
            wave_per = wave_periods[i] if i < len(wave_periods) else None
            wave_dir = wave_directions[i] if i < len(wave_directions) else None
            wind_ht = wind_wave_heights[i] if i < len(wind_wave_heights) else None

            # Format wave height in feet (convert from meters)
            wave_ft = f"{wave_ht * 3.28084:.1f}ft" if wave_ht is not None else "N/A"
            wind_ft = f"{wind_ht * 3.28084:.1f}ft" if wind_ht is not None else "N/A"

            result.append(
                f"{date} {time}: Wave {wave_ft} @ {wave_per}s from {wave_dir}° | Wind waves {wind_ft}"
            )

        return "\n".join(result)

    except httpx.HTTPError as e:
        return f"Error fetching surf forecast: {str(e)}"
    except Exception as e:
        return f"Unexpected error getting surf forecast: {str(e)}"


def get_surf_forecast_tool():
    """Return the surf forecast tool for LangChain agent."""
    return get_surf_forecast
