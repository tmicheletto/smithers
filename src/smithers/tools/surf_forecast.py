"""Surf forecast tool using Open-Meteo Marine and Weather APIs.

Provides intelligent surf forecasts with ratings for Australian surf spots.
Analyzes wave height, period, wind conditions to determine surf quality.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
import httpx
from langchain_core.tools import tool


def degrees_to_compass(degrees: float) -> str:
    """Convert wind direction in degrees to compass direction.

    Args:
        degrees: Wind direction in degrees (0-360)

    Returns:
        Compass direction string (N, NNE, NE, etc.)
    """
    # Normalize to 0-360 range
    degrees = degrees % 360

    # Define compass directions
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]

    # Each direction covers 22.5 degrees (360 / 16)
    # Add 11.25 to center the ranges, then divide by 22.5
    index = int((degrees + 11.25) / 22.5) % 16

    return directions[index]


def geocode_australian_location(location_name: str) -> tuple[float, float, str] | None:
    """Geocode an Australian surf spot name to coordinates.

    Uses Open-Meteo Geocoding API to find locations in Australia.

    Args:
        location_name: Name of the surf spot (e.g., "Bells Beach", "Torquay")

    Returns:
        Tuple of (latitude, longitude, full_location_name) or None if not found
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": location_name,
        "count": 5,
        "language": "en",
        "format": "json",
    }

    try:
        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return None

        # Filter for Australian locations
        australian_results = [
            r for r in results
            if r.get("country_code") == "AU" or r.get("country") == "Australia"
        ]

        if not australian_results:
            # If no Australian results, try first result anyway
            australian_results = results

        # Return first match
        result = australian_results[0]
        lat = result.get("latitude")
        lon = result.get("longitude")
        name = result.get("name", location_name)
        admin = result.get("admin1", "")  # State/region
        country = result.get("country", "")

        full_name = f"{name}, {admin}, {country}" if admin and country else name

        return (lat, lon, full_name)

    except Exception:
        return None


def get_day_offset(when: str) -> tuple[int, str]:
    """Get day offset and display name from time reference.

    Args:
        when: Time reference like "today", "tomorrow", "Monday", "Sunday"

    Returns:
        Tuple of (day_offset, display_name)
        - day_offset: 0=today, 1=tomorrow, etc.
        - display_name: Formatted date string

    Examples:
        "today" → (0, "Today - Thursday, December 5")
        "tomorrow" → (1, "Tomorrow - Friday, December 6")
        "Sunday" → (2, "Sunday, December 7")
    """
    when_lower = when.lower().strip()
    today = datetime.now()

    # Handle "today"
    if when_lower == "today":
        return (0, f"Today - {today.strftime('%A, %B %d')}")

    # Handle "tomorrow"
    if when_lower == "tomorrow":
        tomorrow = today + timedelta(days=1)
        return (1, f"Tomorrow - {tomorrow.strftime('%A, %B %d')}")

    # Handle day names (Monday, Tuesday, etc.)
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if when_lower in days:
        target_day = days.index(when_lower)
        current_day = today.weekday()

        # Calculate offset (0-6)
        offset = (target_day - current_day) % 7
        if offset == 0:
            offset = 7  # Next week if same day

        if offset > 5:
            return (offset, f"{when.capitalize()} is beyond the 5-day forecast")

        target_date = today + timedelta(days=offset)
        return (offset, f"{when.capitalize()}, {target_date.strftime('%B %d')}")

    # Default to today if can't parse
    return (0, f"Today - {today.strftime('%A, %B %d')}")


def circular_mean_direction(directions: list[float]) -> float:
    """Calculate circular mean of wind directions.

    Args:
        directions: List of direction values in degrees

    Returns:
        Mean direction in degrees (0-360)
    """
    if not directions:
        return 0.0

    sin_sum = sum(math.sin(math.radians(d)) for d in directions)
    cos_sum = sum(math.cos(math.radians(d)) for d in directions)
    mean_rad = math.atan2(sin_sum, cos_sum)
    return math.degrees(mean_rad) % 360


def find_tide_extremes(
    times: list[str],
    sea_levels: list[float],
    target_date: str,
) -> list[dict]:
    """Find high and low tides from hourly sea level data.

    Identifies local maxima (high tides) and minima (low tides) in the sea level curve.

    Args:
        times: List of ISO timestamp strings
        sea_levels: List of sea level heights in meters
        target_date: Date string "YYYY-MM-DD" to filter for

    Returns:
        List of dicts with keys: 'time', 'height', 'type' ('high' or 'low')
        Sorted by time.

    Example:
        [
            {'time': '2025-12-05T06:23:00', 'height': 1.8, 'type': 'high'},
            {'time': '2025-12-05T12:45:00', 'height': 0.3, 'type': 'low'},
        ]
    """
    if not times or not sea_levels or len(times) != len(sea_levels):
        return []

    # Filter for target date
    date_indices = []
    for i, time_str in enumerate(times):
        if time_str.startswith(target_date):
            date_indices.append(i)

    if len(date_indices) < 3:
        return []  # Need at least 3 points to find extremes

    extremes = []

    # Find local maxima and minima
    for i in range(1, len(date_indices) - 1):
        idx = date_indices[i]
        prev_idx = date_indices[i - 1]
        next_idx = date_indices[i + 1]

        current = sea_levels[idx]
        prev = sea_levels[prev_idx]
        next = sea_levels[next_idx]

        # Check for local maximum (high tide)
        if current > prev and current > next:
            extremes.append({
                'time': times[idx],
                'height': current,
                'type': 'high'
            })

        # Check for local minimum (low tide)
        elif current < prev and current < next:
            extremes.append({
                'time': times[idx],
                'height': current,
                'type': 'low'
            })

    return sorted(extremes, key=lambda x: x['time'])


def format_tide_time(iso_time: str) -> str:
    """Format ISO timestamp to readable time.

    Args:
        iso_time: ISO format like "2025-12-05T06:23:00"

    Returns:
        Formatted time like "6:23 AM"
    """
    try:
        dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
        return dt.strftime('%-I:%M %p')
    except Exception:
        return iso_time.split('T')[1][:5] if 'T' in iso_time else iso_time


def get_tide_state_for_session(
    tide_extremes: list[dict],
    session_start_hour: int,
    session_end_hour: int,
) -> str | None:
    """Determine tide state during a session.

    Args:
        tide_extremes: List of tide extreme dicts from find_tide_extremes()
        session_start_hour: Session start hour (e.g., 6 for morning)
        session_end_hour: Session end hour (e.g., 10 for morning)

    Returns:
        - "High tide at 6:23 AM (1.8m)" if high tide occurs in session
        - "Low tide at 12:45 PM (0.3m)" if low tide occurs in session
        - "Tide rising" if tide is rising during session
        - "Tide falling" if tide is falling during session
        - None if no tide data
    """
    if not tide_extremes:
        return None

    # Check if any extreme falls within session hours
    for extreme in tide_extremes:
        try:
            time_str = extreme['time']
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            hour = dt.hour

            if session_start_hour <= hour < session_end_hour:
                formatted_time = format_tide_time(time_str)
                tide_type = extreme['type'].capitalize()
                height = extreme['height']
                return f"{tide_type} tide at {formatted_time} ({height:.1f}m)"
        except Exception:
            continue

    # Determine if rising or falling based on surrounding extremes
    session_mid = session_start_hour + (session_end_hour - session_start_hour) / 2

    before = []
    after = []

    for extreme in tide_extremes:
        try:
            time_str = extreme['time']
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            hour = dt.hour

            if hour < session_mid:
                before.append(extreme)
            elif hour >= session_mid:
                after.append(extreme)
        except Exception:
            continue

    if before and after:
        last_before = before[-1]
        first_after = after[0]

        if last_before['type'] == 'low' and first_after['type'] == 'high':
            return "Tide rising"
        elif last_before['type'] == 'high' and first_after['type'] == 'low':
            return "Tide falling"

    return None


def format_tide_summary(tide_extremes: list[dict]) -> str:
    """Format daily tide summary.

    Args:
        tide_extremes: List of tide extreme dicts

    Returns:
        String like "Tides: High 6:23 AM (1.8m), Low 12:45 PM (0.3m), High 6:52 PM (1.7m)"
    """
    if not tide_extremes:
        return ""

    parts = []
    for extreme in tide_extremes:
        time_str = format_tide_time(extreme['time'])
        tide_type = extreme['type'].capitalize()
        height = extreme['height']
        parts.append(f"{tide_type} {time_str} ({height:.1f}m)")

    return "Tides: " + ", ".join(parts)


def aggregate_session_data(
    times: list[str],
    wave_heights: list[float],
    wave_periods: list[float],
    wave_directions: list[float],
    wind_speeds: list[float],
    wind_directions: list[float],
    target_date: str,  # "2025-12-06"
    session: str,  # "morning", "midday", "afternoon"
    tide_state: str | None = None,  # NEW: Tide state for this session
) -> dict | None:
    """Aggregate hourly data into a session.

    Session time ranges:
    - morning: 06:00-10:00 (hours 6,7,8,9,10)
    - midday: 10:00-14:00 (hours 10,11,12,13,14)
    - afternoon: 14:00-18:00 (hours 14,15,16,17,18)

    Returns:
        Dictionary with aggregated data or None if no data
    """
    # Define hour ranges for each session
    session_hours = {
        "morning": range(6, 11),  # 6-10 AM
        "midday": range(10, 15),  # 10 AM - 2 PM
        "afternoon": range(14, 19),  # 2-6 PM
    }

    if session not in session_hours:
        return None

    hours = session_hours[session]

    # Filter data for this session
    session_wave_heights = []
    session_wave_periods = []
    session_wind_speeds = []
    session_wind_dirs = []

    for i, time_str in enumerate(times):
        if not time_str.startswith(target_date):
            continue

        hour = int(time_str.split('T')[1].split(':')[0])
        if hour in hours:
            if i < len(wave_heights) and wave_heights[i] is not None:
                session_wave_heights.append(wave_heights[i])
            if i < len(wave_periods) and wave_periods[i] is not None:
                session_wave_periods.append(wave_periods[i])
            if i < len(wind_speeds) and wind_speeds[i] is not None:
                session_wind_speeds.append(wind_speeds[i])
            if i < len(wind_directions) and wind_directions[i] is not None:
                session_wind_dirs.append(wind_directions[i])

    # Return None if no data found
    if not session_wave_heights:
        return None

    # Calculate averages
    avg_wave_height = sum(session_wave_heights) / len(session_wave_heights)
    avg_wave_period = sum(session_wave_periods) / len(session_wave_periods) if session_wave_periods else 0
    avg_wind_speed = sum(session_wind_speeds) / len(session_wind_speeds) if session_wind_speeds else 0
    avg_wind_direction = circular_mean_direction(session_wind_dirs) if session_wind_dirs else 0

    # Calculate rating
    rating, description = calculate_surf_rating(
        avg_wave_height, avg_wave_period, avg_wind_speed, avg_wind_direction
    )

    return {
        'wave_height': avg_wave_height,
        'wave_period': avg_wave_period,
        'wind_speed': avg_wind_speed,
        'wind_direction': avg_wind_direction,
        'rating': rating,
        'description': description,
        'tide_state': tide_state,
    }


def calculate_surf_rating(
    wave_height: float,  # meters
    wave_period: float,  # seconds
    wind_speed: float,   # km/h
    wind_direction: float,  # degrees
    beach_orientation: float = 180,  # degrees, south-facing default
) -> tuple[int, str]:
    """Calculate surf quality rating (1-10) and description.

    Args:
        wave_height: Wave height in meters
        wave_period: Wave period in seconds
        wind_speed: Wind speed in km/h
        wind_direction: Wind direction in degrees
        beach_orientation: Beach facing direction in degrees (default 180=south)

    Returns:
        Tuple of (rating 1-10, description)
    """
    # Start with base score
    score = 5.0

    # Wave period contribution (0-3 points)
    if wave_period < 6:
        score -= 1  # Choppy
    elif wave_period >= 12:
        score += 3  # Excellent long period
    elif wave_period >= 8:
        score += 2  # Good period
    # 6-8s = neutral (0 points)

    # Wave height contribution (0-3 points)
    if wave_height < 0.5:
        score -= 2  # Too small
    elif wave_height >= 2.5:
        score += 1  # Big waves (advanced)
    elif wave_height >= 1.5:
        score += 3  # Ideal size
    elif wave_height >= 0.5:
        score += 2  # Fun size

    # Wind contribution (-2 to +2 points)
    # Calculate if wind is offshore (blowing from land to sea)
    # Offshore is when wind direction is opposite to beach orientation (±90 degrees)
    wind_angle_diff = abs(((wind_direction - beach_orientation + 180) % 360) - 180)
    is_offshore = wind_angle_diff > 90

    if wind_speed < 5:
        score += 2  # Glassy conditions
    elif wind_speed < 15:
        if is_offshore:
            score += 1  # Light offshore
        else:
            score -= 1  # Light onshore
    else:
        if is_offshore:
            score += 0  # Strong offshore (still rideable)
        else:
            score -= 2  # Blown out

    # Clamp to 1-10 range
    rating = max(1, min(10, round(score)))

    # Generate description based on rating
    if rating <= 3:
        descriptions = [
            "Poor conditions - not recommended",
            "Flat or blown out",
            "Save your energy for another day"
        ]
        quality = "Poor"
    elif rating <= 5:
        descriptions = [
            "Fair conditions - marginal",
            "Small but potentially clean",
            "Beginner-friendly if you're keen"
        ]
        quality = "Fair"
    elif rating <= 7:
        descriptions = [
            "Good conditions - worth a surf",
            "Fun waves for most skill levels",
            "Decent session likely"
        ]
        quality = "Good"
    elif rating <= 9:
        descriptions = [
            "Excellent conditions - firing!",
            "Clean waves with good shape",
            "Get out there!"
        ]
        quality = "Excellent"
    else:  # rating == 10
        descriptions = [
            "Epic conditions - all-time!",
            "Perfect waves",
            "Drop everything and surf!"
        ]
        quality = "Epic"

    # Pick description based on conditions
    if wind_speed >= 20:
        description = "Strong winds affecting conditions"
    elif wave_height < 0.3:
        description = "Waves too small"
    elif wave_period >= 12 and wave_height >= 1.5:
        description = descriptions[0]
    else:
        description = descriptions[0]

    return (rating, f"{quality} - {description}")


@tool
def get_surf_forecast(location_name: str, when: str = "today") -> str:
    """Get intelligent surf forecast for an Australian location.

    Retrieves wave height, wave period, wave direction, and wind data,
    then analyzes conditions to provide a surf quality rating (1-10).
    Forecasts are aggregated into morning, midday, and afternoon sessions.

    Args:
        location_name: Name of the surf spot (e.g., "Bells Beach", "Torquay", "Barwon Heads")
        when: Time reference - "today", "tomorrow", "Monday", "Sunday", etc. (default: "today")

    Returns:
        Formatted surf forecast with session ratings, wave/wind conditions, and recommendations.
    """
    # Geocode the location
    geocode_result = geocode_australian_location(location_name)
    if not geocode_result:
        return f"Could not find location: {location_name}. Please check the spelling or try a nearby town."

    latitude, longitude, full_location_name = geocode_result

    # Parse time intent
    day_offset, date_display = get_day_offset(when)

    # Check if day is beyond 5-day window
    if day_offset > 5:
        return f"{date_display}. Please ask for a day within the next 5 days."

    # Calculate target date string
    target_date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")

    # Fetch marine data (waves) - 5 days
    marine_url = "https://marine-api.open-meteo.com/v1/marine"
    marine_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "wave_height,wave_period,wave_direction,sea_level_height_msl",
        "timezone": "auto",
        "forecast_days": 5,
    }

    # Fetch weather data (wind) - 5 days
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "timezone": "auto",
        "forecast_days": 5,
    }

    try:
        # Fetch both APIs
        marine_response = httpx.get(marine_url, params=marine_params, timeout=10.0)
        marine_response.raise_for_status()
        marine_data = marine_response.json()

        weather_response = httpx.get(weather_url, params=weather_params, timeout=10.0)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        # Parse marine data
        marine_hourly = marine_data.get("hourly", {})
        times = marine_hourly.get("time", [])
        wave_heights = marine_hourly.get("wave_height", [])
        wave_periods = marine_hourly.get("wave_period", [])
        wave_directions = marine_hourly.get("wave_direction", [])
        sea_levels = marine_hourly.get("sea_level_height_msl", [])

        # Parse weather data
        weather_hourly = weather_data.get("hourly", {})
        wind_speeds = weather_hourly.get("wind_speed_10m", [])
        wind_directions = weather_hourly.get("wind_direction_10m", [])

        if not times or not wave_heights:
            return f"No surf forecast data available for {full_location_name}"

        # Calculate tide extremes for target date
        tide_extremes = find_tide_extremes(times, sea_levels, target_date)

        # Calculate tide states for each session
        session_hours = {
            "morning": (6, 10),
            "midday": (10, 14),
            "afternoon": (14, 18),
        }

        tide_states = {}
        if tide_extremes:
            for session_name, (start_h, end_h) in session_hours.items():
                tide_states[session_name] = get_tide_state_for_session(
                    tide_extremes, start_h, end_h
                )

        # Aggregate data into sessions
        morning = aggregate_session_data(
            times, wave_heights, wave_periods, wave_directions,
            wind_speeds, wind_directions, target_date, "morning",
            tide_state=tide_states.get("morning")
        )
        midday = aggregate_session_data(
            times, wave_heights, wave_periods, wave_directions,
            wind_speeds, wind_directions, target_date, "midday",
            tide_state=tide_states.get("midday")
        )
        afternoon = aggregate_session_data(
            times, wave_heights, wave_periods, wave_directions,
            wind_speeds, wind_directions, target_date, "afternoon",
            tide_state=tide_states.get("afternoon")
        )

        # Check if any data exists for target date
        if not morning and not midday and not afternoon:
            return f"No surf forecast data available for {date_display} at {full_location_name}"

        # Format output
        result = [f"🏄 Surf Forecast for {full_location_name}\n"]
        result.append(f"📅 {date_display}\n")

        # Add tide summary if available
        if tide_extremes:
            tide_summary = format_tide_summary(tide_extremes)
            if tide_summary:
                result.append(f"{tide_summary}\n")

        # Helper function to format session
        def format_session(session_name: str, session_data: dict | None, time_range: str) -> list[str]:
            lines = []
            if not session_data:
                lines.append(f"{session_name} ({time_range}):")
                lines.append("No data available\n")
                return lines

            wave_ht = session_data['wave_height']
            wave_per = session_data['wave_period']
            wind_spd = session_data['wind_speed']
            wind_dir = session_data['wind_direction']
            rating = session_data['rating']
            description = session_data['description']
            tide_state = session_data.get('tide_state')

            wave_ft = wave_ht * 3.28084
            wind_compass = degrees_to_compass(wind_dir)

            lines.append(f"{session_name} ({time_range}):")
            lines.append(f"Rating: {rating}/10 - {description}")
            lines.append(f"Wave: {wave_ht:.1f}m ({wave_ft:.1f}ft) @ {wave_per:.0f}s")
            lines.append(f"Wind: {wind_spd:.0f} km/h {wind_compass}")

            # Add tide state if available
            if tide_state:
                lines.append(f"Tide: {tide_state}")

            lines.append("")  # blank line

            return lines

        # Add session forecasts
        result.extend(format_session("Morning", morning, "6 AM - 10 AM"))
        result.extend(format_session("Midday", midday, "10 AM - 2 PM"))
        result.extend(format_session("Afternoon", afternoon, "2 PM - 6 PM"))

        # Find best session and add recommendation
        sessions = []
        if morning:
            sessions.append(("Morning", morning['rating']))
        if midday:
            sessions.append(("Midday", midday['rating']))
        if afternoon:
            sessions.append(("Afternoon", afternoon['rating']))

        if sessions:
            best_session, best_rating = max(sessions, key=lambda x: x[1])
            result.append(f"🌊 Best session: {best_session} ({best_rating}/10)")

            if best_rating >= 8:
                result.append("Recommendation: Excellent conditions! Get out there!")
            elif best_rating >= 6:
                result.append("Recommendation: Good conditions, worth a surf!")
            elif best_rating >= 4:
                result.append("Recommendation: Marginal conditions, but could be fun.")
            else:
                result.append("Recommendation: Poor conditions. Maybe check another spot or wait.")

        return "\n".join(result)

    except httpx.HTTPError as e:
        return f"Error fetching surf forecast: {str(e)}"
    except Exception as e:
        return f"Unexpected error getting surf forecast: {str(e)}"


def get_surf_forecast_tool():
    """Return the surf forecast tool for LangChain agent."""
    return get_surf_forecast
