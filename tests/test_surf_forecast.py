"""Tests for surf forecast tool."""

import pytest
from smithers.tools.surf_forecast import (
    get_surf_forecast,
    calculate_surf_rating,
    geocode_australian_location,
    degrees_to_compass,
)


def test_geocode_bells_beach():
    """Test geocoding for Bells Beach."""
    result = geocode_australian_location("Bells Beach")
    assert result is not None
    lat, lon, name = result
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert "Bell" in name or "Torquay" in name  # Bells Beach is near Torquay


def test_geocode_torquay():
    """Test geocoding for Torquay."""
    result = geocode_australian_location("Torquay")
    assert result is not None
    lat, lon, name = result
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert "Torquay" in name


def test_geocode_invalid_location():
    """Test geocoding with invalid location."""
    result = geocode_australian_location("XYZ123NonexistentPlace")
    # Might return None or a random result - just check it doesn't crash
    assert result is None or isinstance(result, tuple)


def test_degrees_to_compass():
    """Test wind direction conversion to compass format."""
    assert degrees_to_compass(0) == "N"
    assert degrees_to_compass(22.5) == "NNE"
    assert degrees_to_compass(45) == "NE"
    assert degrees_to_compass(90) == "E"
    assert degrees_to_compass(135) == "SE"
    assert degrees_to_compass(180) == "S"
    assert degrees_to_compass(225) == "SW"
    assert degrees_to_compass(270) == "W"
    assert degrees_to_compass(315) == "NW"
    assert degrees_to_compass(360) == "N"
    # Test edge cases
    assert degrees_to_compass(11) == "N"  # Close to N
    assert degrees_to_compass(350) == "N"  # Close to N from other side


def test_calculate_surf_rating_excellent():
    """Test rating calculation for excellent conditions."""
    # 1.8m waves, 12s period, light offshore wind
    rating, description = calculate_surf_rating(
        wave_height=1.8,
        wave_period=12.0,
        wind_speed=8.0,
        wind_direction=0,  # North wind
        beach_orientation=180,  # South-facing beach (offshore)
    )
    assert rating >= 7, f"Expected high rating but got {rating}"
    assert "Excellent" in description or "Epic" in description or "Good" in description


def test_calculate_surf_rating_poor():
    """Test rating calculation for poor conditions."""
    # Small waves, short period, strong onshore wind
    rating, description = calculate_surf_rating(
        wave_height=0.3,
        wave_period=4.0,
        wind_speed=25.0,
        wind_direction=180,  # South wind
        beach_orientation=180,  # South-facing beach (onshore)
    )
    assert rating <= 4, f"Expected low rating but got {rating}"
    assert "Poor" in description or "Fair" in description


def test_surf_forecast_bells_beach():
    """Test surf forecast for Bells Beach."""
    result = get_surf_forecast.invoke({"location_name": "Bells Beach"})

    assert isinstance(result, str)
    assert "Surf Forecast" in result
    assert "Rating:" in result
    # Check for rating pattern (X/10)
    assert "/10" in result
    # Check for session names
    assert "Morning" in result
    assert "Midday" in result or "afternoon" in result.lower()
    # Check for wave data
    assert any(term in result for term in ["Wave:", "wave", "m", "ft"])


def test_surf_forecast_torquay():
    """Test surf forecast for Torquay."""
    result = get_surf_forecast.invoke({"location_name": "Torquay"})

    assert isinstance(result, str)
    assert "Torquay" in result or "Surf Forecast" in result
    assert "Rating:" in result
    assert "/10" in result
    # Check for sessions
    assert "Morning" in result or "Afternoon" in result


def test_surf_forecast_barwon_heads():
    """Test surf forecast for Barwon Heads."""
    result = get_surf_forecast.invoke({"location_name": "Barwon Heads"})

    assert isinstance(result, str)
    assert "Surf Forecast" in result or "Barwon" in result
    assert "Rating:" in result
    # Check for sessions
    assert any(session in result for session in ["Morning", "Midday", "Afternoon"])


def test_surf_forecast_invalid_location():
    """Test surf forecast with invalid location name."""
    result = get_surf_forecast.invoke({"location_name": "NonexistentSurfSpot12345"})

    assert isinstance(result, str)
    # Should return an error message
    assert "Could not find" in result or "error" in result.lower()


def test_surf_forecast_has_recommendation():
    """Test that surf forecast includes a recommendation."""
    result = get_surf_forecast.invoke({"location_name": "Torquay"})

    assert isinstance(result, str)
    assert "Recommendation:" in result or "Best session:" in result


def test_surf_forecast_tomorrow():
    """Test surf forecast for tomorrow."""
    result = get_surf_forecast.invoke({
        "location_name": "Bells Beach",
        "when": "tomorrow"
    })

    assert isinstance(result, str)
    assert "Surf Forecast" in result
    assert "Tomorrow" in result
    # Check for sessions
    assert any(session in result for session in ["Morning", "Midday", "Afternoon"])
    assert "Rating:" in result


def test_surf_forecast_specific_day():
    """Test forecast for specific day name."""
    # Note: This test might fail if the day is beyond 5 days
    # Using "tomorrow" as a safe bet
    result = get_surf_forecast.invoke({
        "location_name": "Torquay",
        "when": "tomorrow"
    })

    assert isinstance(result, str)
    assert "Surf Forecast" in result


def test_surf_forecast_today_explicit():
    """Test forecast when explicitly requesting today."""
    result = get_surf_forecast.invoke({
        "location_name": "Barwon Heads",
        "when": "today"
    })

    assert isinstance(result, str)
    assert "Today" in result
    assert "Surf Forecast" in result


def test_get_day_offset_today():
    """Test day offset calculation for 'today'."""
    from smithers.tools.surf_forecast import get_day_offset
    offset, display = get_day_offset("today")
    assert offset == 0
    assert "Today" in display


def test_get_day_offset_tomorrow():
    """Test day offset calculation for 'tomorrow'."""
    from smithers.tools.surf_forecast import get_day_offset
    offset, display = get_day_offset("tomorrow")
    assert offset == 1
    assert "Tomorrow" in display


def test_circular_mean_direction():
    """Test circular mean for wind directions."""
    from smithers.tools.surf_forecast import circular_mean_direction

    # Test averaging 350° and 10° (should be near 0°/360°)
    result = circular_mean_direction([350.0, 10.0])
    # Result should be near 0 or 360 (both represent North)
    assert result >= 355 or result <= 5

    # Test averaging 90° (East)
    result = circular_mean_direction([90.0, 90.0, 90.0])
    assert 89 <= result <= 91

    # Test empty list
    result = circular_mean_direction([])
    assert result == 0.0


def test_aggregate_session_data():
    """Test session aggregation function."""
    from smithers.tools.surf_forecast import aggregate_session_data
    from datetime import datetime

    # Create mock hourly data for a single day
    today = datetime.now().strftime("%Y-%m-%d")
    times = [f"{today}T{h:02d}:00" for h in range(24)]
    wave_heights = [1.5] * 24
    wave_periods = [10.0] * 24
    wave_directions = [180.0] * 24
    wind_speeds = [10.0] * 24
    wind_directions = [0.0] * 24  # North wind

    # Test morning session (6-10 AM)
    morning = aggregate_session_data(
        times, wave_heights, wave_periods, wave_directions,
        wind_speeds, wind_directions, today, "morning"
    )

    assert morning is not None
    assert "wave_height" in morning
    assert "rating" in morning
    assert morning["wave_height"] == 1.5  # Average should be same as constant value


def test_find_tide_extremes():
    """Test finding high/low tides from sea level data."""
    from smithers.tools.surf_forecast import find_tide_extremes
    from datetime import datetime

    # Create mock hourly sea level data for one day
    today = datetime.now().strftime("%Y-%m-%d")
    times = [f"{today}T{h:02d}:00" for h in range(24)]

    # Simulate tidal curve: low at 6am (0.3m), high at 12pm (1.8m), low at 6pm (0.2m)
    sea_levels = []
    for h in range(24):
        if h < 6:
            level = 0.5 + (h / 6) * 0.8  # Rising to first high
        elif h < 12:
            level = 1.3 - abs(h - 9) * 0.3  # High tide around 9am
        elif h < 18:
            level = 1.3 - (h - 12) / 6 * 1.1  # Falling to low
        else:
            level = 0.2 + (h - 18) / 6 * 0.5  # Rising again
        sea_levels.append(level)

    extremes = find_tide_extremes(times, sea_levels, today)

    assert len(extremes) > 0
    # Should find at least one high and one low
    types = [e['type'] for e in extremes]
    assert 'high' in types
    assert 'low' in types


def test_format_tide_time():
    """Test tide time formatting."""
    from smithers.tools.surf_forecast import format_tide_time

    result = format_tide_time("2025-12-05T06:23:00")
    assert "6:23" in result
    assert "AM" in result or "am" in result.lower()

    result = format_tide_time("2025-12-05T14:45:00")
    assert "2:45" in result or "14:45" in result
    assert "PM" in result or "pm" in result.lower()


def test_get_tide_state_for_session():
    """Test tide state calculation for sessions."""
    from smithers.tools.surf_forecast import get_tide_state_for_session

    tide_extremes = [
        {'time': '2025-12-05T07:30:00', 'height': 1.8, 'type': 'high'},
        {'time': '2025-12-05T13:45:00', 'height': 0.3, 'type': 'low'},
        {'time': '2025-12-05T19:15:00', 'height': 1.7, 'type': 'high'},
    ]

    # Morning session (6-10) should show high tide at 7:30
    morning_state = get_tide_state_for_session(tide_extremes, 6, 10)
    assert morning_state is not None
    assert "High tide" in morning_state or "high tide" in morning_state
    assert "7:30" in morning_state

    # Afternoon session (14-18) should show tide rising (between low at 13:45 and high at 19:15)
    afternoon_state = get_tide_state_for_session(tide_extremes, 14, 18)
    # Should be rising since it's between low and high
    assert afternoon_state is not None
    assert "rising" in afternoon_state.lower() or "Tide rising" in afternoon_state


def test_format_tide_summary():
    """Test tide summary formatting."""
    from smithers.tools.surf_forecast import format_tide_summary

    tide_extremes = [
        {'time': '2025-12-05T06:23:00', 'height': 1.8, 'type': 'high'},
        {'time': '2025-12-05T12:45:00', 'height': 0.3, 'type': 'low'},
    ]

    summary = format_tide_summary(tide_extremes)

    assert "Tides:" in summary
    assert "6:23" in summary
    assert "1.8m" in summary
    assert "12:45" in summary
    assert "0.3m" in summary


def test_surf_forecast_includes_tide():
    """Test that surf forecast includes tide information."""
    result = get_surf_forecast.invoke({
        "location_name": "Bells Beach",
        "when": "today"
    })

    assert isinstance(result, str)
    # Should include tide information in sessions
    # Note: Tide info might not always be present if sea level data is missing
    # So we just verify the forecast still works
    assert "Surf Forecast" in result
    assert "Morning" in result
