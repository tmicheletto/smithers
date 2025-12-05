"""Tests for surf forecast tool."""

import pytest
from smithers.tools.surf_forecast import get_surf_forecast


def test_surf_forecast_basic():
    """Test basic surf forecast retrieval."""
    # Pipeline, Hawaii coordinates
    result = get_surf_forecast.invoke({
        "latitude": 21.6644,
        "longitude": -158.0533,
        "location_name": "Pipeline"
    })

    assert isinstance(result, str)
    assert "Pipeline" in result
    assert "Surf Forecast" in result
    # Check for wave data indicators
    assert "Wave" in result or "wave" in result


def test_surf_forecast_mavericks():
    """Test surf forecast for Mavericks."""
    # Mavericks, California coordinates
    result = get_surf_forecast.invoke({
        "latitude": 37.4936,
        "longitude": -122.4969,
        "location_name": "Mavericks"
    })

    assert isinstance(result, str)
    assert "Mavericks" in result
    assert "Surf Forecast" in result


def test_surf_forecast_invalid_coordinates():
    """Test surf forecast with out-of-range coordinates."""
    # Invalid latitude (>90)
    result = get_surf_forecast.invoke({
        "latitude": 100.0,
        "longitude": -158.0,
        "location_name": "Invalid"
    })

    # Should return error message, not crash
    assert isinstance(result, str)
    assert "error" in result.lower() or "Error" in result


def test_surf_forecast_response_format():
    """Test that surf forecast returns expected data format."""
    result = get_surf_forecast.invoke({
        "latitude": 33.6584,
        "longitude": -118.0056,
        "location_name": "Huntington Beach"
    })

    assert isinstance(result, str)
    assert "Huntington Beach" in result
    # Check for wave metrics (height, period, or direction)
    has_wave_data = any(term in result for term in ["ft", "°", "Wave", "wave"])
    assert has_wave_data, "Response should contain wave data"
