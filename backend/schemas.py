from pydantic import BaseModel, Field
from typing import Optional


class WeatherRequest(BaseModel):
    user_message: str = Field(..., example="Bugün ne giysem?")
    location: Optional[str] = Field(None, example="Sivas, TR")
    lat: Optional[float] = Field(None, example=39.7477)
    lon: Optional[float] = Field(None, example=37.0179)

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_message": "Bugün ne giysem?",
                "location": "Sivas, TR",
                "lat": 39.7477,
                "lon": 37.0179,
            }
        }
    }


class WeatherResponse(BaseModel):
    vibe: str = Field(..., description="2 cümlelik Türkçe tavsiye, emoji içerir")
    emoji: str = Field(..., description="Ana hava emojisi")
    share_text: str = Field(..., description="Telegram'da paylaşılabilir kısa metin")
    location_name: str


class AgriGuardResponse(BaseModel):
    frost_probability: float = Field(..., ge=0, le=100, description="Don olasılığı %")
    risk_level: str = Field(..., description="low / medium / high / critical")
    warning_text: str = Field(..., description="Türkçe çiftçi uyarısı")
    crop_stage: str
    action_required: bool


class RawWeatherData(BaseModel):
    temp: float
    feels_like: float
    humidity: int
    wind_speed: float
    wind_deg: int
    pressure: int
    weather_main: str          # "Snow", "Rain", "Clear" vb.
    weather_description: str
    clouds: int
    visibility: Optional[int] = None
    rain_1h: Optional[float] = None
    snow_1h: Optional[float] = None
    dew_point: Optional[float] = None
    location_name: str
    lat: float
    lon: float
    dt: int                    # Unix timestamp