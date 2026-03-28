"""
VibeWeather — OpenWeatherMap Client

OWM API çağrılarını agent.py'den ayırır.
Tek sorumluluk: ham hava verisini çek + çiğ noktasını hesapla.
Bias correction ml_model.py'de kalır.
"""

import httpx
from typing import Optional
from config import OWM_API_KEY, OWM_BASE_URL


class WeatherClient:
    """OpenWeatherMap REST client. Async, context manager uyumlu."""

    def __init__(self, timeout: float = 10.0):
        self._client = httpx.AsyncClient(timeout=timeout)

    async def get_current(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        city: Optional[str] = None,
    ) -> dict:
        """
        Anlık hava durumu çeker.
        lat/lon veya city parametrelerinden biri zorunlu.
        Dönen dict: ham OWM verisi + hesaplanan dew_point.
        """
        if not OWM_API_KEY:
            raise RuntimeError("OPENWEATHERMAP_API_KEY env değişkeni eksik")

        if lat is not None and lon is not None:
            params = {
                "lat": lat, "lon": lon,
                "appid": OWM_API_KEY,
                "units": "metric",
                "lang": "tr",
            }
        elif city:
            params = {
                "q": city,
                "appid": OWM_API_KEY,
                "units": "metric",
                "lang": "tr",
            }
        else:
            raise ValueError("lat/lon veya city parametresi gerekli")

        resp = await self._client.get(f"{OWM_BASE_URL}/weather", params=params)
        resp.raise_for_status()
        d = resp.json()

        T  = d["main"]["temp"]
        RH = d["main"]["humidity"]

        return {
            # Sıcaklık & nem
            "temp":                T,
            "feels_like":          d["main"]["feels_like"],
            "temp_min":            d["main"]["temp_min"],
            "temp_max":            d["main"]["temp_max"],
            "humidity":            RH,
            "dew_point":           round(T - ((100 - RH) / 5), 2),  # Magnus yaklaşımı

            # Rüzgar
            "wind_speed":          d["wind"]["speed"],
            "wind_deg":            d["wind"].get("deg", 0),
            "wind_gust":           d["wind"].get("gust"),

            # Basınç & görüş
            "pressure":            d["main"]["pressure"],
            "sea_level":           d["main"].get("sea_level"),
            "grnd_level":          d["main"].get("grnd_level"),
            "visibility":          d.get("visibility"),

            # Durum
            "weather_id":          d["weather"][0]["id"],
            "weather_main":        d["weather"][0]["main"],
            "weather_description": d["weather"][0]["description"],
            "weather_icon":        d["weather"][0]["icon"],
            "clouds":              d["clouds"]["all"],

            # Yağış (varsa)
            "rain_1h":             d.get("rain", {}).get("1h"),
            "rain_3h":             d.get("rain", {}).get("3h"),
            "snow_1h":             d.get("snow", {}).get("1h"),
            "snow_3h":             d.get("snow", {}).get("3h"),

            # Konum
            "location_name":       d["name"],
            "country":             d["sys"]["country"],
            "lat":                 d["coord"]["lat"],
            "lon":                 d["coord"]["lon"],

            # Zaman
            "dt":                  d["dt"],
            "sunrise":             d["sys"]["sunrise"],
            "sunset":              d["sys"]["sunset"],
            "timezone":            d.get("timezone", 0),
        }

    async def get_forecast_5day(
        self,
        lat: float,
        lon: float,
    ) -> list[dict]:
        """
        5 günlük / 3 saatlik tahmin (AgriGuard gece don riski için).
        OWM free tier'da mevcut.
        """
        params = {
            "lat": lat, "lon": lon,
            "appid": OWM_API_KEY,
            "units": "metric",
            "lang": "tr",
            "cnt": 16,   # 16 x 3saat = 2 gün (gece saatleri için yeterli)
        }
        resp = await self._client.get(f"{OWM_BASE_URL}/forecast", params=params)
        resp.raise_for_status()
        data = resp.json()

        return [
            {
                "dt":          item["dt"],
                "temp":        item["main"]["temp"],
                "feels_like":  item["main"]["feels_like"],
                "humidity":    item["main"]["humidity"],
                "wind_speed":  item["wind"]["speed"],
                "weather_main": item["weather"][0]["main"],
                "rain_3h":     item.get("rain", {}).get("3h"),
                "snow_3h":     item.get("snow", {}).get("3h"),
                "pop":         item.get("pop", 0),   # precipitation probability
            }
            for item in data.get("list", [])
        ]

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()