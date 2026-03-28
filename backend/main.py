from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn

from config import validate, APP_ENV
from agent import VibeWeatherAgent
from ml_model import BiasCorrector
from schemas import WeatherRequest, WeatherResponse, AgriGuardResponse

validate()

app = FastAPI(
    title="VibeWeather API",
    description="AI-Powered Social Weather Assistant — Anadolu Hackathon 2026",
    version="1.0.0",
    docs_url="/docs" if APP_ENV == "development" else "/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Telegram Mini App için
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton'lar
agent = VibeWeatherAgent()
bias_corrector = BiasCorrector()


@app.get("/")
def root():
    return {"status": "ok", "project": "VibeWeather", "hackathon": "Anadolu 2026"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": bias_corrector.is_loaded()}


# ─── Ana Endpoint: Kıyafet / Vibe Tavsiyesi ───────────────────────────────────

@app.post("/vibe", response_model=WeatherResponse)
async def get_vibe(request: WeatherRequest):
    """
    Kullanıcıya lokasyona göre 2 cümlelik vibe tavsiyesi üretir.
    ReAct Agent → get_weather() tool → GPT-4o-mini → Türkçe tavsiye
    """
    try:
        result = await agent.run(
            user_message=request.user_message,
            location=request.location,
            lat=request.lat,
            lon=request.lon,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── AgriGuard Endpoint ───────────────────────────────────────────────────────

@app.post("/agriguard", response_model=AgriGuardResponse)
async def agriguard(
    lat: float = Query(..., description="Enlem"),
    lon: float = Query(..., description="Boylam"),
    crop_stage: str = Query("cicek", description="Bitki fenoloji evresi: tohum/filiz/cicek/meyve"),
):
    """
    Çiftçi modu: don riski skoru + Türkçe proaktif uyarı üretir.
    """
    try:
        result = await agent.run_agriguard(lat=lat, lon=lon, crop_stage=crop_stage)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Ham Hava Verisi (debug / frontend test) ──────────────────────────────────

@app.get("/weather/raw")
async def raw_weather(
    lat: float = Query(...),
    lon: float = Query(...),
):
    """
    Bias-corrected ham hava verisini döndürür (kullanıcıya gösterilmez, debug amaçlı).
    """
    try:
        raw = await agent.fetch_weather(lat=lat, lon=lon)
        corrected = bias_corrector.correct(raw)
        return {"raw": raw, "corrected": corrected}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)