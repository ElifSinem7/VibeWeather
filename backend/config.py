"""
VibeWeather — Merkezi Ayar Yönetimi
Tüm env değişkenleri buradan okunur, diğer modüller buradan import eder.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle (Docker dışında local geliştirme için)
load_dotenv(Path(__file__).parent.parent / ".env")


# ─── OpenWeatherMap ───────────────────────────────────────────────────────────

OWM_API_KEY  = os.getenv("OPENWEATHERMAP_API_KEY", "")
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5"

# ─── Visual Crossing (XGBoost eğitim verisi) ─────────────────────────────────

VC_API_KEY   = os.getenv("VISUAL_CROSSING_API_KEY", "")
VC_BASE_URL  = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

# ─── Ollama / LLM ────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.1:8b")

# ─── Uygulama ─────────────────────────────────────────────────────────────────

APP_ENV   = os.getenv("APP_ENV", "production")   # development | production
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── Sivas Koordinatları (default hedef şehir) ───────────────────────────────

SIVAS_LAT = 39.7477
SIVAS_LON = 37.0179

# ─── Model Dosya Yolu ─────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "ml_model.pkl"

# ─── Doğrulama ───────────────────────────────────────────────────────────────

def validate():
    """Uygulama başlarken kritik env değişkenlerini kontrol eder."""
    missing = []
    if not OWM_API_KEY:
        missing.append("OPENWEATHERMAP_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Eksik environment değişkenleri: {', '.join(missing)}\n"
            f"cp .env.example .env yapıp doldurun."
        )


if __name__ == "__main__":
    validate()
    print("✅ Tüm zorunlu env değişkenleri mevcut")
    print(f"   Ollama : {OLLAMA_BASE_URL} / {OLLAMA_MODEL}")
    print(f"   OWM    : {'*' * (len(OWM_API_KEY) - 4) + OWM_API_KEY[-4:] if OWM_API_KEY else 'EKSİK'}")
    print(f"   VC     : {'*' * (len(VC_API_KEY)  - 4) + VC_API_KEY[-4:]  if VC_API_KEY  else 'boş (opsiyonel)'}")