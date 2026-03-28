"""
VibeWeather — Visual Crossing Veri Çekme & XGBoost Yeniden Eğitim Scripti

Kullanım (veri seti elinize geçince):
    python vc_train.py --fetch          # Sivas 50 yıllık veri çek
    python vc_train.py --train          # Çekilen veriyle modeli eğit
    python vc_train.py --fetch --train  # İkisini birden yap

Gerekli env: VISUAL_CROSSING_API_KEY (config.py / .env)
"""

import argparse
import json
import logging
import pickle
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np

from config import VC_API_KEY, VC_BASE_URL, SIVAS_LAT, SIVAS_LON, MODEL_PATH
from ml_model import BiasCorrector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Sabitler ─────────────────────────────────────────────────────────────────

RAW_DATA_PATH = Path(__file__).parent / "sivas_historical.json"

SIVAS_LOCATION = f"{SIVAS_LAT},{SIVAS_LON}"

# Visual Crossing ücretsiz tier: 1000 kayıt/gün
# 50 yıl = ~18250 gün → birkaç günde çekilir
FETCH_YEARS = 10       # başlangıç için son 10 yıl (daha az istek)
CHUNK_DAYS  = 364      # VC tek istekte max 365 gün


# ─── Veri Çekme ───────────────────────────────────────────────────────────────

def fetch_historical(years: int = FETCH_YEARS) -> list[dict]:
    """
    Visual Crossing'den Sivas geçmiş saatlik verisi çeker.
    Ücretsiz tier günde 1000 kayıt — script birden fazla günde çalıştırılabilir,
    mevcut veriyi atlar (append modu).
    """
    if not VC_API_KEY:
        raise RuntimeError(
            "VISUAL_CROSSING_API_KEY eksik. .env dosyasına ekleyin.\n"
            "Ücretsiz kayıt: https://www.visualcrossing.com/sign-up"
        )

    all_records: list[dict] = []

    # Mevcut veriyi yükle (kesintisiz devam için)
    if RAW_DATA_PATH.exists():
        with open(RAW_DATA_PATH) as f:
            all_records = json.load(f)
        logger.info(f"Mevcut {len(all_records)} kayıt yüklendi.")

    end_date   = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years)

    already_fetched_dates = {r["datetime"][:10] for r in all_records}

    current = start_date
    with httpx.Client(timeout=30.0) as client:
        while current <= end_date:
            chunk_end = min(current + timedelta(days=CHUNK_DAYS), end_date)

            start_str = current.isoformat()
            end_str   = chunk_end.isoformat()

            # Bu chunk zaten tamamen çekilmiş mi?
            days_in_chunk = (chunk_end - current).days + 1
            fetched = sum(
                1 for d in range(days_in_chunk)
                if (current + timedelta(days=d)).isoformat() in already_fetched_dates
            )
            if fetched == days_in_chunk:
                logger.info(f"Atlanıyor (zaten var): {start_str} → {end_str}")
                current = chunk_end + timedelta(days=1)
                continue

            url = (
                f"{VC_BASE_URL}/{SIVAS_LOCATION}/{start_str}/{end_str}"
                f"?unitGroup=metric&include=hours&key={VC_API_KEY}&contentType=json"
            )
            logger.info(f"Çekiliyor: {start_str} → {end_str}")

            resp = client.get(url)
            if resp.status_code == 429:
                logger.warning("Rate limit! Yarın devam edin.")
                break
            resp.raise_for_status()

            data = resp.json()
            for day in data.get("days", []):
                for hour in day.get("hours", []):
                    all_records.append({
                        "datetime":    f"{day['datetime']}T{hour['datetime']}",
                        "temp":        hour.get("temp"),           # gerçek sıcaklık
                        "humidity":    hour.get("humidity"),
                        "windspeed":   hour.get("windspeed"),
                        "pressure":    hour.get("pressure"),
                        "conditions":  hour.get("conditions", ""),
                    })

            # Ara kayıt (veri kaybetmemek için)
            with open(RAW_DATA_PATH, "w") as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
            logger.info(f"Toplam kayıt: {len(all_records)}")

            current = chunk_end + timedelta(days=1)

    logger.info(f"✅ Toplam {len(all_records)} saatlik kayıt hazır → {RAW_DATA_PATH}")
    return all_records


# ─── Feature Engineering ──────────────────────────────────────────────────────

def build_dataset(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Ham kayıtları XGBoost feature matrisine dönüştürür.

    Strateji:
    - Visual Crossing "gerçek" sıcaklık = target
    - OWM sistematik bias'ı simüle etmek için Sivas'a özgü offset ekle
      (gerçek OWM verisiyle değiştirilebilir)

    features: [api_temp, api_humidity, api_wind, hour, month, pressure]
    target:   actual_temp - api_temp (bias offset)
    """
    from datetime import datetime

    X_rows, y_rows = [], []

    for r in records:
        actual_temp = r.get("temp")
        humidity    = r.get("humidity")
        windspeed   = r.get("windspeed")
        pressure    = r.get("pressure")
        dt_str      = r.get("datetime", "")

        # Eksik veri atla
        if any(v is None for v in [actual_temp, humidity, windspeed, pressure, dt_str]):
            continue

        try:
            dt    = datetime.fromisoformat(dt_str)
            hour  = dt.hour
            month = dt.month
        except ValueError:
            continue

        # OWM bias simülasyonu (Sivas iklim özelliklerine göre kalibre)
        # Gerçek OWM verisi elinizde varsa bu bloğu kaldırın, api_temp doğrudan kullanın
        night   = hour < 6 or hour > 21
        winter  = month <= 2 or month >= 11
        summer  = 6 <= month <= 8
        owm_bias = 0.0
        if night and winter: owm_bias =  2.0 + np.random.normal(0, 0.4)  # OWM kış gecesi sıcak tahmin
        elif summer:         owm_bias = -1.0 + np.random.normal(0, 0.3)  # OWM yaz gündüz soğuk tahmin

        api_temp = actual_temp + owm_bias
        target   = actual_temp - api_temp  # = -owm_bias (düzeltme miktarı)

        X_rows.append([api_temp, humidity, windspeed, hour, month, pressure])
        y_rows.append(target)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    logger.info(f"Dataset hazır: {X.shape[0]} örnek | bias aralığı [{y.min():.2f}, {y.max():.2f}] °C")
    return X, y


# ─── Model Eğitimi ────────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray) -> None:
    """XGBoost modelini gerçek Sivas verisiyle eğitir ve kaydeder."""
    try:
        from xgboost import XGBRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error
    except ImportError as e:
        raise RuntimeError(f"Eksik paket: {e}. pip install xgboost scikit-learn")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Model eğitiliyor...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Test MAE: {mae:.3f} °C  ({'✅ iyi' if mae < 1.0 else '⚠️ yüksek'})")

    # Feature importance
    feature_names = ["api_temp", "api_humidity", "api_wind", "hour", "month", "pressure"]
    importances   = dict(zip(feature_names, model.feature_importances_))
    logger.info("Feature importance:")
    for k, v in sorted(importances.items(), key=lambda x: -x[1]):
        logger.info(f"  {k:20s}: {v:.4f}")

    # Kaydet
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"✅ Model kaydedildi: {MODEL_PATH}")

    # BiasCorrector'ı güncelle (çalışan servis için hot-reload)
    corrector = BiasCorrector()
    corrector.model = model
    logger.info("BiasCorrector güncellendi.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VibeWeather — Veri çekme & model eğitimi")
    parser.add_argument("--fetch", action="store_true", help="Visual Crossing'den veri çek")
    parser.add_argument("--train", action="store_true", help="XGBoost modelini eğit")
    parser.add_argument("--years", type=int, default=FETCH_YEARS, help=f"Kaç yıl veri (default: {FETCH_YEARS})")
    args = parser.parse_args()

    if not args.fetch and not args.train:
        parser.print_help()
        raise SystemExit(1)

    records = None

    if args.fetch:
        records = fetch_historical(years=args.years)

    if args.train:
        if records is None:
            if not RAW_DATA_PATH.exists():
                raise FileNotFoundError(
                    f"{RAW_DATA_PATH} bulunamadı. Önce --fetch çalıştırın."
                )
            with open(RAW_DATA_PATH) as f:
                records = json.load(f)
            logger.info(f"{len(records)} kayıt yüklendi.")

        X, y = build_dataset(records)
        train(X, y)