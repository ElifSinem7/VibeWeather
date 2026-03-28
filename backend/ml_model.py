"""
VibeWeather — XGBoost Bias Correction Katmanı

Amaç: OpenWeatherMap API'nin Sivas gibi hiper-lokal noktalarda
yaptığı sistematik tahmin hatasını düzeltmek.

Şu an (veri seti yok): Mock veriyle train edilmiş örnek model
Sonra (Visual Crossing verisi gelince): gerçek 50 yıllık veriye geç

Kullanım:
    corrector = BiasCorrector()
    corrected = corrector.correct(raw_weather_dict)
    print(corrected["corrected_temp"])
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "ml_model.pkl"

# ─── Feature Engineering ──────────────────────────────────────────────────────

def extract_features(weather: dict) -> list[float]:
    """
    Ham hava verisinden XGBoost feature vektörü oluşturur.
    Plan dokümantasyonuyla birebir eşleşir:
    ['api_temp', 'api_humidity', 'api_wind', 'hour', 'month', 'pressure']
    """
    import datetime
    dt = datetime.datetime.fromtimestamp(weather.get("dt", 0))

    return [
        weather.get("temp", 0.0),            # api_temp
        weather.get("humidity", 50),          # api_humidity
        weather.get("wind_speed", 0.0),       # api_wind
        dt.hour,                              # hour (0-23)
        dt.month,                             # month (1-12)
        weather.get("pressure", 1013),        # pressure
    ]


# ─── Model Eğitimi (Mock Veri) ────────────────────────────────────────────────

def train_mock_model() -> object:
    """
    Gerçek Visual Crossing verisi gelene kadar Sivas iklimine uygun
    sentetik veriyle XGBoost modelini eğitir.

    Gerçek pipeline:
    1. Visual Crossing'den Sivas (39.75°N, 37.02°E) 50 yıllık veri çek
    2. OWM API tahminleriyle karşılaştır → bias = actual - api_temp
    3. Bu fonksiyonu gerçek X, y ile çalıştır
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logger.warning("XGBoost yüklü değil. pip install xgboost")
        return None

    np.random.seed(42)
    n = 5000

    # Sivas iklim özellikleri: kıta iklimi, yüksek rakım (1285m)
    # OWM genellikle Sivas'ı 1-3°C sıcak tahmin eder (kentsel ısı adası etkisi eksikliği)
    hours = np.random.randint(0, 24, n)
    months = np.random.randint(1, 13, n)

    # Mevsimsel sıcaklık dağılımı (Sivas gerçekçi)
    seasonal_temp = -5 + 20 * np.sin((months - 3) * np.pi / 6)
    api_temp = seasonal_temp + np.random.normal(0, 3, n)

    # Bias pattern:
    # - Kış gecelerinde OWM +2°C fazla tahmin eder (radyatif soğumayı kaçırır)
    # - Yaz öğlelerinde -1°C az tahmin eder (asfaltsız alan)
    night_mask = (hours < 6) | (hours > 21)
    winter_mask = (months <= 2) | (months >= 11)
    summer_mask = (months >= 6) & (months <= 8)

    bias = np.zeros(n)
    bias[night_mask & winter_mask] = np.random.normal(-2.0, 0.5, np.sum(night_mask & winter_mask))
    bias[summer_mask] = np.random.normal(1.0, 0.4, np.sum(summer_mask))
    bias += np.random.normal(0, 0.3, n)  # gürültü

    X = np.column_stack([
        api_temp,
        np.random.randint(30, 95, n),    # humidity
        np.abs(np.random.normal(3, 4, n)),  # wind
        hours,
        months,
        np.random.normal(1013, 8, n),    # pressure
    ])
    y = bias  # target: düzeltme miktarı (°C)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)

    logger.info(f"Mock model eğitildi. Bias aralığı: [{y.min():.2f}, {y.max():.2f}] °C")
    return model


# ─── BiasCorrector Sınıfı ─────────────────────────────────────────────────────

class BiasCorrector:
    """
    Singleton-safe XGBoost bias correction wrapper.
    Model yoksa otomatik train eder + pickle'a kaydeder.
    """

    def __init__(self):
        self.model = None
        self._load_or_train()

    def _load_or_train(self):
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"Model yüklendi: {MODEL_PATH}")
                return
            except Exception as e:
                logger.warning(f"Model yüklenemedi: {e}, yeniden eğitiliyor...")

        logger.info("Model bulunamadı, mock veriyle eğitiliyor...")
        self.model = train_mock_model()

        if self.model is not None:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(self.model, f)
            logger.info(f"Model kaydedildi: {MODEL_PATH}")

    def is_loaded(self) -> bool:
        return self.model is not None

    def correct(self, weather: dict) -> dict:
        """
        Ham hava verisini alır, bias-corrected sıcaklık döner.
        XGBoost model yoksa ham veriyi aynen döner.
        """
        result = dict(weather)

        if self.model is None:
            result["corrected_temp"] = weather.get("temp", 0.0)
            result["bias_applied"] = 0.0
            result["correction_source"] = "passthrough"
            return result

        try:
            features = extract_features(weather)
            X = np.array(features).reshape(1, -1)
            bias = float(self.model.predict(X)[0])

            result["corrected_temp"] = round(weather["temp"] + bias, 2)
            result["bias_applied"] = round(bias, 3)
            result["correction_source"] = "xgboost_mock"
        except Exception as e:
            logger.error(f"Bias correction hatası: {e}")
            result["corrected_temp"] = weather.get("temp", 0.0)
            result["bias_applied"] = 0.0
            result["correction_source"] = "error_passthrough"

        return result

    def retrain_with_real_data(self, X: np.ndarray, y: np.ndarray):
        """
        Visual Crossing verisi geldiğinde çağrılır.
        X: feature matrix, y: actual_temp - api_temp farkları
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise RuntimeError("xgboost paketi yüklü değil")

        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        self.model.fit(X, y)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)

        logger.info(f"Model gerçek veriyle yeniden eğitildi ve kaydedildi: {MODEL_PATH}")

    def feature_importance(self) -> Optional[dict]:
        """Feature importance skorları (debug/analiz amaçlı)."""
        if self.model is None:
            return None

        feature_names = ["api_temp", "api_humidity", "api_wind", "hour", "month", "pressure"]
        try:
            scores = self.model.feature_importances_
            return dict(zip(feature_names, scores.tolist()))
        except Exception:
            return None


# ─── CLI: Modeli elle eğitmek için ───────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("BiasCorrector başlatılıyor...")
    corrector = BiasCorrector()

    # Test
    mock_weather = {
        "temp": -3.0,
        "humidity": 78,
        "wind_speed": 7.8,
        "wind_deg": 270,
        "pressure": 1012,
        "weather_main": "Snow",
        "weather_description": "kar",
        "clouds": 90,
        "location_name": "Sivas",
        "lat": 39.7477,
        "lon": 37.0179,
        "dt": 1700000000,
        "dew_point": -5.4,
    }

    result = corrector.correct(mock_weather)
    print(f"\nHam sıcaklık  : {mock_weather['temp']}°C")
    print(f"Bias uygulanan: {result['bias_applied']:+.3f}°C")
    print(f"Düzeltilmiş   : {result['corrected_temp']}°C")
    print(f"Kaynak        : {result['correction_source']}")

    fi = corrector.feature_importance()
    if fi:
        print("\nFeature Importance:")
        for k, v in sorted(fi.items(), key=lambda x: -x[1]):
            print(f"  {k:20s}: {v:.4f}")