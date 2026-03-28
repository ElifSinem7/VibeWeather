"""
VibeWeather — ReAct LLM Agent (Ollama / Llama3 uyumlu)

Strateji:
  1. Hava verisini her zaman önceden çek (güvenilir)
  2. Llama'nın tool calling'i varsa kullan, yoksa prompt-inject fallback
  Ollama, OpenAI-uyumlu API sunar — sadece base_url değişir.
"""

import os
import json
import httpx
from typing import Optional
from openai import AsyncOpenAI
from ml_model import BiasCorrector
from schemas import WeatherResponse, AgriGuardResponse

# ─── Sabitler ────────────────────────────────────────────────────────────────

OWM_BASE = "https://api.openweathermap.org/data/2.5"
OWM_KEY  = os.getenv("OPENWEATHERMAP_API_KEY", "")

# Docker Compose içinde servis adı "ollama", dışarıdan localhost:11434
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
MODEL           = os.getenv("OLLAMA_MODEL",    "llama3.1:8b")

# ─── System Prompts ───────────────────────────────────────────────────────────

VIBE_SYSTEM_PROMPT = """Sen VibeWeather'sın — Türkiye'nin en samimi hava asistanı.

KURALLAR:
1. Teknik veri (derece, hPa, km/h, nem yüzdesi) ASLA kullanıcıya gösterme.
2. Maksimum 2 cümle: BİRİNCİSİ vibe/duygu, İKİNCİSİ somut eylem.
3. Türkçe ve samimi dil kullan.
4. Her yanıta uygun bir emoji ile başla.

ÖRNEK ÇIKTILAR:
❄️ Ghosting weather. Kalın mont + termos çay — bu soğukta dışarıda işin yok.
🌧️ Hesabı isteyin, sağanak kapıda. Şemsiye değil, yağmurluk gün.
☀️ Sivas bugün sana gülümsüyor. Hafif bir hırka yeter, güzel bir gün."""

AGRIGUARD_SYSTEM_PROMPT = """Sen VibeWeather AgriGuard modülüsün — Sivas çiftçilerine don uyarısı yapan AI.

KURALLAR:
1. Don riski yüzdesi ve bitki evresine göre uyarı yaz.
2. Türkçe, net, acil eylem odaklı — teknik terim kullanma.
3. Maksimum 2 cümle.

FENOLOJI EVRELERİ:
- tohum: düşük risk (tohumlar dona dayanıklı)
- filiz: orta risk
- cicek: KRİTİK — don = yüzde seksen-yüz rekolte kaybı
- meyve: yüksek risk"""

# ─── Tool Tanımları (llama3.1:8b+ destekler) ─────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Verilen koordinat veya şehir için bias-corrected anlık hava durumu döner.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat":  {"type": "number", "description": "Enlem"},
                    "lon":  {"type": "number", "description": "Boylam"},
                    "city": {"type": "string", "description": "Şehir adı"},
                },
            },
        },
    }
]


# ─── Ana Agent ────────────────────────────────────────────────────────────────

class VibeWeatherAgent:

    def __init__(self):
        # Ollama, OpenAI SDK'sıyla birebir uyumlu — sadece base_url değişir
        self.client = AsyncOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama API key gerektirmez
        )
        self.bias_corrector = BiasCorrector()
        self._http = httpx.AsyncClient(timeout=15.0)

    # ── Hava Verisi ───────────────────────────────────────────────────────

    async def fetch_weather(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        city: Optional[str] = None,
    ) -> dict:
        """OpenWeatherMap → XGBoost bias correction → temiz dict döner."""
        if lat is not None and lon is not None:
            params = {"lat": lat, "lon": lon, "appid": OWM_KEY, "units": "metric", "lang": "tr"}
        elif city:
            params = {"q": city, "appid": OWM_KEY, "units": "metric", "lang": "tr"}
        else:
            raise ValueError("lat/lon veya city gerekli")

        resp = await self._http.get(f"{OWM_BASE}/weather", params=params)
        resp.raise_for_status()
        d = resp.json()

        T  = d["main"]["temp"]
        RH = d["main"]["humidity"]

        raw = {
            "temp":                T,
            "feels_like":          d["main"]["feels_like"],
            "humidity":            RH,
            "wind_speed":          d["wind"]["speed"],
            "wind_deg":            d["wind"].get("deg", 0),
            "pressure":            d["main"]["pressure"],
            "weather_main":        d["weather"][0]["main"],
            "weather_description": d["weather"][0]["description"],
            "clouds":              d["clouds"]["all"],
            "visibility":          d.get("visibility"),
            "rain_1h":             d.get("rain", {}).get("1h"),
            "snow_1h":             d.get("snow", {}).get("1h"),
            "dew_point":           T - ((100 - RH) / 5),
            "location_name":       d["name"],
            "lat":                 d["coord"]["lat"],
            "lon":                 d["coord"]["lon"],
            "dt":                  d["dt"],
        }

        corrected = self.bias_corrector.correct(raw)
        raw["temp"] = corrected["corrected_temp"]
        return raw

    # ── Tool Dispatcher ───────────────────────────────────────────────────

    async def _dispatch_tool(self, name: str, args: dict, ctx: dict) -> str:
        if name == "get_weather":
            data = await self.fetch_weather(
                lat=args.get("lat") or ctx.get("lat"),
                lon=args.get("lon") or ctx.get("lon"),
                city=args.get("city") or ctx.get("location"),
            )
            return json.dumps(data, ensure_ascii=False)
        return json.dumps({"error": f"Bilinmeyen tool: {name}"})

    # ── Vibe Akışı ────────────────────────────────────────────────────────

    async def run(
        self,
        user_message: str,
        location: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> WeatherResponse:
        """
        Llama uyumlu iki katmanlı strateji:
          A) Hava verisini önceden çek — her iki yolda da kullan
          B) Tool calling çalışırsa → ReAct tamamla
          C) Çalışmazsa → hava verisini doğrudan prompt'a göm (fallback)
        """
        weather = await self.fetch_weather(lat=lat, lon=lon, city=location)
        loc_name = weather.get("location_name", location or "Bilinmeyen")
        weather_str = json.dumps(weather, ensure_ascii=False)

        vibe_text = None

        # ── Yol A: Tool calling (llama3.1:8b+ destekler) ──────────────────
        try:
            r1 = await self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": VIBE_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"{user_message} (Konum: {loc_name})"},
                ],
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=300,
            )
            m1 = r1.choices[0].message

            if m1.tool_calls:
                tool_results = []
                for tc in m1.tool_calls:
                    result = await self._dispatch_tool(
                        tc.function.name,
                        json.loads(tc.function.arguments),
                        {"location": location, "lat": lat, "lon": lon},
                    )
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                r2 = await self.client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": VIBE_SYSTEM_PROMPT},
                        {"role": "user",   "content": f"{user_message} (Konum: {loc_name})"},
                        m1,
                        *tool_results,
                    ],
                    max_tokens=300,
                )
                vibe_text = r2.choices[0].message.content.strip()

            elif m1.content and len(m1.content.strip()) > 10:
                vibe_text = m1.content.strip()

        except Exception:
            pass  # Fallback'e düş

        # ── Yol B: Prompt-inject fallback (her Llama versiyonuyla çalışır) ──
        if not vibe_text:
            r_fb = await self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": VIBE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Hava durumu bilgisi (kullanıcıya gösterme): {weather_str}\n\n"
                            f"Kullanıcı sorusu: {user_message}\n\n"
                            "Yukarıdaki hava verisine göre 2 cümle Türkçe tavsiye yaz. "
                            "Teknik değer yazma, emoji ile başla."
                        ),
                    },
                ],
                max_tokens=300,
            )
            vibe_text = r_fb.choices[0].message.content.strip()

        emoji = self._extract_emoji(vibe_text)
        return WeatherResponse(
            vibe=vibe_text,
            emoji=emoji,
            share_text=self._make_share_text(vibe_text, loc_name),
            location_name=loc_name,
        )

    # ── AgriGuard Akışı ───────────────────────────────────────────────────

    async def run_agriguard(
        self,
        lat: float,
        lon: float,
        crop_stage: str = "cicek",
    ) -> AgriGuardResponse:
        stage_multipliers = {"tohum": 0.5, "filiz": 1.2, "cicek": 2.0, "meyve": 1.5}
        multiplier = stage_multipliers.get(crop_stage, 1.0)

        weather  = await self.fetch_weather(lat=lat, lon=lon)
        frost_p  = self._frost_probability(weather, multiplier)
        risk_lvl = self._risk_level(frost_p)

        r = await self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": AGRIGUARD_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Hava: {json.dumps(weather, ensure_ascii=False)}\n"
                        f"Don olasılığı: yüzde{frost_p:.0f} | Evre: {crop_stage} | Risk: {risk_lvl}\n\n"
                        "Çiftçiye 2 cümle Türkçe uyarı yaz."
                    ),
                },
            ],
            max_tokens=200,
        )
        warning = r.choices[0].message.content.strip()

        return AgriGuardResponse(
            frost_probability=round(frost_p, 1),
            risk_level=risk_lvl,
            warning_text=warning,
            crop_stage=crop_stage,
            action_required=frost_p >= 50,
        )

    # ── Yardımcılar ───────────────────────────────────────────────────────

    def _frost_probability(self, w: dict, multiplier: float) -> float:
        temp = w["temp"]; dew = w.get("dew_point", temp - 5); wind = w["wind_speed"]
        base = 0.0
        if   temp <= 0: base = 90.0
        elif temp <= 2: base = 60.0
        elif temp <= 4: base = 30.0
        elif temp <= 6: base = 10.0
        if wind < 2:    base *= 1.2
        if dew  <= 0:   base  = min(100, base * 1.3)
        return min(100.0, base * multiplier)

    def _risk_level(self, p: float) -> str:
        if p >= 75: return "critical"
        if p >= 50: return "high"
        if p >= 25: return "medium"
        return "low"

    def _extract_emoji(self, text: str) -> str:
        for c in text:
            if ord(c) > 127:
                return c
        return "🌤️"

    def _make_share_text(self, vibe: str, location: str) -> str:
        first = vibe.split(".")[0].strip()
        return f"{first} — VibeWeather ile kontrol et 🌦️"

    async def __aenter__(self): return self
    async def __aexit__(self, *a): await self._http.aclose()