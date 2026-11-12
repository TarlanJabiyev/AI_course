"""
===================================================================================
2. Google Gemini ilə Function Calling (Alətlər / Tools İstifadəsi) - google-genai SDK
===================================================================================

Bu skript Böyük Dil Modellərinin (LLM) ən güclü xüsusiyyətlərindən biri olan
"Function Calling" (və ya Tool Use — Alətlərin Çağırılması) mexanizmini rəsmi
Google GenAI (`from google import genai`) SDK-sı vasitəsilə nümayiş etdirir.

Function Calling Nədir və Niyə Lazımdır?
----------------------------------------
Dil modelləri (LLM) təlim olunduqları tarixdən sonrakı canlı məlumatları (məsələn:
hazırkı hava durumu, canlı valyuta məzənnəsi, verilənlər bazasındakı cari qalıq)
bilmirlər və birbaşa xarici sistemlərə çıxışları yoxdur.

Function Calling bu problemi həll edir:
1. Biz modelə Python funksiyamızı (adını, parametr növlərini və docstring təsvirini) təqdim edirik.
2. Model istifadəçinin sualını təhlil edir və başa düşür ki, bu suala cavab vermək üçün
   funksiya çağırılmalıdır.
3. Google GenAI SDK bu prosesi tam avtomatlaşdırır:
   - Funksiyanın Python tiplərindən və docstring-dən avtomatik JSON Schema yaradır (əllə yazmağa ehtiyac yoxdur).
   - Model funksiya çağırmaq istədikdə, SDK Python funksiyanı yerli mühitdə avtomatik icra edir.
   - Funksiyanın nəticəsi dərhal modelə qaytarılır və model istifadəçiyə səlis, təbii yekun cavab verir.
===================================================================================
"""

import os
import sys
import json
from google import genai
from google.genai import types

# Əgər layihədə .env faylı varsa, açarları avtomatik mühitə yükləyirik
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------------
# 1. Google Gemini API Açarının Əldə Edilməsi və Client Yaradılması
# ---------------------------------------------------------------------------------
# Əgər mühitdə (və ya .env faylında) açar varsa birbaşa oxunur, yoxdursa konsoldan istənilir:
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("\n" + "="*70)
    print("[DİQQƏT] Gemini API açarı mühitdə tapılmadı!")
    print("="*70)
    raw_key = input("Gemini API açarınızı bura daxil edin (AIzaSy...): ").strip()
    api_key = raw_key.strip('"').strip("'")

if not api_key:
    print("[XƏTA] API açarı daxil edilmədi. Proqram dayandırılır.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# İstifadə ediləcək ən son və tövsiyə olunan Gemini modeli:
MODEL_NAME = "gemini-3.8-flash"


# =================================================================================
# 2. Real və ya Simulyasiya Edilən Python Funksiyası (Tool / Alət)
# =================================================================================
# DİQQƏT: Google GenAI SDK-da funksiyanın Type Hint-ləri (məs: location: str) və
# Docstring təsviri (funksiyanın və arqumentlərin izahı) ÇOX VACİBDİR!
# Model məhz bu təsvirə baxaraq funksiyanın nə iş gördüyünü və nə vaxt
# çağırılmalı olduğunu anlayır.
# =================================================================================

def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Verilmiş şəhər və ya məkan üzrə cari hava durumunu əldə edir.

    Args:
        location: Şəhər və ya bölgə adı, məsələn: Bakı, London, İstanbul
        unit: Temperatur vahidi (celsius və ya fahrenheit)
    """
    print(f"\n" + "-"*50)
    print(f"[ALƏT İŞƏ DÜŞDÜ] Real Python funksiyası çağırıldı:")
    print(f" -> get_current_weather(location='{location}', unit='{unit}')")
    print("-"*50)

    # Real layihələrdə burada xarici hava xidmətinə (məs: OpenWeatherMap API) sorğu göndərilir.
    weather_info = {
        "location": location,
        "temperature": "22" if unit.lower() == "celsius" else "72",
        "unit": unit,
        "forecast": ["günəşli", "mülayim küləkli"],
    }
    return json.dumps(weather_info, ensure_ascii=False)


# =================================================================================
# 3. Dialoqun və Function Calling Prosesinin İdarə Edilməsi
# =================================================================================

def run_conversation(prompt: str) -> str:
    """
    İstifadəçi sualını və əlçatan alətləri (tools) Gemini modelinə göndərir.
    Əgər model alətə ehtiyac duyarsa, SDK funksiyanı avtomatik icra edib
    yekun təbii cavabı qaytarır.
    """
    print("\n" + "="*60)
    print("[ADDIM 1] Sual və Alət (get_current_weather) Gemini-yə göndərilir...")
    print("="*60)

    # `google-genai` SDK-da Python funksiyasını birbaşa `tools` siyahısına veririk.
    # SDK avtomatik olaraq:
    # 1. Funksiyanı JSON Schema-ya çevirir
    # 2. Model funksiya çağırışı tələb etdikdə onu icra edir
    # 3. Nəticəni modelə çatdırıb son istifadəçi cavabını alır
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[get_current_weather],
            temperature=0.0  # Deterministik və dəqiq qərar üçün temperatur 0 qoyulur
        )
    )

    return response.text


# =================================================================================
# 4. Proqramın İcrası (İstifadəçi ilə Əlaqə)
# =================================================================================
if __name__ == "__main__":
    print("\nGoogle Gemini Function Calling Nümunəsi İşə Salındı.")
    prompt = input("Hava haqqında bir sual verin (məs: Bakıda hava necədir?): ")
    
    final_answer = run_conversation(prompt)
    
    print("\n" + "="*60)
    print("YEKUN CAVAB (Modelin Təbii Dildə İzahı):")
    print("="*60)
    print(final_answer)


"""
===================================================================================
TEST ÜÇÜN ÖRNƏK PROMPTLAR (SUALLAR):
===================================================================================

Aşağıdakı sualları skriptdə yoxlayaraq Function Calling-in necə işlədiyini müşahidə edə bilərsiniz:

1. Funksiyanı dərhal işə salan birbaşa suallar:
   - "Bakıda hava necədir?"
   - "İstanbulda hazırda hava durumu necədir?"
   - "What's the weather like in London?"
   - "Nyu-Yorkda havanın temperaturu neçə dərəcədir?"
   (Nəticə: [ALƏT İŞƏ DÜŞDÜ] bildirişi görünəcək və funksiya icra olunacaq)

2. Parametrləri dəqiq göstərən suallar (Məkan + Temperatur vahidi):
   - "Tokioda hava necədir? Temperaturu Selsi (celsius) ilə göstər."
   - "Tell me the current weather in San Francisco in fahrenheit."
   - "Londonda havanı Farenheytlə yoxla."
   (Model avtomatik unit='celsius' və ya unit='fahrenheit' parametrini seçəcək)

3. Qeyri-birbaşa (Məntiqli) suallar (Modelin dərketmə gücünü yoxlamaq üçün):
   - "Sabah Parisə gedirəm, özümlə çətir götürməliyəm?"
     (Model anlayır ki, çətir üçün əvvəlcə Parisin havasını bilmək lazımdır və
      get_current_weather(location="Paris") çağırır)
   - "Şəkidə gəzintiyə çıxmaq üçün hava uyğundurmu?"

4. Funksiyanı çağırmayan adi suallar (Kontrol / Fərqi görmək üçün):
   - "Salam, kofe hazırlamaq üçün ən yaxşı üsul hansıdır?"
     (Bu zaman hava alətinə ehtiyac duyulmur, funksiya çağırılmır və
      model dərhal ümumi cavab verir)
===================================================================================
"""
