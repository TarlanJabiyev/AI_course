"""
===================================================================================
1. Google Gemini API ilə Mətn Generasiyası və Söhbət (google-genai SDK)
===================================================================================

Bu skript rəsmi "google-genai" (`from google import genai`) kitabxanası vasitəsilə
Google Gemini API-dan ən müasir və avtomatlaşdırılmış üsulla istifadə etməyi nümayiş etdirir.

Əsas Üstünlüklər:
1. `client = genai.Client()`: API açarı sistem mühit dəyişənindən (`GEMINI_API_KEY`
   və ya `GOOGLE_API_KEY`) və ya `.env` faylından avtomatik oxunur.
2. Ən son `gemini-3.8-flash` modeli birbaşa təyin edilir (əlavə axtarış sorğularına ehtiyac yoxdur).
3. `client.models.generate_content`: `system_instruction` və `temperature` parametrləri
   vasitəsilə dialoqun və modelin davranışının idarə olunması.
===================================================================================
"""

import os
import sys
from google import genai
from google.genai import types

# Əgər layihədə .env faylı varsa, açarı mühitə avtomatik yükləyirik
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
MODEL = "gemini-3.8-flash"


# =================================================================================
# NÜMUNƏ 1: Əsas Sual-Cavab və Model ilə Ümumi Dialoq
# =================================================================================
# `google-genai` SDK-da əsas metod `client.models.generate_content`-dir:
#   - `model`: Sorğunun göndəriləcəyi model adı
#   - `contents`: İstifadəçinin göndərdiyi sual / prompt
#   - `config`: Sistem təlimatı (`system_instruction`), `temperature` və digər parametrlər
# =================================================================================

prompt = input("Enter your prompt: ")

response = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config=types.GenerateContentConfig(
        # Modelə əsas davranış rolunu təyin edirik
        system_instruction="You are a helpful assistant."
    )
)

# ---------------------------------------------------------------------------------
# Modelin Cavabının Çıxarılması:
# `response.text` birbaşa modelin generasiya etdiyi əsas mətni qaytarır.
# ---------------------------------------------------------------------------------
print("\n" + "="*50)
print(f"Gemini ({MODEL}) Cavabı:")
print("="*50)
print(response.text)

# ---------------------------------------------------------------------------------
# API Cavabının Tam Strukturu:
# `response` obyekti mətndən əlavə metadata məlumatlarını da ehtiva edir:
#   - candidates: Generasiya edilən cavab variantları və tokenləri
#   - model_version: Cavab verən modelin dəqiq versiyası
#   - usage_metadata: Sərf olunan tokenlər (prompt_token_count, candidates_token_count)
# ---------------------------------------------------------------------------------
print("\n" + "="*50)
print("Complete response message (Tam metadata obyekti):")
print("="*50)
print(response)


# =================================================================================
# NÜMUNƏ 2: Təbii Dil Emalı (NLP) — Hissiyyat Təhlili (Sentiment Analysis)
# =================================================================================
# Bu nümunədə `system_instruction` vasitəsilə modeli xüsusi bir NLP təsnifatçısına
# (classifier) çeviririk.
# Dəqiq qayda: Mətni yalnız pozitiv və ya neqativ olaraq qiymətləndir.
# Bu yanaşma Prompt Engineering adlanır.
# =================================================================================

prompt = input("\nEnter a string for sentiment analysis: ")

response = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config=types.GenerateContentConfig(
        # Modeli təsnifatçı kimi ixtisaslaşdırırıq
        system_instruction="Classify user messages as positive or negative sentiment."
    )
)

print("\n" + "="*50)
print("Hissiyyat Təhlilinin Nəticəsi:")
print("="*50)
print(response.text)

print("\n" + "="*50)
print("Complete response message (Hissiyyat Təhlili metadata):")
print("="*50)
print(response)


# =================================================================================
# NÜMUNƏ 3: `temperature` Parametrinin Rolu və Yaradıcılığın İdarə Edilməsi
# =================================================================================
# `temperature` parametri 0.0 ilə 2.0 arasında qiymət alır:
#   - Aşağı temperatur (0.0 - 0.3):
#       * Model deterministik, konkret və daha az təsadüfi (predictable) cavablar verir.
#       * Faktiki suallar, kod yazılması, riyazi hesablamalar və təsnifat üçün idealdır.
#   - Orta temperatur (0.7):
#       * Ümumi söhbətlər və mətn generasiyası üçün balanslaşdırılmış standart dəyər.
#   - Yüksək temperatur (1.0 - 1.5+):
#       * Model daha gözlənilməz, yaradıcı və fərqli ifadələrdən istifadə edir.
#       * Hekayələr, şeirlər və beyin fırtınası (brainstorming) üçün idealdır.
# =================================================================================

print("\n" + "="*50)
print("Tell me a joke (with low temperature = 0.2):")
print("="*50)

response = client.models.generate_content(
    model=MODEL,
    contents="Tell me a joke.",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",
        temperature=0.2  # Aşağı yaradıcılıq dərəcəsi (daha sabit və deterministik nəticə)
    )
)

print("\n" + response.text)


"""
===================================================================================
TEST ÜÇÜN ÖRNƏK PROMPTLAR VƏ CÜMLƏLƏR:
===================================================================================

Aşağıdakı sual və cümlələri skriptin müvafiq bölmələrində sınaqdan keçirə bilərsiniz:

1. NÜMUNƏ 1 ÜÇÜN — Ümumi Sual-Cavab və Dialoq ("Enter your prompt:"):
   - "Süni intellekt (AI) ilə Maşın Öyrənməsi (Machine Learning) arasındakı fərqi sadə dildə izah et."
   - "Python-da siyahı (list) ilə kortej (tuple) arasındakı 3 əsas fərq nədir?"
   - "Explain quantum computing in one short paragraph for a 10-year-old."
   - "Bakı şəhərinin tarixi haqqında 3 maraqlı fakt yaz."

2. NÜMUNƏ 2 ÜÇÜN — Hissiyyat Təhlili / Sentiment Analysis ("Enter a string for sentiment analysis:"):
   Pozitiv (Positive) nümunələr:
   - "Bu məhsulun keyfiyyəti gözlədiyimdən də əladır, çatdırılma çox sürətli oldu!"
   - "I absolutely loved the new movie, the acting and soundtrack were masterpiece!"

   Neqativ (Negative) nümunələr:
   - "Xidmət səviyyəsi çox bərbaddır, sifarişim 2 saat gecikdi və soyuq gəldi."
   - "This was the worst customer support experience I have ever had."

   Mürəkkəb / Qarışıq (Sarkazm və ya gizli hissiyyat) nümunələr:
   - "Telefonun dizaynı qəşəngdir, amma batareyası 2 saata bitir, almağa dəyməz."
   - "Bəli, əlbəttə, proqramın hər 5 dəqiqədən bir çökməsi məni çox 'sevindirir'!"

3. NÜMUNƏ 3 ÜÇÜN — `temperature` Parametrinin Müqayisəli Testi:
   Kodda `temperature` dəyərini dəyişərək eyni sualda fərqi müşahidə edin:
   - `temperature=0.0` və ya `0.2` -> Hər dəfə işə saldıqda demək olar ki, eyni, konkret cavab verir.
   - `temperature=1.2` və ya `1.5` -> Hər dəfə tamamilə fərqli, yaradıcı və qeyri-adi zarafatlar / hekayələr qurur.
===================================================================================
"""
