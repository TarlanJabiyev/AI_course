"""
===================================================================================
1. NVIDIA NIM API ilə Mətn Generasiyası və Söhbət (nvidia/nemotron-3.5-super-vl-120b-a12b)
===================================================================================

Bu skript NVIDIA NIM (`https://integrate.api.nvidia.com/v1`) platforması üzərində
işləyən güclü "nvidia/nemotron-3.5-super-vl-120b-a12b" modelindən istifadə etməyi
nümayiş etdirir.

Əsas Üstünlüklər:
1. `client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=...)`:
   API açarı sistem mühit dəyişənindən (`NVIDIA_API_KEY`) və ya `.env` faylından
   avtomatik oxunur.
2. `nvidia/nemotron-3.5-super-vl-120b-a12b`: NVIDIA-nın 120B parametrli (12B aktiv)
   çoxmodallı (Vision-Language) və dərin mühakimə (Thinking/Reasoning) modeli.
3. 3 Praktiki Nümayiş:
   - Nümunə 1: Əsas Sual-Cavab və Model ilə Ümumi Dialoq.
   - Nümunə 2: Təbii Dil Emalı (NLP) — Hissiyyat Təhlili (Sentiment Analysis).
   - Nümunə 3: `temperature` Parametrinin Rolu və Yaradıcılığın İdarə Edilməsi.
===================================================================================
"""

import os
import sys
from openai import OpenAI

# Əgər layihədə .env faylı varsa, açarı mühitə avtomatik yükləyirik
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------------
# 1. NVIDIA API Açarının Əldə Edilməsi və Client Yaradılması
# ---------------------------------------------------------------------------------
# Əgər mühitdə (və ya .env faylında) NVIDIA_API_KEY varsa birbaşa oxunur,
# yoxdursa konsoldan istənilir:
api_key = os.environ.get("NVIDIA_API_KEY")

if not api_key:
    print("\n" + "="*70)
    print("[DİQQƏT] NVIDIA API açarı mühitdə tapılmadı!")
    print("="*70)
    raw_key = input("NVIDIA API açarınızı bura daxil edin (nvapi-...): ").strip()
    api_key = raw_key.strip('"').strip("'")

if not api_key:
    print("[XƏTA] API açarı daxil edilmədi. Proqram dayandırılır.")
    sys.exit(1)

os.environ["NVIDIA_API_KEY"] = api_key

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

# İstifadə ediləcək əsas NVIDIA NIM modeli:
MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


# =================================================================================
# NÜMUNƏ 1: Əsas Sual-Cavab və Model ilə Ümumi Dialoq
# =================================================================================
# NVIDIA NIM API-da əsas metod `client.chat.completions.create`-dir:
#   - `model`: 'nvidia/nemotron-3.5-super-vl-120b-a12b'
#   - `messages`: Sistem təlimatı ("system") və istifadəçi sualı ("user")
#   - `extra_body`: `{"chat_template_kwargs": {"enable_thinking": True}}` ilə
#                   modelin daxili mühakimə (thinking) rejimi idarə olunur.
# =================================================================================

prompt = input("Enter your prompt: ")
if not prompt.strip():
    prompt = "Süni intellekt nədir və necə işləyir?"

completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=4096,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    stream=False
)

# ---------------------------------------------------------------------------------
# Modelin Cavabının Çıxarılması:
# `completion.choices[0].message.content` birbaşa modelin generasiya etdiyi mətni qaytarır.
# ---------------------------------------------------------------------------------
print("\n" + "="*50)
print(f"NVIDIA ({MODEL}) Cavabı:")
print("="*50)
print(completion.choices[0].message.content)

# ---------------------------------------------------------------------------------
# API Cavabının Tam Strukturu:
# `completion` obyekti mətndən əlavə metadata məlumatlarını da ehtiva edir:
#   - id: Sorğunun unikal identifikatoru
#   - model: Cavab verən modelin dəqiq versiyası
#   - choices: Generasiya edilən cavab variantları və daxili reasoning/thinking mətni
#   - usage: Sərf olunan tokenlər (prompt_tokens, completion_tokens, total_tokens)
# ---------------------------------------------------------------------------------
print("\n" + "="*50)
print("Complete response message (Tam metadata obyekti):")
print("="*50)
print(completion)


# =================================================================================
# NÜMUNƏ 2: Təbii Dil Emalı (NLP) — Hissiyyat Təhlili (Sentiment Analysis)
# =================================================================================
# Bu nümunədə `system` mesajı vasitəsilə modeli xüsusi bir NLP təsnifatçısına
# (classifier) çeviririk.
# Dəqiq qayda: Mətni yalnız pozitiv və ya neqativ olaraq qiymətləndir.
# Bu yanaşma Prompt Engineering adlanır.
# =================================================================================

prompt = input("\nEnter a string for sentiment analysis: ")
if not prompt.strip():
    prompt = "Bu məhsulun keyfiyyəti gözlədiyimdən də əladır!"

completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Classify user messages as positive or negative sentiment."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2,
    top_p=0.95,
    max_tokens=1024,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)

print("\n" + "="*50)
print("Hissiyyat Təhlilinin Nəticəsi:")
print("="*50)
print(completion.choices[0].message.content)

print("\n" + "="*50)
print("Complete response message (Hissiyyat Təhlili metadata):")
print("="*50)
print(completion)


# =================================================================================
# NÜMUNƏ 3: `temperature` Parametrinin Rolu və Yaradıcılığın İdarə Edilməsi
# =================================================================================
# `temperature` parametri 0.0 ilə 2.0 arasında qiymət alır:
#   - Aşağı temperatur (0.0 - 0.3):
#       * Model deterministik, konkret və daha az təsadüfi (predictable) cavablar verir.
#       * Faktiki suallar, kod yazılması, riyazi hesablamalar və təsnifat üçün idealdır.
#   - Orta temperatur (0.7 - 1.0):
#       * Ümumi söhbətlər və mətn generasiyası üçün balanslaşdırılmış standart dəyər.
#   - Yüksək temperatur (1.2 - 1.5+):
#       * Model daha gözlənilməz, yaradıcı və fərqli ifadələrdən istifadə edir.
#       * Hekayələr, şeirlər və beyin fırtınası (brainstorming) üçün idealdır.
# =================================================================================

print("\n" + "="*50)
print("Tell me a joke (with low temperature = 0.2):")
print("="*50)

completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
    temperature=0.2,  # Aşağı yaradıcılıq dərəcəsi (daha sabit və deterministik nəticə)
    top_p=0.95,
    max_tokens=1024,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)

print("\n" + completion.choices[0].message.content)


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
