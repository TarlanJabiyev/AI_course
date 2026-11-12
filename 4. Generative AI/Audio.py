"""
===================================================================================
8. Google Gemini ilə Audio (Səs) Emalı: Transkripsiya və Tərcümə (google-genai SDK)
===================================================================================

Bu skript OpenAI-ın Audio nümunəsinə (`client.audio.transcriptions` və 
`client.audio.translations`) uyğun olaraq, səs fayllarının mətnə çevrilməsini 
və xarici dildən birbaşa tərcüməsini rəsmi google-genai (`from google import genai`) 
SDK-sı ilə nümayiş etdirir.

OpenAI və Google Gemini-də Audio Emalı Fərqi:
---------------------------------------------
1. OpenAI:
   - Səs üçün ayrıca ixtisaslaşmış modeldən istifadə edir (`model="whisper-1"`).
   - Transkripsiya və tərcümə üçün iki fərqli API endpointi çağırılır:
     `client.audio.transcriptions.create` və `client.audio.translations.create`.

2. Google Gemini:
   - Gemini "Native Multimodal" (anadangəlmə çoxmodallı) modeldir.
   - Səs üçün ayrıca modelə ehtiyac yoxdur; eyni `gemini-3.8-flash` modeli mətni, 
     şəkli və səsi birbaşa dərk edir.
   - `types.Part.from_bytes` vasitəsilə audio baytları modelə ötürülür və prompt 
     vasitəsilə həm dəqiq transkripsiya, həm də xarici dildən tərcümə tələb olunur.
===================================================================================
"""

import os
import sys
from google import genai
from google.genai import types

# Əgər dotenv varsa, .env faylından açarları avtomatik oxuyur
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------------
# 1. Google Gemini API Açarının Əldə Edilməsi və Client Yaradılması
# ---------------------------------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("\n" + "="*70)
    print("[DİQQƏT] Gemini API açarı mühitdə tapılmadı!")
    print("="*70)
    raw_key = input("Gemini API açarınızı daxil edin (AIzaSy...): ").strip()
    api_key = raw_key.strip('"').strip("'")

if not api_key:
    print("[XƏTA] API açarı daxil edilmədi. Proqram dayandırılır.")
    sys.exit(1)

client = genai.Client(api_key=api_key)
model_name = "gemini-3.8-flash"


# =================================================================================
# 2. Səsin Mətnə Çevrilməsi (Transcription - Speech to Text)
# =================================================================================
# OpenAI-dakı `client.audio.transcriptions.create(model="whisper-1", file=audio_file)`
# metodunun birbaşa Gemini qarşılığı:
# =================================================================================

print("Transcribing hi.mp3:\n")

audio_path = "hi.mp3"
if not os.path.exists(audio_path):
    print(f"[XƏTA] '{audio_path}' faylı qovluqda tapılmadı!")
    sys.exit(1)

with open(audio_path, "rb") as f:
    hi_audio_bytes = f.read()

# Gemini modelinə audio baytlarını və transkripsiya təlimatını göndəririk:
response = client.models.generate_content(
    model=model_name,
    contents=[
        types.Part.from_bytes(data=hi_audio_bytes, mime_type="audio/mp3"),
        "Transcribe this audio file accurately. Output only the transcribed text:"
    ]
)

print(response.text)


# =================================================================================
# 3. Xarici Dildəki Səsin Birbaşa İngiliscəyə Tərcüməsi (Audio Translation)
# =================================================================================
# OpenAI-dakı `client.audio.translations.create(model="whisper-1", file=audio_file)`
# metodunun birbaşa Gemini qarşılığı:
# Fransız dilində "Je ne parle pas français" ("Mən fransızca danışmıram") səsini
# dinləyərək birbaşa ingilis dilinə tərcümə edir:
# =================================================================================

print("\nTranslating french.mp3:\n")

french_audio_path = "french.mp3"
if not os.path.exists(french_audio_path):
    print(f"[XƏTA] '{french_audio_path}' faylı qovluqda tapılmadı!")
    sys.exit(1)

with open(french_audio_path, "rb") as f:
    french_audio_bytes = f.read()

response = client.models.generate_content(
    model=model_name,
    contents=[
        types.Part.from_bytes(data=french_audio_bytes, mime_type="audio/mp3"),
        "Translate this French audio into English. Output only the translated English text:"
    ]
)

# Will it actually work?
print(response.text)


"""
===================================================================================
QEYD VƏ ƏLAVƏ İMKANLAR:
===================================================================================

1. Inline Data (`types.Part.from_bytes`):
   - 20 MB-a qədər audio fayllar üçün idealdır. Faylı birbaşa RAM-dan göndərir,
     heç bir xarici serverə yükləmə gecikməsi olmur.

2. Böyük Audio Fayllar (Podkastlar, Mühazirələr) üçün Files API:
   - 20 MB-dan böyük audio fayllar üçün Google File API tövsiyə olunur:
     audio_file = client.files.upload(file="boyuk_fayl.mp3")
     response = client.models.generate_content(model=model_name, contents=[audio_file, "Transcribe"])

3. Çoxdilli Dəstək:
   - Gemini Azərbaycan, İngilis, Fransız, Rus, Türk və 100-dən çox dildə danışılan
     audionu dərk edir, transkripsiya və ya istənilən dilə tərcümə edə bilir.
===================================================================================
"""
