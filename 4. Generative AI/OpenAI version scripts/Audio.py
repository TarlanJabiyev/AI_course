"""
===================================================================================
8. OpenAI Whisper ilə Audio (Səs) Emalı: Transkripsiya və Tərcümə
===================================================================================

Bu skript OpenAI-ın güclü "Whisper" nitq tanıma modeli (`whisper-1`) vasitəsilə 
səs fayllarının yüksək dəqiqliklə mətnə çevrilməsini (Transcription / Speech-to-Text) 
və xarici dildəki audionun birbaşa ingilis dilinə tərcüməsini (Translation) nümayiş etdirir.

OpenAI Whisper Modeli Haqqında:
-------------------------------
- Whisper internetdən toplanmış 680,000 saatlıq çoxdilli və müxtəlif vurğulu audio 
  məlumatları üzərində təlim keçmiş qabaqcıl neyron şəbəkəsidir (ASR - Automatic Speech Recognition).
- Küy, arxa plan səsləri və texniki jarqonları yüksək dəqiqliklə təmizləyib mətni bərpa edir.

Əsas 2 Metod:
--------------
1. `client.audio.transcriptions.create`: Audionu dinləyir və deyilən sözləri 
   danışıldığı dildə dəqiq mətnə çevirir (Speech-to-Text).
2. `client.audio.translations.create`: İstənilən xarici dildə (məs: fransız, alman, rus və s.) 
   danışılan audionu avtomatik başa düşür və birbaşa ingilis dilinə tərcümə edir.
===================================================================================
"""

import os
import sys
from openai import OpenAI

# Əgər layihədə .env faylı varsa, açarları avtomatik mühitə yükləyirik
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------------
# 1. OpenAI API Açarının Əldə Edilməsi və Client Yaradılması
# ---------------------------------------------------------------------------------
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    print("\n" + "="*70)
    print("[DİQQƏT] OpenAI API açarı mühitdə tapılmadı!")
    print("="*70)
    raw_key = input("OpenAI API açarınızı bura daxil edin (sk-...): ").strip()
    api_key = raw_key.strip('"').strip("'")

if not api_key:
    print("[XƏTA] API açarı daxil edilmədi. Proqram dayandırılır.")
    sys.exit(1)

client = OpenAI(api_key=api_key)
MODEL_NAME = "whisper-1"


# Köməkçi funksiya: Faylı həm cari, həm də bir üst qovluqda axtarır
def find_audio_file(filename: str) -> str:
    candidates = [
        filename,
        os.path.join("..", filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), "..", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return filename


# =================================================================================
# 2. Səsin Mətnə Çevrilməsi (Transcription - Speech to Text)
# =================================================================================
# 'hi.mp3' faylı oxunur və Whisper modeli tərəfindən danışıq birbaşa mətnə çevrilir.
# =================================================================================
print("Transcribing hi.mp3:\n")

hi_audio_path = find_audio_file("hi.mp3")
if not os.path.exists(hi_audio_path):
    print(f"[XƏTA] '{hi_audio_path}' faylı tapılmadı! Zəhmət olmasa 'hi.mp3' faylının qovluqda olduğunu yoxlayın.")
    sys.exit(1)

with open(hi_audio_path, "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=audio_file
    )

# Modelin qaytardığı transkripsiya obyekti:
print(transcript)
if hasattr(transcript, "text"):
    print(f"\n[Transkripsiya Mətni]: \"{transcript.text}\"")


# =================================================================================
# 3. Xarici Dildəki Səsin Birbaşa İngiliscəyə Tərcüməsi (Audio Translation)
# =================================================================================
# 'french.mp3' faylında fransızca danışıq ("Je ne parle pas français") qeyd olunub.
# `client.audio.translations.create` bu audionu dinləyərək dərhal ingiliscəyə çevirir.
# =================================================================================
print("\n" + "="*70)
print("Translating french.mp3:\n")

french_audio_path = find_audio_file("french.mp3")
if not os.path.exists(french_audio_path):
    print(f"[XƏTA] '{french_audio_path}' faylı tapılmadı! Zəhmət olmasa 'french.mp3' faylının qovluqda olduğunu yoxlayın.")
    sys.exit(1)

with open(french_audio_path, "rb") as audio_file:
    translation = client.audio.translations.create(
        model=MODEL_NAME,
        file=audio_file
    )

# Will it actually work? (Doğrudan da işləyəcəkmi?)
print(translation)
if hasattr(translation, "text"):
    print(f"\n[İngilis dilinə tərcümə]: \"{translation.text}\"")


"""
===================================================================================
QEYD VƏ ƏLAVƏ WHISPER İMKANLARI:
===================================================================================

1. Dəstəklənən Fayl Formatları və Ölçü Həddi:
   - Whisper 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm' formatlarını dəstəkləyir.
   - Maksimum fayl ölçüsü: 25 MB (Daha böyük faylları pydub ilə hissələrə bölmək tövsiyə olunur).

2. Çıxış Formatları (`response_format` parametri):
   - `json` (standart): {'text': '...'}
   - `text`: Birbaşa string mətni qaytarır.
   - `verbose_json`: Hər bir cümlə və kəlmənin dəqiq başlama və bitmə vaxt damğalarını (timestamps) qaytarır.
   - `srt` və ya `vtt`: YouTube və videolar üçün birbaşa subtitr faylı yaradır!

   Nümunə:
   client.audio.transcriptions.create(
       model="whisper-1",
       file=audio_file,
       response_format="verbose_json"
   )

3. Temperatur Parametri (`temperature`):
   - Standart olaraq 0.0 qoyulur (deterministik və dəqiq transkripsiya üçün).
===================================================================================
"""
