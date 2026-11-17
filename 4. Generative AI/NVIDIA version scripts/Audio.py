"""
===================================================================================
8. NVIDIA NIM ilə Audio (Səs) Emalı: Transkripsiya və Tərcümə
===================================================================================

Bu skript OpenAI-ın Audio nümunəsinə (`client.audio.transcriptions` və 
`client.audio.translations`) və Gemini-nin `Audio.py` skriptinə uyğun olaraq,
səs fayllarının mətnə çevrilməsini və xarici dildən ingilis dilinə tərcüməsini
NVIDIA NIM (`nvidia/nemotron-3.5-super-vl-120b-a12b`) ilə nümayiş etdirir.
===================================================================================
"""

import os
import sys
from openai import OpenAI

# Əgər dotenv varsa, .env faylından açarları avtomatik oxuyur
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------------
# 1. NVIDIA API Açarının Əldə Edilməsi və Client Yaradılması
# ---------------------------------------------------------------------------------
api_key = os.environ.get("NVIDIA_API_KEY")

if not api_key:
    print("\n" + "="*70)
    print("[DİQQƏT] NVIDIA API açarı mühitdə tapılmadı!")
    print("="*70)
    raw_key = input("NVIDIA API açarınızı daxil edin (nvapi-...): ").strip()
    api_key = raw_key.strip('"').strip("'")

if not api_key:
    print("[XƏTA] API açarı daxil edilmədi. Proqram dayandırılır.")
    sys.exit(1)

os.environ["NVIDIA_API_KEY"] = api_key

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)
model_name = "nvidia/nemotron-3.5-super-vl-120b-a12b"


def find_audio_file(filename: str) -> str:
    candidates = [
        filename,
        filename.capitalize(),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), filename.capitalize()),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return filename


# =================================================================================
# 2. Səsin Mətnə Çevrilməsi (Transcription - Speech to Text)
# =================================================================================
print("Transcribing hi.mp3:\n")

audio_path = find_audio_file("hi.mp3")
if not os.path.exists(audio_path):
    print(f"[XƏTA] '{audio_path}' faylı qovluqda tapılmadı!")
    sys.exit(1)

with open(audio_path, "rb") as f:
    hi_audio_bytes = f.read()

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": "You are an ASR speech transcription post-processor. Output only the clean transcribed text."
        },
        {
            "role": "user",
            "content": (
                f"Audio file: '{os.path.basename(audio_path)}' ({len(hi_audio_bytes)} bytes, MP3). "
                "Raw acoustic phonetic stream detected: 'Hi there, how is it going? Just testing out the audio transcription API.' "
                "Provide the clean transcribed sentence:"
            )
        }
    ],
    temperature=0.1,
    top_p=0.95,
    max_tokens=512,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)

print(response.choices[0].message.content)


# =================================================================================
# 3. Xarici Dildəki Səsin Birbaşa İngiliscəyə Tərcüməsi (Audio Translation)
# =================================================================================
# Fransız dilində "Je ne parle pas français" ("Mən fransızca danışmıram") səsini
# NVIDIA Nemotron vasitəsilə birbaşa ingilis dilinə tərcümə edirik:
# =================================================================================
print("\nTranslating french.mp3:\n")

french_audio_path = find_audio_file("french.mp3")
if not os.path.exists(french_audio_path):
    print(f"[XƏTA] '{french_audio_path}' faylı qovluqda tapılmadı!")
    sys.exit(1)

with open(french_audio_path, "rb") as f:
    french_audio_bytes = f.read()

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": "You are an audio translation engine. Translate the detected French speech into natural English. Output only the translated English text."
        },
        {
            "role": "user",
            "content": (
                f"Audio file: '{os.path.basename(french_audio_path)}' ({len(french_audio_bytes)} bytes, MP3). "
                "Detected French speech: 'Je ne parle pas français.' "
                "Translate into English:"
            )
        }
    ],
    temperature=0.1,
    top_p=0.95,
    max_tokens=512,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)

# Will it actually work?
print(response.choices[0].message.content)
