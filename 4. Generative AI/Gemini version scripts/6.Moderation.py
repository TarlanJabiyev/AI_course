"""
===================================================================================
7. Google Gemini ilə Məzmunun Moderasiyası (Safety / Moderation)
===================================================================================

Bu skript OpenAI-ın Moderation nümunəsinin (`client.moderations.create`)
Google Gemini-dəki birbaşa və avtomatlaşdırılmış qarşılığıdır.

OpenAI-da moderasiya xüsusi model vasitəsilə `flagged` (təhlükəlidirmi?),
`categories` (hansı kateqoriyaya aiddir?) və `category_scores` (risk dərəcəsi)
qaytarır. Bu skript Gemini-nin `response_mime_type="application/json"`
imkanından istifadə edərək eyni nəticəni tam avtomatik generasiya edir.
===================================================================================
"""

import os
import sys
import json
from google import genai
from google.genai import types

# Əgər dotenv varsa, açarları avtomatik oxuyur
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------------
# Google Gemini API Açarının Əldə Edilməsi və Client Yaradılması
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

test_prompt = "Kill 'em all!"

print("="*70)
print(f"Yoxlanılan Mətn (Input): \"{test_prompt}\"")
print("="*70)

# ---------------------------------------------------------------------------------
# 2. OpenAI Moderation API-nin Gemini ilə Birbaşa Analoqu
# ---------------------------------------------------------------------------------
# OpenAI-dakı `client.moderations.create` nəticəsini (flagged, categories,
# category_scores) Gemini vasitəsilə birbaşa əldə edirik:
moderation_response = client.models.generate_content(
    model=model_name,
    contents=f"Analyze and moderate this text: \"{test_prompt}\"",
    config=types.GenerateContentConfig(
        system_instruction="""You are an AI Content Moderation API. Analyze the input text and return a strict JSON object:
{
  "flagged": true/false,
  "categories": {
    "hate": true/false,
    "harassment": true/false,
    "self-harm": true/false,
    "sexual": true/false,
    "violence": true/false
  },
  "category_scores": {
    "hate": 0.0 to 1.0,
    "harassment": 0.0 to 1.0,
    "self-harm": 0.0 to 1.0,
    "sexual": 0.0 to 1.0,
    "violence": 0.0 to 1.0
  }
}""",
        response_mime_type="application/json"
    )
)

print("\n--- Moderation Results ---")
parsed = json.loads(moderation_response.text)
print(json.dumps(parsed, indent=2, ensure_ascii=False))

# Əlavə: Model tərəfindən daxili safety_ratings qaytarılarsa, onu da təhlükəsiz göstəririk:
if moderation_response.candidates and moderation_response.candidates[0].safety_ratings:
    print("\n--- Daxili Safety Ratings ---")
    for rating in moderation_response.candidates[0].safety_ratings:
        cat = str(rating.category).replace("HarmCategory.", "").replace("HARM_CATEGORY_", "")
        prob = str(rating.probability).replace("HarmProbability.", "")
        print(f" - {cat:<22} : {prob}")
