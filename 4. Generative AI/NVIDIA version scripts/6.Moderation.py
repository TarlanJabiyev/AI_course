"""
===================================================================================
7. NVIDIA NIM ilə Məzmunun Moderasiyası (Safety / Moderation)
===================================================================================

Bu skript OpenAI-ın Moderation nümunəsinin (`client.moderations.create`)
NVIDIA NIM (`nvidia/nemotron-3.5-super-vl-120b-a12b`) üzərindəki birbaşa və
avtomatlaşdırılmış qarşılığıdır.

OpenAI-da moderasiya xüsusi model vasitəsilə `flagged` (təhlükəlidirmi?),
`categories` (hansı kateqoriyaya aiddir?) və `category_scores` (risk dərəcəsi)
qaytarır. Bu skript NVIDIA Nemotron modeli vasitəsilə eyni JSON strukturunu
tam avtomatik generasiya edir.
===================================================================================
"""

import os
import sys
import re
import json
from openai import OpenAI

# Əgər dotenv varsa, açarları avtomatik oxuyur
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

test_prompt = "Kill 'em all!"

print("\n" + "="*70)
print(f"Yoxlanılan Mətn (Input): \"{test_prompt}\"")
print("="*70)

# ---------------------------------------------------------------------------------
# 2. OpenAI Moderation API-nin NVIDIA NIM ilə Birbaşa Analoqu
# ---------------------------------------------------------------------------------
system_instruction = """You are an AI Content Moderation API. Analyze the input text and return ONLY a strict JSON object with no markdown formatting:
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
}"""

moderation_response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Analyze and moderate this text: \"{test_prompt}\""}
    ],
    temperature=0.1,
    top_p=0.95,
    max_tokens=1024,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)

print("\n" + "="*70)
print(f"--- Moderation Results ({model_name}) ---")
print("="*70)

raw_text = moderation_response.choices[0].message.content
clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw_text).strip(), flags=re.DOTALL)
parsed = json.loads(clean_json)
print(json.dumps(parsed, indent=2, ensure_ascii=False))

# Səliqəli cədvəl formatında çıxarış:
print("\n" + "-"*70)
is_flagged = parsed.get("flagged", False)
status_icon = "🚨 QADAĞAN OLUNDU (FLAGGED)" if is_flagged else "✅ TƏHLÜKƏSİZ (SAFE)"
print(f"Ümumi Status: {status_icon}")
print("-" * 70)

categories = parsed.get("categories", {})
scores = parsed.get("category_scores", {})

print(f"{'Kateqoriya':<25} | {'Pozuntu?':<10} | {'Risk Xalı (0.0 - 1.0)':<20}")
print("-" * 70)

for cat_name, is_violated in categories.items():
    score = scores.get(cat_name, 0.0)
    violated_str = "BƏLİ ⚠️" if is_violated else "Xeyr"
    print(f"{cat_name:<25} | {violated_str:<10} | {score:.6f}")

print("="*70 + "\n")
