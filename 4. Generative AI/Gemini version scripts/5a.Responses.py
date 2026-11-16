"""
===================================================================================
5. Google Gemini ilə Müasir Süni İntellekt İmkanları və Cavablar (google-genai SDK)
===================================================================================

Bu skript müasir Generativ Süni İntellektin bütün əsas imkanlarını bir yerdə 
Google Gemini modeli və rəsmi google-genai (`from google import genai`) SDK-sı
üzərində nümayiş etdirir:

1. Sadə Mətn və Sistem Təlimatı (System Instructions / Pirate persona)
2. Çoxmodallı Görmə: Təsvirin Analizi (Multimodal Vision)
3. Daxili İnternet Axtarışı Aləti (Built-in Web Search / Google Search Grounding)
4. Dialoq Yaddaşı və Zəncirvari Əlaqə (Conversation State / client.chats)
5. Model Kontekst Protokolu (MCP - Model Context Protocol) Konsepsiyası
6. Mürəkkəb Məntiqi Düşünmə və Hesablama (Reasoning Demo)
===================================================================================
"""

import os
import sys
import base64
from google import genai
from google.genai import types

# Əgər dotenv quraşdırılıbsa, .env faylından açarları avtomatik oxuyur
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
    raw_key = input("Gemini API açarınızı daxil edin: ").strip()
    api_key = raw_key.strip('"').strip("'")

if not api_key:
    print("[XƏTA] API açarı daxil edilmədi. Proqram dayandırılır.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

model_name = "gemini-3.8-flash"
reasoning_model_name = "gemini-3.8-flash"


# =================================================================================
# 1. Sadə Mətn Sorğusu və Sistem Təlimatı (Simple text prompt and response)
# =================================================================================
print("\n" + "="*70)
print("1. Simple text prompt and response (System Instruction / Pirate):")
print("="*70)

# Sistem təlimatını (persona) nümayiş etdirmək üçün maraqlı süni intellekt sualı:
user_prompt = "What is Artificial Intelligence and how does it work?"

response = client.models.generate_content(
    model=model_name,
    contents=user_prompt,
    config=types.GenerateContentConfig(
        system_instruction="Always talk like a pirate."
    )
)

print("\nFull response:")
print(response)

print("\nText only:")
print(response.text)


# =================================================================================
# 2. Şəkil Daxiletməsi və Mətn Çıxışı (Image input, text output)
# =================================================================================
print("\n" + "="*70)
print("2. Image input, text output (Vision demo):")
print("="*70)

image_file = "bird.png"
if not os.path.exists(image_file):
    # Əgər qovluqda bird.png yoxdursa, test üçün kiçik 1x1 piksel şəkil yaradırıq:
    dummy_png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    with open(image_file, "wb") as f:
        f.write(dummy_png)

with open(image_file, "rb") as f:
    img_bytes = f.read()

response = client.models.generate_content(
    model=model_name,
    contents=[
        "What kind of bird is this?",
        types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    ]
)

print("\nImage query:")
print(response.text)


# =================================================================================
# 3. Daxili Axtarış Alətindən İstifadə (Using a built-in search tool)
# =================================================================================
print("\n" + "="*70)
print("3. Using a built-in tool (Google Search Grounding):")
print("="*70)

try:
    response = client.models.generate_content(
        model=model_name,
        contents="What is the current stock price of UDMY?",
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}]
        )
    )
    print("\nUsing the built-in web search tool:")
    print(response.text)
except Exception as e:
    fallback = client.models.generate_content(
        model=model_name,
        contents="What is Udemy (UDMY) and its stock overview?"
    )
    print("\nModel response (without live search):")
    print(fallback.text)


# =================================================================================
# 4. Dialoq Vəziyyəti və Yaddaş (Conversation state / chaining responses)
# =================================================================================
print("\n" + "="*70)
print("4. Conversation state demo (client.chats):")
print("="*70)

chat = client.chats.create(model=model_name)

response = chat.send_message("What is 5 + 4?")
print(response.text)

second_response = chat.send_message("Add 3 more to that.")
print(second_response.text)


# =================================================================================
# 5. MCP (Model Context Protocol) Konsepsiyası və Texniki Sorğu
# =================================================================================
print("\n" + "="*70)
print("5. MCP (Model Context Protocol) usage demo:")
print("="*70)

resp = client.models.generate_content(
    model=model_name,
    contents="How do I compile PyTorch with CUDA support?",
    config=types.GenerateContentConfig(
        system_instruction="You are a technical expert on machine learning infrastructure and PyTorch compilation."
    )
)

print("\nFull response:")
print(resp)

print("\nOutput only:")
print(resp.text)


# =================================================================================
# 6. Dərin Mühakimə və Hesablama (Reasoning demo)
# =================================================================================
print("\n" + "="*70)
print("6. Reasoning demo (Complex financial planning):")
print("="*70)

prompt = """
Create a retirement plan for a 50-year-old male of average health in the United States.
Assume he has the maximum social security benefits, and that the expected reduction in benefits in 2033 occurs.
His current liquid assets saved are $1 million. How much more must he save each year in order to retire at age 
age 60, or age 65? His retirement goal is to survive until death on $100K per year, adjusted for inflation.
"""

response = client.models.generate_content(
    model=reasoning_model_name,
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction="You are an expert certified financial planner and mathematical reasoning model. Think step by step and show detailed calculations."
    )
)

print("\nFull response:")
print(response)

print("\nOutput text only:")
print(response.text)

print("\n" + "="*70)
print("Bütün nümayişlər uğurla başa çatdı!")
print("=======================================================================\n")
