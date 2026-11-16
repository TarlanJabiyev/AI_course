"""
===================================================================================
6. Google Gemini ilə Function Calling (Alətlər / Tools) və Cavab Mexanizmi
===================================================================================

Bu skript Generativ Süni İntellekt modellərinin (Google Gemini) xarici Python 
funksiyaları (Alətlər / Tools) ilə 3 addımlı əl ilə (manual) qarşılıqlı əlaqə 
prosesini rəsmi google-genai (`from google import genai`) SDK-sı ilə nümayiş etdirir.

Function Calling Prosesinin 3 Əsas Addımı:
------------------------------------------
Step 1: Modelin İlkin Qərarı (İstifadəçi sualı verilir, model funksiya çağırışı tələb edir).
Step 2: Funksiyanın İcrası (Modelin arqumentləri oxunur və yerli Python funksiyası icra edilir).
Step 3: Nəticənin Modelə Göndərilməsi və Yekun Cavab (Funksiyanın nəticəsi ilə model yekun cavabı verir).
===================================================================================
"""

import os
import sys
import json
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
    raw_key = input("Gemini API açarınızı daxil edin (AIzaSy...): ").strip()
    api_key = raw_key.strip('"').strip("'")

if not api_key:
    print("[XƏTA] API açarı daxil edilmədi. Proqram dayandırılır.")
    sys.exit(1)

client = genai.Client(api_key=api_key)
model_name = "gemini-3.8-flash"


# =================================================================================
# Xarici Python Funksiyası (Dummy Weather Function)
# =================================================================================
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a given location.

    Args:
        location: The city and state, e.g. Orlando, Florida or San Francisco, CA
        unit: Temperature unit (celsius or fahrenheit)
    """
    weather_info = {
        "location": location,
        "temperature": "22" if unit.lower() == "celsius" else "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


# =================================================================================
# Alətin Təsviri (Tool / Function Declaration)
# =================================================================================
# OpenAI-dakı `tools = [{"type": "function", ...}]` kimi, Google Gemini-də
# əl ilə (manual) 3 addımlı icra prosesi üçün funksiyanın deklarasiyasını veririk.
#
# DİQQƏT: Əgər `tools=[get_current_weather]` şəklində birbaşa Python funksiyasını
# versəniz, SDK avtomatik çağırış (AFC) edərək funksiyanı dərhal arxa planda icra
# edəcək və `response.function_calls` boş (None) olacaq. Əl ilə hər addımı görmək
# üçün funksiyanı sxem (FunctionDeclaration) kimi təyin edirik:
weather_declaration = types.FunctionDeclaration(
    name="get_current_weather",
    description="Get the current weather for a given location.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "location": types.Schema(
                type=types.Type.STRING,
                description="The city and state, e.g. Orlando, Florida or San Francisco, CA",
            ),
            "unit": types.Schema(
                type=types.Type.STRING,
                description="Temperature unit (celsius or fahrenheit)",
                enum=["celsius", "fahrenheit"],
            ),
        },
        required=["location"],
    ),
)

weather_tool = types.Tool(function_declarations=[weather_declaration])

# Alətlər konfiqurasiyası:
config = types.GenerateContentConfig(
    tools=[weather_tool],
    temperature=0.0
)

user_prompt = "What is the weather like in Orlando, Florida today?"


# =================================================================================
# Step 1: Initial result for weather in Orlando
# =================================================================================
print("Step 1: Initial result for weather in Orlando:")

response = client.models.generate_content(
    model=model_name,
    contents=user_prompt,
    config=config
)

# Təhlükəsiz yoxlama: Modelin funksiya çağırışı tələb etdiyini yoxlayırıq
if not response.function_calls:
    print("[MƏLUMAT] Model funksiya çağırmadı, birbaşa cavab verdi:")
    print(response.text)
    sys.exit(0)

tool_call = response.function_calls[0]
print(tool_call)


# =================================================================================
# Step 2: Call the tool it wants
# =================================================================================
print("\nStep 2: Call the tool it wants")

args = tool_call.args
print(f"Arguments: {args}")

# Modelin tələb etdiyi arqumentlərlə Python funksiyamızı çağırırıq:
result = get_current_weather(**args)
print(result)


# =================================================================================
# Step 3: Incorporate results into prompt
# =================================================================================
print("\nStep 3: Incorporate results into prompt")

# JSON nəticəni model üçün lüğət (dict) formatına gətiririk:
try:
    result_dict = json.loads(result) if isinstance(result, str) else result
except Exception:
    result_dict = {"output": result}

# OpenAI-da funksiya cavabı üçün `role="tool"` istifadə olunur.
# Lakin Google Gemini API arxitekturasında funksiya cavabları müştəri (client)
# tərəfindən göndərildiyi üçün `role="user"` qəbul edilir:
tool_response_content = types.Content(
    role="user",
    parts=[
        types.Part.from_function_response(
            name=tool_call.name,
            response={"result": result_dict}
        )
    ]
)

response_2 = client.models.generate_content(
    model=model_name,
    contents=[
        types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]),
        response.candidates[0].content,
        tool_response_content
    ],
    config=config
)

print(response_2.text)


"""
===================================================================================
TEST ÜÇÜN DİGƏR MARAQLI ÖRNƏK SUALLAR:
===================================================================================

Skriptdəki `user_prompt` dəyərini aşağıdakı suallarla dəyişib sınaqdan keçirə bilərsiniz:

1. Fərqli şəhər və temperatur vahidi:
   user_prompt = "How is the weather in Miami, Florida in celsius?"
   (Model avtomatik unit='celsius' və location='Miami, Florida' arqumentlərini çıxaracaq)

2. Azərbaycan dilində sual:
   user_prompt = "Orlando şəhərində bu gün hava necədir?"
   (Model sualın dilindən asılı olmayaraq arqumentləri düzgün çıxaracaq və cavabı Azərbaycan dilində verəcək)

3. Geyim / Çətir məsləhəti (Dolayı məntiqi sual):
   user_prompt = "I am traveling to Orlando today, should I take a jacket or umbrella?"
   (Model əvvəlcə havanı yoxlayacaq, sonra küləkli və 72 dərəcə olduğunu görüb müvafiq məsləhət verəcək)
===================================================================================
"""
