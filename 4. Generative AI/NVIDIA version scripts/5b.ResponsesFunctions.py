"""
===================================================================================
6. NVIDIA NIM ilə Function Calling (Alətlər / Tools) və Cavab Mexanizmi
===================================================================================

Bu skript Generativ Süni İntellekt modellərinin (`nvidia/nemotron-3.5-super-vl-120b-a12b`)
xarici Python funksiyaları (Alətlər / Tools) ilə 3 addımlı əl ilə (manual) qarşılıqlı
əlaqə prosesini NVIDIA NIM (`https://integrate.api.nvidia.com/v1`) ilə nümayiş etdirir.

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
from openai import OpenAI

# Əgər dotenv quraşdırılıbsa, .env faylından açarları avtomatik oxuyur
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------------
# NVIDIA API Açarının Əldə Edilməsi və Client Yaradılması
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
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Orlando, Florida or San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Temperature unit (celsius or fahrenheit)",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        }
    }
]

user_prompt = "What is the weather like in Orlando, Florida today?"
messages = [{"role": "user", "content": user_prompt}]


# =================================================================================
# Step 1: Initial result for weather in Orlando
# =================================================================================
print("Step 1: Initial result for weather in Orlando:")

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    parallel_tool_calls=True,
    temperature=0.2,
    top_p=0.95,
    max_tokens=2048,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    stream=False
)

response_message = response.choices[0].message

if not response_message.tool_calls:
    print("[MƏLUMAT] Model funksiya çağırmadı, birbaşa cavab verdi:")
    print(response_message.content)
    sys.exit(0)

tool_call = response_message.tool_calls[0]
print(tool_call)


# =================================================================================
# Step 2: Call the tool it wants
# =================================================================================
print("\nStep 2: Call the tool it wants")

args = json.loads(tool_call.function.arguments)
print(f"Arguments: {args}")

# Modelin tələb etdiyi arqumentlərlə Python funksiyamızı çağırırıq:
result = get_current_weather(
    location=args.get("location", "Orlando, Florida"),
    unit=args.get("unit") or "celsius"
)
print(result)


# =================================================================================
# Step 3: Incorporate results into prompt
# =================================================================================
print("\nStep 3: Incorporate results into prompt")

messages.append(response_message)
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "name": tool_call.function.name,
    "content": str(result)
})

response_2 = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    temperature=0.2,
    top_p=0.95,
    max_tokens=2048,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)

print(response_2.choices[0].message.content)


"""
===================================================================================
TEST ÜÇÜN DİGƏR MARAQLI ÖRNƏK SUALLAR:
===================================================================================

Skriptdəki `user_prompt` dəyərini aşağıdakı suallarla dəyişib sınaqdan keçirə bilərsiniz:

1. Fərqli şəhər və temperatur vahidi:
   user_prompt = "How is the weather in Miami, Florida in celsius?"

2. Azərbaycan dilində sual:
   user_prompt = "Orlando şəhərində bu gün hava necədir?"

3. Geyim / Çətir məsləhəti (Dolayı məntiqi sual):
   user_prompt = "I am traveling to Orlando today, should I take a jacket or umbrella?"
===================================================================================
"""
