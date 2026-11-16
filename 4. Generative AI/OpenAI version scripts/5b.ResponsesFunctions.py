"""
===================================================================================
6. OpenAI Responses API ilə Function Calling (Alətlər / Tools) və Cavab Mexanizmi
===================================================================================

Bu skript Generativ Süni İntellekt modellərinin (OpenAI GPT-4o) xarici Python 
funksiyaları (Alətlər / Tools) ilə 3 addımlı əl ilə (manual) qarşılıqlı əlaqə 
prosesini ən müasir OpenAI "Responses API" (`client.responses.create`) ilə nümayiş etdirir.

Responses API ilə Function Calling Prosesinin 3 Əsas Addımı:
-----------------------------------------------------------
Step 1: Modelin İlkin Qərarı (İstifadəçi sualı və alət sxemi göndərilir, model funksiya çağırışı tələb edir).
Step 2: Funksiyanın İcrası (Modelin arqumentləri oxunur və yerli Python funksiyası icra edilir).
Step 3: Nəticənin Modelə Göndərilməsi və Yekun Cavab (Funksiyanın icra nəticəsi `function_call_output` olaraq
        göndərilir və model istifadəçiyə səlis yekun təbii cavab verir).
===================================================================================
"""

import os
import sys
import json
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
model_name = "gpt-4o"


# =================================================================================
# 2. Xarici Python Funksiyası (Dummy Weather Function)
# =================================================================================
# Real layihələrdə bu funksiya xarici hava xidmətinə (məs: OpenWeatherMap),
# daxili verilənlər bazasına və ya şirkətin API-larına sorğu göndərir.
# =================================================================================

def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Verilmiş şəhər və ya məkan üzrə cari hava durumunu əldə edir.

    Args:
        location: Şəhər və ştat / ölkə adı, məsələn: 'Orlando, Florida' və ya 'Baku'
        unit: Temperatur vahidi ('celsius' və ya 'fahrenheit')
    """
    print(f"\n" + "-"*60)
    print(f"[ALƏT İŞƏ DÜŞDÜ] Real Python funksiyası çağırıldı:")
    print(f" -> get_current_weather(location='{location}', unit='{unit}')")
    print("-"*60)

    weather_info = {
        "location": location,
        "temperature": "72" if unit.lower() == "fahrenheit" else "22",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info, ensure_ascii=False)


# =================================================================================
# 3. Alətin Təsviri və JSON Sxemi (Tools Definition)
# =================================================================================
# OpenAI Responses API formatında alət sxemi birbaşa top-level açarlarla təyin edilir:
tools = [{
    "type": "function",
    "name": "get_current_weather",
    "description": "Get the current weather for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA or Orlando, Florida"
            },
            "unit": {
                "type": "string", 
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit (celsius or fahrenheit)"
            }
        },
        "required": [
            "location"
        ],
        "additionalProperties": False
    }
}]

user_prompt = "What is the weather like in Orlando, Florida today?"
input_messages = [{"role": "user", "content": user_prompt}]


# =================================================================================
# Step 1: Modelin İlkin Qərarı (Initial result for weather in Orlando)
# =================================================================================
# İstifadəçi sualı və alətlər modelə göndərilir. Model başa düşür ki, suala cavab
# vermək üçün `get_current_weather` aləti çağırılmalıdır:
# =================================================================================
print("Step 1: Initial result for weather in Orlando:")

response = client.responses.create(
    model=model_name,
    input=input_messages,
    tools=tools
)

print(response.output)


# =================================================================================
# Step 2: Funksiyanın İcrası (Call the tool it wants)
# =================================================================================
# Modelin seçdiyi ilk alət çağırışı (`response.output[0]`) oxunur, onun arqumentləri
# JSON-dan çıxarılır və yerli Python funksiyası çağırılır:
# =================================================================================
print("\nStep 2: Call the tool it wants")

tool_call = response.output[0]
args = json.loads(tool_call.arguments)
print(f"Arguments: {args}")

# Modelin çıxardığı arqumentlərlə funksiyanı çağırırıq:
result = get_current_weather(
    location=args.get("location"),
    unit=args.get("unit", "fahrenheit")
)
print(result)


# =================================================================================
# Step 3: Nəticənin Modelə Göndərilməsi və Yekun Cavab (Incorporate results)
# =================================================================================
# 1. Modelin verdiyi funksiya çağırışı mesajı dialoq tarixinə əlavə edilir.
# 2. Funksiyamızın icra nəticəsi `function_call_output` olaraq əlavə edilir.
# 3. Model bütün dialoq kontekstini görüb son istifadəçi üçün səlis cavab formalaşdırır.
# =================================================================================
print("\nStep 3: Incorporate results into prompt")

# Modelin funksiya çağırışı obyektini dialoqa əlavə edirik:
input_messages.append(tool_call)

# Funksiyanın qaytardığı nəticəni müvafiq `call_id` ilə əlavə edirik:
input_messages.append({
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": str(result)
})

# Yekun cavab üçün modelə təkrar müraciət edirik:
response_2 = client.responses.create(
    model=model_name,
    input=input_messages,
    tools=tools,
)

print("\nFinal response:")
print(response_2.output_text)


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