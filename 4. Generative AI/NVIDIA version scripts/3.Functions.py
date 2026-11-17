"""
===================================================================================
2. NVIDIA NIM ilə Function Calling (Alətlər / Tools İstifadəsi)
===================================================================================

Bu skript Böyük Dil Modellərinin (LLM) ən güclü xüsusiyyətlərindən biri olan
"Function Calling" (və ya Tool Use — Alətlərin Çağırılması) mexanizmini NVIDIA NIM
(`nvidia/nemotron-3.5-super-vl-120b-a12b`) modeli vasitəsilə nümayiş etdirir.

Function Calling Nədir və Niyə Lazımdır?
----------------------------------------
Dil modelləri (LLM) təlim olunduqları tarixdən sonrakı canlı məlumatları (məsələn:
hazırkı hava durumu, canlı valyuta məzənnəsi, verilənlər bazasındakı cari qalıq)
bilmirlər və birbaşa xarici sistemlərə çıxışları yoxdur.

NVIDIA Nemotron-3.5 ilə Function Calling:
1. Biz modelə Python funksiyamızın JSON Schema təsvirini (`tools`) və `parallel_tool_calls=True`
   parametrini təqdim edirik.
2. Model istifadəçinin sualını təhlil edir və başa düşür ki, bu suala cavab vermək üçün
   funksiya çağırılmalıdır (`tool_calls`).
3. Funksiya yerli mühitdə icra edilir, nəticəsi modelə qaytarılır və model istifadəçiyə
   səlis, təbii yekun cavab verir.
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
# 1. NVIDIA API Açarının Əldə Edilməsi və Client Yaradılması
# ---------------------------------------------------------------------------------
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
MODEL_NAME = "nvidia/nemotron-3.5-super-vl-120b-a12b"


# =================================================================================
# 2. Real və ya Simulyasiya Edilən Python Funksiyası (Tool / Alət)
# =================================================================================
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Verilmiş şəhər və ya məkan üzrə cari hava durumunu əldə edir.

    Args:
        location: Şəhər və ya bölgə adı, məsələn: Bakı, London, İstanbul
        unit: Temperatur vahidi (celsius və ya fahrenheit)
    """
    print(f"\n" + "-"*50)
    print(f"[ALƏT İŞƏ DÜŞDÜ] Real Python funksiyası çağırıldı:")
    print(f" -> get_current_weather(location='{location}', unit='{unit}')")
    print("-"*50)

    # Real layihələrdə burada xarici hava xidmətinə (məs: OpenWeatherMap API) sorğu göndərilir.
    weather_info = {
        "location": location,
        "temperature": "22" if unit.lower() == "celsius" else "72",
        "unit": unit,
        "forecast": ["günəşli", "mülayim küləkli"],
    }
    return json.dumps(weather_info, ensure_ascii=False)


# =================================================================================
# 3. Dialoqun və Function Calling Prosesinin İdarə Edilməsi
# =================================================================================

def run_conversation(prompt: str) -> str:
    """
    İstifadəçi sualını və əlçatan alətləri (tools) NVIDIA Nemotron modelinə göndərir.
    Əgər model alətə ehtiyac duyarsa, funksiyanı icra edib yekun təbii cavabı qaytarır.
    """
    print("\n" + "="*60)
    print("[ADDIM 1] Sual və Alət (get_current_weather) NVIDIA NIM-ə göndərilir...")
    print("="*60)

    messages = [{"role": "user", "content": prompt}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Verilmiş şəhər və ya məkan üzrə cari hava durumunu əldə edir.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Şəhər və ya bölgə adı, məsələn: Bakı, London, İstanbul, Orlando",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperatur vahidi (celsius və ya fahrenheit)",
                        },
                    },
                    "required": ["location"],
                },
            }
        }
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        parallel_tool_calls=True,
        temperature=0.2,
        top_p=0.95,
        max_tokens=4096,
        seed=42,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        stream=False
    )

    response_message = response.choices[0].message

    if response_message.tool_calls:
        available_functions = {
            "get_current_weather": get_current_weather,
        }

        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            function_response = function_to_call(
                location=function_args.get("location", "Bakı"),
                unit=function_args.get("unit") or "celsius",
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response,
                }
            )

        second_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            top_p=0.95,
            max_tokens=4096,
            seed=42,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            stream=False
        )
        return second_response.choices[0].message.content

    return response_message.content


# =================================================================================
# 4. Proqramın İcrası (İstifadəçi ilə Əlaqə)
# =================================================================================
if __name__ == "__main__":
    print("\nNVIDIA NIM Function Calling Nümunəsi İşə Salındı.")
    prompt = input("Hava haqqında bir sual verin (məs: Bakıda hava necədir?): ")
    if not prompt.strip():
        prompt = "Bakıda hava necədir?"

    final_answer = run_conversation(prompt)

    print("\n" + "="*60)
    print("YEKUN CAVAB (Modelin Təbii Dildə İzahı):")
    print("="*60)
    print(final_answer)


"""
===================================================================================
TEST ÜÇÜN ÖRNƏK PROMPTLAR (SUALLAR):
===================================================================================

Aşağıdakı sualları skriptdə yoxlayaraq Function Calling-in necə işlədiyini müşahidə edə bilərsiniz:

1. Funksiyanı dərhal işə salan birbaşa suallar:
   - "Bakıda hava necədir?"
   - "İstanbulda hazırda hava durumu necədir?"
   - "What's the weather like in London?"
   - "Nyu-Yorkda hava fahrenheit ilə neçə dərəcədir?"

2. Dolayı və məntiqi suallar (Model yenə də funksiyanı çağıracaq):
   - "Sabah Londona uçuram, özümlə çətir götürüm?"
   - "Bakıda gəzməyə çıxmaq istəyirəm, qalın geyinim?"

3. Funksiyanı ÇAĞIRMAYAN adi suallar (Model alətə ehtiyac olmadığını anlayır):
   - "Salam, necəsən?"
   - "Günəş sistemində neçə planet var?"
   - "Python-da funksiya necə yaradılır?"
===================================================================================
"""
