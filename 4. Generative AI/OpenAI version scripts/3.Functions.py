"""
===================================================================================
2. OpenAI ilə Function Calling (Alətlər / Tools İstifadəsi) — Chat Completions API
===================================================================================

Bu skript Böyük Dil Modellərinin (LLM) ən güclü xüsusiyyətlərindən biri olan
"Function Calling" (Alətlərdən İstifadə / Tool Use) mexanizmini rəsmi OpenAI
Python SDK-sı (`from openai import OpenAI`) vasitəsilə addım-addım nümayiş etdirir.

Function Calling Nədir və Niyə Lazımdır?
----------------------------------------
Böyük dil modelləri (LLM) təlim olunduqları tarixdən sonrakı canlı məlumatları
(məsələn: hazırkı hava durumu, canlı valyuta məzənnələri, şirkətin daxili SQL
verilənlər bazası və s.) birbaşa bilmirlər və xarici sistemlərə birbaşa çıxışları yoxdur.

Function Calling bu problemi 4 addımlı dövrlə həll edir:
1. Alətlərin Təqdimatı (Tools Definition): Modelə icra edə biləcəyimiz Python funksiyasının
   adını, təsvirini və parametr sxemini (JSON Schema) göndəririk.
2. Modelin Qərarı (Function Call Request): Model sualı təhlil edir və anlayır ki,
   cavab vermək üçün bu funksiyanı çağırmaq lazımdır. Funksiyanın adını və tələb
   olunan arqumentləri qaytarır (`tool_calls`).
3. Yerli İcra (Local Execution): Proqramçı tərəfindən həmin Python funksiyası modelin
   seçdiyi arqumentlərlə yerli mühitdə çağırılır və nəticə əldə edilir.
4. Yekun Sintez (Final Answer): Funksiyanın nəticəsi dialoq tarixinə (`role: "tool"`)
   əlavə edilərək modelə göndərilir və model istifadəçiyə səlis, təbii dildə yekun cavab verir.
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

# İstifadə ediləcək OpenAI modeli (gpt-3.5-turbo, gpt-4o-mini və ya gpt-4o):
MODEL_NAME = "gpt-3.5-turbo"


# =================================================================================
# 2. Xarici və ya Simulyasiya Edilən Python Funksiyası (Dummy Weather Function)
# =================================================================================
# Real layihələrdə bu funksiya daxilində canlı API-lara (məs: OpenWeatherMap),
# SQL verilənlər bazasına və ya daxili CRM sisteminə müraciət edilir.
# =================================================================================

def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Verilmiş şəhər və ya məkan üzrə cari hava durumunu əldə edir.

    Args:
        location: Şəhər və ştat / ölkə adı, məsələn: 'San Francisco, CA' və ya 'Baku'
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
        "forecast": ["günəşli", "mülayim küləkli"],
    }
    return json.dumps(weather_info, ensure_ascii=False)


# =================================================================================
# 3. Dialoqun və Function Calling Prosesinin İdarə Edilməsi
# =================================================================================

def run_conversation(prompt: str) -> str:
    """
    İstifadəçi sualını və əlçatan alətləri (tools) OpenAI modelinə göndərir.
    Model alət tələb etdikdə addım-addım icra edib yekun cavabı qaytarır.
    """
    print("\n" + "="*70)
    print(f"İstifadəçi Sorğusu: \"{prompt}\"")
    print("="*70)

    # -----------------------------------------------------------------------------
    # ADDIM 1: Alətlərin (Tools) Təsviri və İstifadəçi Mesajının Göndərilməsi
    # -----------------------------------------------------------------------------
    # OpenAI-a hansı funksiyaların mövcud olduğunu JSON Schema vasitəsilə bildiririk.
    # Model bu sxemə baxaraq hansı arqumentləri çıxarmalı olduğunu anlayır:
    messages = [{"role": "user", "content": prompt}]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location (Müəyyən məkan üzrə cari hava durumunu əldə edir)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA or Baku, Azerbaijan",
                        },
                        "unit": {
                            "type": "string", 
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit (celsius or fahrenheit)"
                        },
                    },
                    "required": ["location"],
                },
            }
        }
    ]

    print("\n[ADDIM 1] Sual və Alətlər OpenAI modelinə göndərilir...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto"  # Model alətdən istifadə edib-etməməyə özü qərar verir
    )
    
    response_message = response.choices[0].message

    print("\n[İLKİN CAVAB] Modelin verdiyi ilkin reaksiya:")
    print(response_message)

    # -----------------------------------------------------------------------------
    # ADDIM 2: Modelin Funksiya Çağırmaq İstədiyinin Yoxlanılması
    # -----------------------------------------------------------------------------
    if response_message.tool_calls:
        print("\n" + "="*70)
        print("[ADDIM 2] Model qərar verdi: Bu suala cavab üçün alət çağırılmalıdır!")
        print("="*70)

        # -------------------------------------------------------------------------
        # ADDIM 3: Tələb Olunan Funksiyanın Yerli Python Mühitində İcrası
        # -------------------------------------------------------------------------
        available_functions = {
            "get_current_weather": get_current_weather,
        }

        # Modelin seçdiyi ilk alət çağırışı:
        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        
        # Modelin generasiya etdiyi JSON arqumentləri oxunur:
        function_args = json.loads(tool_call.function.arguments)
        print(f"\n[ADDIM 3] Modelin çıxardığı arqumentlər:")
        print(f" - Funksiya adı: {function_name}")
        print(f" - Arqumentlər: {function_args}")

        # Python funksiyasını çağırırıq:
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit") or "celsius",
        )

        print("\nFunksiyanın qaytardığı nəticə (JSON):")
        print(function_response)

        # -------------------------------------------------------------------------
        # ADDIM 4: Nəticənin Dialoqa Əlavə Edilməsi və Modelin Yekun Cavabı
        # -------------------------------------------------------------------------
        print("\n[ADDIM 4] Funksiya cavabı dialoqa ('role: tool') əlavə olunur və modelə göndərilir...")

        # 1. Modelin funksiya çağırışı mesajını dialoq tarixinə əlavə edirik:
        messages.append(response_message)

        # 2. Funksiyanın icra nəticəsini müvafiq 'tool_call_id' ilə dialoqa əlavə edirik:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": function_response,
            }
        )

        # 3. Modelə bütün dialoq tarixçəsini göndərərək yekun insan dilində cavab alırıq:
        second_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )

        final_text = second_response.choices[0].message.content
        return final_text
    else:
        # Əgər istifadəçi hava ilə bağlı olmayan adi sual veribsə:
        print("\n[MƏLUMAT] Sual üçün heç bir alətə ehtiyac duyulmadı.")
        return response_message.content


# =================================================================================
# 4. Proqramın İcrası (İstifadəçi ilə Dialoq)
# =================================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("OpenAI Chat Completions Function Calling Nümunəsi İşə Salındı.")
    print("="*70)
    
    prompt = input("\nHava haqqında bir sual verin (məs: Orlando və ya Bakıda hava necədir?): ")
    if not prompt.strip():
        prompt = "What is the weather like in Orlando, Florida today?"

    final_answer = run_conversation(prompt)

    print("\n" + "="*70)
    print("YEKUN CAVAB (Modelin Təbii Dildə İzahı):")
    print("="*70)
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
   - "How is the weather in Miami, Florida in celsius?"

2. Dolayı və məntiqi suallar (Model yenə də alətə müraciət edir):
   - "Sabah San Fransiskoya gedirəm, çətir götürməliyəm?"
   - "Orlando üçün isti geyinməyə ehtiyac var?"

3. Alət tələb etməyən adi suallar (Model funksiyanı çağırmadan birbaşa cavab verir):
   - "Salam, kimsən və nə bacarırsan?"
   - "Kvant kompüterləri nədir?"
===================================================================================
"""
