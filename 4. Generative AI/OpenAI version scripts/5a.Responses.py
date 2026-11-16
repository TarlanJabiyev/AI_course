"""
===================================================================================
5. OpenAI Responses API ilə Müasir Süni İntellekt İmkanları və Cavablar
===================================================================================

Bu skript OpenAI-ın ən müasir "Responses API" (`client.responses.create`) mexanizmini
və onun təqdim etdiyi bütün qabaqcıl imkanları rəsmi `openai` SDK-sı vasitəsilə nümayiş etdirir:

Responses API Nədir və Chat Completions-dan Fərqi Nədir?
-------------------------------------------------------
Klassik `client.chat.completions.create` metodundan fərqli olaraq, yeni Responses API
müasir süni intellekt tətbiqləri üçün bütün qabaqcıl imkanları tək vahid interfeysdə birləşdirir:

1. Sadə Mətn və Sistem Təlimatı (Instructions / Pirate persona)
2. Çoxmodallı Görmə (Multimodal Vision — Şəkillərin Fayl ID ilə analizi)
3. Daxili İnternet Axtarışı (Built-in Web Search — `web_search_preview`)
4. Server Tərəfli Dialoq Yaddaşı (Conversation State — `previous_response_id` ilə zəncirləmə)
5. Model Kontekst Protokolu (MCP - Model Context Protocol inteqrasiyası)
6. Dərin Mühakimə Modelləri (Reasoning Models — `o4-mini`, `o3-mini`, `o1`)
===================================================================================
"""

import os
import sys
import base64
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

# Əsas universal model və dərin mühakimə (reasoning) modeli:
model_name = "gpt-4o"
reasoning_model_name = "o4-mini"


# =================================================================================
# 1. Sadə Mətn Sorğusu və Sistem Təlimatı (Simple text prompt and response)
# =================================================================================
# `instructions` parametri modelin davranış tərzini (personas) müəyyənləşdirir.
# Burada modelə hər zaman bir pirat kimi danışmaq tapşırılır.
# =================================================================================
print("\n" + "="*70)
print("1. Simple text prompt and response (System Instruction / Pirate Persona):")
print("="*70)

# Sistem təlimatını nümayiş etdirmək üçün maraqlı süni intellekt sualı:
user_prompt = "What is Artificial Intelligence and how does it work?"

response = client.responses.create(
    model=model_name,
    input=user_prompt,
    instructions="Always talk like a pirate."
)

print("\nFull response (Tam JSON metadata obyekti):")
parsed = response.to_dict()
print(json.dumps(parsed, indent=2, ensure_ascii=False))

print("\nText only (Yalnız modelin cavab mətni):")
print(response.output[0].content[0].text)


# =================================================================================
# 2. Şəkil Daxiletməsi və Mətn Çıxışı (Multimodal Vision)
# =================================================================================
# OpenAI Responses API-da şəkil əvvəlcə Files API vasitəsilə buluda yüklənir
# və əldə olunan unikal `file_id` birbaşa sorğunun `input` parametrində istifadə edilir.
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

# Şəkli OpenAI Files API-ya 'vision' məqsədilə yükləyirik:
with open(image_file, "rb") as img_f:
    file_response = client.files.create(file=img_f, purpose="vision")

response = client.responses.create(
    model=model_name,
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What kind of bird is this?"},
            {
                "type": "input_image",
                "file_id": file_response.id,
            },
        ],
    }],
)

print("\nImage query (Şəkil üzrə təhlil cavabı):")
print(response.output_text)


# =================================================================================
# 3. Daxili İnternet Axtarışı Aləti (Using a built-in search tool)
# =================================================================================
# Responses API daxili `web_search_preview` aləti ilə təchiz olunub. Model cari
# internet məlumatlarını (məsələn: canlı səhm qiymətlərini) canlı vebdən axtarır.
# =================================================================================
print("\n" + "="*70)
print("3. Using a built-in tool (Web Search Preview):")
print("="*70)

try:
    response = client.responses.create(
        model=model_name,
        tools=[{"type": "web_search_preview"}],
        input="What is the current stock price of UDMY?"
    )
    print("\nUsing the built-in web search tool (Canlı veb axtarış cavabı):")
    print(response.output_text)
except Exception as e:
    print(f"\n[QEYD] Daxili veb axtarış sorğusunda xəta baş verdi: {e}")
    # Ehtiyat olaraq daxili biliklə cavab almaq:
    fallback = client.responses.create(
        model=model_name,
        input="What is Udemy (UDMY) and what is its stock ticker overview?"
    )
    print("\nModel cavabı (Axtarışsız):")
    print(fallback.output_text)


# =================================================================================
# 4. Dialoq Vəziyyəti və Server Yaddaşı (Conversation state / chaining responses)
# =================================================================================
# Responses API-nin ən böyük inqilabi yeniliyi:
# Əvvəllər bütün keçmiş mesajları hər dəfə təkrar göndərmək lazım idi.
# İndi isə sadəcə `previous_response_id=response.id` ötürməklə OpenAI serveri
# dialoq tarixçəsini və konteksti avtomatik zəncirvari şəkildə davam etdirir!
# =================================================================================
print("\n" + "="*70)
print("4. Conversation state demo (previous_response_id ilə zəncirləmə):")
print("="*70)

# Dialoqun 1-ci addımı:
response = client.responses.create(
    model=model_name,
    input="What is 5 + 4?",
)
print("Sual 1: 'What is 5 + 4?' -> Cavab:")
print(response.output_text)

# Dialoqun 2-ci addımı (əvvəlki cavabın ID-sini ötürürük):
second_response = client.responses.create(
    model=model_name,
    previous_response_id=response.id,
    input="Add 3 more to that.",
)
print("\nSual 2: 'Add 3 more to that.' (Əvvəlki kontekst saxlanılır) -> Cavab:")
print(second_response.output_text)


# =================================================================================
# 5. Model Kontekst Protokolu (MCP - Model Context Protocol) İstifadəsi
# =================================================================================
# MCP — LLM modellərinin kənar repozitoriyalara, verilənlər bazalarına və ya alətlərə
# standartlaşdırılmış şəkildə qoşulması üçün müasir protokoldur.
# Burada model GitMCP vasitəsilə birbaşa PyTorch-un rəsmi GitHub repozitoriyasına
# sorğu göndərərək texniki sualı cavablandırır.
# =================================================================================
print("\n" + "="*70)
print("5. MCP (Model Context Protocol) usage demo:")
print("="*70)

try:
    resp = client.responses.create(
        model=model_name,
        tools=[
            {
                "type": "mcp",
                "server_label": "gitmcp",
                "server_url": "https://gitmcp.io/pytorch/pytorch",
                "require_approval": "never",
            },
        ],
        input="How do I compile PyTorch with CUDA support?",
    )

    print("\nMCP istifadəsi ilə tam cavab obyekti:")
    parsed = resp.to_dict()
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

    print("\nOutput text only (Yekun texniki cavab):")
    print(resp.output_text)
except Exception as e:
    print(f"\n[QEYD] MCP serverinə qoşularkən xəta baş verdi: {e}")
    fallback_resp = client.responses.create(
        model=model_name,
        input="How do I compile PyTorch with CUDA support? Explain step by step."
    )
    print("\nModel cavabı (MCP-siz):")
    print(fallback_resp.output_text)


# =================================================================================
# 6. Dərin Mühakimə və Hesablama (Reasoning demo — o-series)
# =================================================================================
# OpenAI-ın 'o' seriyalı modelləri (o4-mini, o3-mini, o1) cavab verməzdən əvvəl
# daxili 'düşüncə zənciri' (Chain of Thought) qurur.
# `reasoning={"effort": "medium"}` parametri modelin nə qədər dərindən mühakimə
# yürüdəcəyini (low, medium, high) təyin edir.
# =================================================================================
print("\n" + "="*70)
print("6. Reasoning demo (Mürəkkəb riyazi və maliyyə planlaşdırılması):")
print("="*70)

prompt = """
Create a retirement plan for a 50-year-old male of average health in the United States.
Assume he has the maximum social security benefits, and that the expected reduction in benefits in 2033 occurs.
His current liquid assets saved are $1 million. How much more must he save each year in order to retire at age 
age 60, or age 65? His retirement goal is to survive until death on $100K per year, adjusted for inflation.
"""

try:
    response = client.responses.create(
        model=reasoning_model_name,
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )

    print("\nReasoning tam cavab obyekti:")
    parsed = response.to_dict()
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

    print("\nOutput text only (Məntiqi mühakimə nəticəsi):")
    print(response.output_text)
except Exception as e:
    print(f"\n[QEYD] Reasoning modeli ilə sorğuda xəta: {e}")
    # Ehtiyat olaraq standart gpt-4o ilə mühakimə aparırıq:
    fallback_reasoning = client.responses.create(
        model=model_name,
        input=prompt
    )
    print("\nModel cavabı (gpt-4o ilə):")
    print(fallback_reasoning.output_text)

print("\n" + "="*70)
print("Bütün nümayişlər uğurla başa çatdı!")
print("=======================================================================\n")
