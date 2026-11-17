R"""
===================================================================================
5. NVIDIA NIM ilə Müasir Süni İntellekt İmkanları və Cavablar
   (nvidia/nemotron-3.5-super-vl-120b-a12b)
===================================================================================

Bu skript müasir Generativ Süni İntellektin bütün əsas imkanlarını bir yerdə
NVIDIA NIM (`https://integrate.api.nvidia.com/v1`) və `nvidia/nemotron-3.5-super-vl-120b-a12b`
modeli üzərində nümayiş etdirir:

1. Sadə Mətn və Sistem Təlimatı (System Instructions / Pirate persona)
2. Çoxmodallı Görmə: Təsvirin Analizi (Multimodal Vision — Yol nişanı və bird.png)
3. Maliyyə və Səhm Sorğusu (UDMY Stock Overview)
4. Dialoq Yaddaşı və Zəncirvari Əlaqə (Conversation State / Multi-turn Chat)
5. Model Kontekst Protokolu (MCP - Model Context Protocol) Konsepsiyası
6. Mürəkkəb Məntiqi Düşünmə və Hesablama (Reasoning Demo — `enable_thinking: True`)
===================================================================================
"""

import os
import sys
import base64
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
reasoning_model_name = "nvidia/nemotron-3.5-super-vl-120b-a12b"


# =================================================================================
# 1. Sadə Mətn Sorğusu və Sistem Təlimatı (Simple text prompt and response)
# =================================================================================
print("\n" + "="*70)
print("1. Simple text prompt and response (System Instruction / Pirate):")
print("="*70)

user_prompt = "What is Artificial Intelligence and how does it work?"

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "Always talk like a pirate."},
        {"role": "user", "content": user_prompt}
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=2048,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)

print("\nFull response:")
print(response)

print("\nText only:")
print(response.choices[0].message.content)


# =================================================================================
# 2. Şəkil Daxiletməsi və Mətn Çıxışı (Image input, text output — Vision Demo)
# =================================================================================
# `nvidia/nemotron-3.5-super-vl-120b-a12b` çoxmodallı (Vision-Language) modeldir.
# Həm uzaq URL şəkillərini, həm də yerli `bird.png` faylını (base64) birbaşa oxuyur!
# =================================================================================
print("\n" + "="*70)
print("2. Image input, text output (NVIDIA Nemotron-VL Vision demo):")
print("="*70)

# A) NVIDIA rəsmi prototipindəki yol nişanı şəklinin oxunması (Streaming ilə):
print("\n[A] NVIDIA Yol Nişanı Təsvirinin Analizi (stream=True):")
vision_stream = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read the traffic sign in this image. What does it say, and what color is the sign?"},
                {"type": "image_url", "image_url": {"url": "https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1a.jpg"}}
            ]
        }
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=4096,
    parallel_tool_calls=True,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    stream=True
)

for chunk in vision_stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

# B) Yerli 'bird.png' faylının analizi:
image_file = "bird.png"
if os.path.exists(image_file):
    with open(image_file, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    bird_resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What kind of bird is this?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }
        ],
        temperature=0.5,
        top_p=0.95,
        max_tokens=1024,
        seed=42,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        stream=False
    )
    print("\n[B] Yerli 'bird.png' Şəklinin Analizi:")
    print(bird_resp.choices[0].message.content)


# =================================================================================
# 3. Maliyyə və Səhm Sorğusu (Stock / Web Knowledge Demo)
# =================================================================================
print("\n" + "="*70)
print("3. Stock & Web Knowledge Query (UDMY):")
print("="*70)

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": "What is Udemy (UDMY) and its stock overview?"}
    ],
    temperature=0.5,
    max_tokens=1024,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)
print("\nModel response:")
print(response.choices[0].message.content)


# =================================================================================
# 4. Dialoq Vəziyyəti və Yaddaş (Conversation state / chaining responses)
# =================================================================================
print("\n" + "="*70)
print("4. Conversation state demo (Multi-turn chat):")
print("="*70)

chat_messages = [
    {"role": "user", "content": "What is 5 + 4?"}
]

response_1 = client.chat.completions.create(
    model=model_name,
    messages=chat_messages,
    temperature=0.0,
    max_tokens=512,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)
ans_1 = response_1.choices[0].message.content
print(ans_1)

chat_messages.append({"role": "assistant", "content": ans_1})
chat_messages.append({"role": "user", "content": "Add 3 more to that."})

response_2 = client.chat.completions.create(
    model=model_name,
    messages=chat_messages,
    temperature=0.0,
    max_tokens=512,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)
print(response_2.choices[0].message.content)


# =================================================================================
# 5. MCP (Model Context Protocol) Konsepsiyası və Texniki Sorğu
# =================================================================================
print("\n" + "="*70)
print("5. MCP (Model Context Protocol) usage demo:")
print("="*70)

resp = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a technical expert on machine learning infrastructure and PyTorch compilation."},
        {"role": "user", "content": "How do I compile PyTorch with CUDA support?"}
    ],
    temperature=0.5,
    max_tokens=2048,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)

print("\nOutput only:")
print(resp.choices[0].message.content)


# =================================================================================
# 6. Dərin Mühakimə və Hesablama (Reasoning demo — enable_thinking: True)
# =================================================================================
print("\n" + "="*70)
print("6. Reasoning demo (Complex financial planning with enable_thinking=True):")
print("="*70)

prompt = """
Create a retirement plan for a 50-year-old male of average health in the United States.
Assume he has the maximum social security benefits, and that the expected reduction in benefits in 2033 occurs.
His current liquid assets saved are $1 million. How much more must he save each year in order to retire at age 
age 60, or age 65? His retirement goal is to survive until death on $100K per year, adjusted for inflation.
"""

response = client.chat.completions.create(
    model=reasoning_model_name,
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=8192,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    stream=False
)

print("\nOutput text only:")
print(response.choices[0].message.content)

print("\n" + "="*70)
print("Bütün nümayişlər uğurla başa çatdı!")
print("=======================================================================\n")
