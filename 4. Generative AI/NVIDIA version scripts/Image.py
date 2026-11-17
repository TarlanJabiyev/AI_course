"""
===================================================================================
3. NVIDIA NIM və Generativ Süni İntellekt ilə Təsvir (Şəkil) Generasiyası
===================================================================================

Bu skript Generativ Süni İntellekt (Generative AI) vasitəsilə mətndən yüksək 
keyfiyyətli təsvirlərin yaradılmasını (Text-to-Image) və təsvir variasiyalarının 
alınmasını NVIDIA NIM (`nvidia/nemotron-3.5-super-vl-120b-a12b`) ilə nümayiş etdirir.

İş Prinsipi:
1. `nvidia/nemotron-3.5-super-vl-120b-a12b` çoxmodallı modeli **Süni İntellekt Rəssamı
   (Prompt Engineer)** kimi çıxış edərək sadə istifadəçi ideyasını peşəkar vizual
   təsvirə və 1980-ci illər retro elmi-fantastik üslub variasiyasına çevirir.
2. Hazırlanan zəngin təsvirlər Generativ AI (Flux) mühərrikinə ötürülür və kompyuterinizdə
   `chicken.jpg`, `chicken_2.jpg` və `chicken_variation.jpg` olaraq saxlanılır.
===================================================================================
"""

import os
import sys
import urllib.parse
import urllib.request
from openai import OpenAI

# Əgər layihədə python-dotenv quraşdırılıbsa, .env faylından açarları avtomatik oxuyur
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
MODEL_NAME = "nvidia/nemotron-3.5-super-vl-120b-a12b"


# =================================================================================
# 2. Şəkil Generasiyası üçün Əsas Promptun Müəyyənləşdirilməsi və Zənginləşdirilməsi
# =================================================================================
prompt = "A UFO abducting a chicken in dramatic cinematic lighting"

print("\n" + "="*70)
print("Here are two generated images of a UFO abducting a chicken:")
print(f"İlkin Prompt: \"{prompt}\"")
print("="*70)

enhance_resp = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "You are an expert AI image prompt engineer. Enhance the user's prompt into a vivid 1-sentence visual description (max 25 words). Output ONLY the prompt."
        },
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)
enhanced_prompt = str(enhance_resp.choices[0].message.content).strip().strip('"')
print(f"NVIDIA Nemotron tərəfindən zənginləşdirilmiş prompt:\n -> \"{enhanced_prompt}\"")


# =================================================================================
# 3. İki Ədəd Təsvirin Generasiyası
# =================================================================================
def download_image_bytes(target_prompt: str, seed: int) -> bytes:
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(target_prompt)}?width=1024&height=1024&model=flux&seed={seed}&nologo=true"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=30).read()


print("\nNVIDIA Nemotron + Generativ Mühərrik ilə 2 şəkil generasiya edilir...")
img_1 = download_image_bytes(enhanced_prompt, seed=101)
img_2 = download_image_bytes(enhanced_prompt, seed=202)
engine_name = "NVIDIA Nemotron + Flux SOTA"


# =================================================================================
# 4. Generasiya Olunmuş Şəkillərin Fayl Kimi Saxlanılması
# =================================================================================
with open("chicken.jpg", "wb") as f:
    f.write(img_1)
print(f" -> [UĞURLU] 1-ci şəkil saxlandı: 'chicken.jpg' ({len(img_1)/1024:.1f} KB) [{engine_name}]")

with open("chicken_2.jpg", "wb") as f:
    f.write(img_2)
print(f" -> [UĞURLU] 2-ci şəkil saxlandı: 'chicken_2.jpg' ({len(img_2)/1024:.1f} KB) [{engine_name}]")


# =================================================================================
# 5. Təsvir Variasiyası (OpenAI client.images.create_variation analoqu)
# =================================================================================
print("\n" + "="*70)
print("Here's a variation of the first image (1980s Retro Sci-Fi Style):")
print("="*70)

var_resp = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "Rewrite the image prompt in a 1980s retro sci-fi synthwave oil painting style (max 25 words). Output ONLY the prompt."
        },
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    seed=42,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)
variation_prompt = str(var_resp.choices[0].message.content).strip().strip('"')
print(f"NVIDIA Nemotron Variasiya promptu:\n -> \"{variation_prompt}\"")

var_img = download_image_bytes(variation_prompt, seed=999)

with open("chicken_variation.jpg", "wb") as f:
    f.write(var_img)

print(f" -> [UĞURLU] Variasiya saxlandı: 'chicken_variation.jpg' ({len(var_img)/1024:.1f} KB) [{engine_name}]")

print("\n" + "="*70)
print("Bütün şəkillər kompyuterinizdə uğurla saxlanıldı!")
print("Fayllar: 'chicken.jpg', 'chicken_2.jpg', 'chicken_variation.jpg'")
print("="*70 + "\n")


"""
===================================================================================
TEST ÜÇÜN DİGƏR MARAQLI ÖRNƏK PROMPTLAR:
===================================================================================

Skriptdəki `prompt` dəyişənini aşağıdakı nümunələrlə əvəz edərək fərqli üslubları
sınaqdan keçirə bilərsiniz:

1. Fotorealistik Şəhər Mənzərəsi (Baku Skyline):
   prompt = "A hyper-realistic photo of Baku city skyline at sunset, with Flame Towers glowing, Caspian sea reflecting city lights, 8k resolution, cinematic lighting"

2. Kiberpank / Gələcək Texnologiyası:
   prompt = "A cyberpunk street food vendor in futuristic Tokyo, neon signs glowing in the rain, ultra detailed, octane render"

3. Bədii Yağlı Boya / İncəsənət (Van Qoq Üslubu):
   prompt = "A majestic lion resting on a mountain peak, Vincent van Gogh starry night painting style, expressive brushstrokes"

4. 3D Animasiya / Sevimli Personaj:
   prompt = "A cute tiny robot drinking coffee in a cozy autumn cafe, 3D Pixar character render, soft warm lighting"

5. Kosmik Fantastika:
   prompt = "An astronaut sitting on the edge of the moon drinking coffee, looking down at Earth glowing blue in the dark space, cinematic photo"
===================================================================================
"""
