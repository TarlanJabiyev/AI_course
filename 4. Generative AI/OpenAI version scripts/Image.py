"""
===================================================================================
4. OpenAI DALL-E ilə Təsvir (Şəkil) Generasiyası və Variasiyalar (Images API)
===================================================================================

Bu skript Generativ Süni İntellekt (Generative AI) vasitəsilə mətndən yüksək 
keyfiyyətli təsvirlərin yaradılmasını (Text-to-Image) və mövcud təsvirdən yeni 
variasiyaların alınmasını (Image Variation) rəsmi `openai` SDK-sı ilə nümayiş etdirir.

Text-to-Image (Mətndən Təsvirə) Nədir?
---------------------------------------
- DALL-E kimi Generativ Diffuziya Modelləri (Diffusion Models) milyonlarla təsvir 
  və onların mətn təsvirləri üzərində təlim keçmişdir.
- İstifadəçi təbii dildə bir təsvir (prompt) daxil etdikdə, model təsadüfi səs-küydən
  (noise) addım-addım pikselləri formalaşdıraraq fotorealistik şəkillər yaradır.

OpenAI Images API Mexanizmi:
----------------------------
1. `client.images.generate`: Mətndən yeni təsvirlər yaradır və onları müvəqqəti 
   bulud URL ünvanı (`data[i].url`) şəklində qaytarır.
2. `urllib.request.urlretrieve`: Qaytarılan URL-dən təsviri birbaşa yerli kompyuterinizə endirir.
3. `client.images.create_variation`: Mövcud şəkli (məs: `chicken.png`) analiz edərək
   onun kompozisiyasını qoruyan, lakin fərqli detallara sahib yeni variasiyasını hasil edir.

QEYD:
DALL-E 2 modeli eyni vaxtda birdən çox təsvir (`n=2`) və birbaşa təsvir variasiyası
(`create_variation`) imkanını dəstəkləyir.
===================================================================================
"""

import os
import sys
import urllib.request
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


# =================================================================================
# 2. Şəkil Generasiyası üçün Əsas Promptun Müəyyənləşdirilməsi
# =================================================================================
# Klassik məzəli test promptu:
# "Dramatik kinematoqrafik işıqlandırmada toyuğu qaçıran UNO (UFO)"
prompt = "A UFO abducting a chicken in dramatic cinematic lighting"

print("\n" + "="*70)
print(f"Here are two generated images of a UFO abducting a chicken:")
print(f"Prompt: \"{prompt}\"")
print("="*70)


# =================================================================================
# 3. İki Ədəd Təsvirin Generasiyası (client.images.generate)
# =================================================================================
# OpenAI Images API parametrləri:
#   - prompt: Yaradılacaq təsvirin təsviri
#   - n: Generasiya ediləcək şəkil sayı (DALL-E 2 ilə n=2)
#   - size: Təsvirin ölçüsü ('1024x1024', '512x512' və ya '256x256')
# =================================================================================
print("\nOpenAI DALL-E mühərriki ilə 2 ədəd şəkil generasiya edilir...")

newImage = client.images.generate(
    model="dall-e-2",
    prompt=prompt,
    n=2,
    size="1024x1024"
)

# Hər iki generasiya olunmuş şəklin müvəqqəti URL ünvanı:
image_url_1 = newImage.data[0].url
image_url_2 = newImage.data[1].url

print("\nGenerasiya olunmuş 1-ci təsvirin linki:")
print(image_url_1)

print("\nGenerasiya olunmuş 2-ci təsvirin linki:")
print(image_url_2)


# =================================================================================
# 4. Birinci Təsvirin Yerli Kompyuterə Endirilməsi və Saxlanılması
# =================================================================================
# OpenAI təsvirləri buludda müvəqqəti saxladığı üçün onları fayl olaraq yükləyirik:
filename_1 = "chicken.png"
print(f"\n1-ci şəkil endirilir və saxlanılır: '{filename_1}'...")
urllib.request.urlretrieve(image_url_1, filename_1)

print(f" -> [UĞURLU] Şəkil saxlandı: '{filename_1}' ({os.path.getsize(filename_1)/1024:.1f} KB)")


# =================================================================================
# 5. Təsvir Variasiyası (client.images.create_variation)
# =================================================================================
# Mövcud 'chicken.png' faylı modelə göndərilir və OpenAI həmin şəkildəki
# elementlərin mövqeyini, üslubunu dəyişərək yeni vizual variasiya formalaşdırır:
# =================================================================================
print("\n" + "="*70)
print("Here's a variation of the first image:")
print("="*70)

print(f"'{filename_1}' faylı əsasında yeni təsvir variasiyası generasiya edilir...")

with open(filename_1, "rb") as image_file:
    variation_response = client.images.create_variation(
        model="dall-e-2",
        image=image_file,
        n=1,
        size="1024x1024"
    )

variation_url = variation_response.data[0].url
print("\nVariasiya təsvirinin linki:")
print(variation_url)

# Variasiya təsvirini də kompyuterimizdə saxlayırıq:
filename_var = "chicken_variation.png"
print(f"\nVariasiya şəkli endirilir: '{filename_var}'...")
urllib.request.urlretrieve(variation_url, filename_var)
print(f" -> [UĞURLU] Variasiya saxlandı: '{filename_var}' ({os.path.getsize(filename_var)/1024:.1f} KB)")

print("\n" + "="*70)
print("Bütün şəkillər kompyuterinizdə uğurla saxlanıldı!")
print(f"Fayllar: '{filename_1}', '{filename_var}'")
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
