"""
===================================================================================
3. Google Gemini və Generativ Süni İntellekt ilə Təsvir (Şəkil) Generasiyası
===================================================================================

Bu skript Generativ Süni İntellekt (Generative AI) vasitəsilə mətndən yüksək 
keyfiyyətli təsvirlərin yaradılmasını (Text-to-Image) və təsvir variasiyalarının 
alınmasını rəsmi google-genai (`from google import genai`) SDK-sı ilə nümayiş etdirir.

Text-to-Image (Mətndən Təsvirə) Nədir?
---------------------------------------
- Süni intellekt modelləri (məsələn, Google Imagen 3, OpenAI DALL-E, Stable Diffusion / Flux) 
  milyonlarla şəkil və onların mətn təsvirləri üzərində təlim keçmişdir.
- İstifadəçi təbii dildə bir təsvir (prompt) yazdıqda, model diffuziya (diffusion) 
  prosesi vasitəsilə təsadüfi səs-küydən (noise) addım-addım fotorealistik şəkillər formalaşdırır.

OpenAI və Google Imagen Arasındakı Əsas Fərq:
--------------------------------------------
- OpenAI DALL-E şəkli internetdə müvəqqəti bir URL ünvanı kimi qaytarır və endirmək 
  üçün əlavə `urllib.request.urlretrieve` tələb olunur.
- Google GenAI (Imagen 3) isə şəkli birbaşa ikili bayt massivi (`image_bytes`) 
  olaraq qaytarır; nəticədə şəkli dərhal `open("fayl.jpg", "wb").write(...)` ilə 
  kompyuterinizdə saxlaya bilərsiniz.

QEYD (Kvota Qorunması):
Google AI Studio-da Imagen 3 modeli Free Tier (ödənişsiz plan) üçün kvota limiti (429)
tətbiq etdikdə, dərsin dayanmaması üçün skript avtomatik olaraq açıq Generativ AI (Flux)
mühərriki ilə təsvirləri eyni parametr və promptlarla generasiya edib saxlayır.
===================================================================================
"""

import os
import sys
import random
import urllib.parse
import urllib.request
from google import genai
from google.genai import types

# Əgər layihədə python-dotenv quraşdırılıbsa, .env faylından açarları avtomatik oxuyur
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------------
# 1. Google Gemini API Açarının Əldə Edilməsi və Client Yaradılması
# ---------------------------------------------------------------------------------
# Əvvəlcə mühit dəyişənlərini yoxlayırıq:
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# Mühitdə açar tapılmazsa, terminaldan daxil edilməsini istəyirik:
if not api_key:
    print("\n" + "="*70)
    print("[DİQQƏT] Gemini API açarı mühitdə tapılmadı!")
    print("="*70)
    raw_key = input("Gemini API açarınızı daxil edin (AIzaSy...): ").strip()
    api_key = raw_key.strip('"').strip("'")

# google-genai Client obyektinin başladılması:
client = genai.Client(api_key=api_key) if api_key else genai.Client()


# =================================================================================
# 2. Şəkil Generasiyası üçün Əsas Promptun Müəyyənləşdirilməsi
# =================================================================================
# OpenAI nümunəsindəki klassik məzəli prompt:
# "Dramatik kinematoqrafik işıqlandırmada toyuğu qaçıran UNO (UFO)"
prompt = "A UFO abducting a chicken in dramatic cinematic lighting"

print("\n" + "="*70)
print(f"Here are two generated images of a UFO abducting a chicken:")
print(f"Prompt: \"{prompt}\"")
print("="*70)


# =================================================================================
# 3. İki Ədəd Təsvirin Generasiyası (OpenAI client.images.generate analoqu)
# =================================================================================
# Google GenAI-da `client.models.generate_images` metodu istifadə olunur:
#   - model: "imagen-3.0-generate-002" (Google-un ən son qabaqcıl təsvir modeli)
#   - prompt: Yaradılacaq təsvirin mətni
#   - config: Təsvir parametrləri (types.GenerateImagesConfig):
#       * number_of_images : Generasiya ediləcək şəkil sayı (burada 2 ədəd)
#       * output_mime_type : Təsvir formatı ('image/jpeg' və ya 'image/png')
#       * aspect_ratio     : Şəkil tərəflərinin nisbəti ('1:1' kvadrat, '16:9', '9:16', '4:3', '3:4')
# =================================================================================

try:
    print("\nGoogle Imagen 3 mühərriki ilə 2 şəkil generasiya edilir...")
    result = client.models.generate_images(
        model="imagen-3.0-generate-002",
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=2,
            output_mime_type="image/jpeg",
            aspect_ratio="1:1"
        )
    )
    # Rəsmi modeldən alınan şəkil baytları:
    img_1 = result.generated_images[0].image.image_bytes
    img_2 = result.generated_images[1].image.image_bytes
    engine_name = "Google Imagen 3"

except Exception as e:
    # Əgər hesabınızda ödəniş kartı bağlanmayıbsa və Google 429 limiti verərsə:
    print(f" -> [MƏLUMAT] Google Imagen 3 ödənişsiz rejimdə kvota limiti verdi (HTTP 429).")
    print("    Dərsin və təcrübənin dayanmaması üçün avtomatik Açıq Generativ AI (Flux) mühərrikinə keçilir...")
    
    url1 = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width=1024&height=1024&model=flux&seed=101&nologo=true"
    url2 = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width=1024&height=1024&model=flux&seed=202&nologo=true"
    
    req_func = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})).read()
    img_1, img_2 = req_func(url1), req_func(url2)
    engine_name = "Açıq Generativ AI (Flux SOTA)"


# =================================================================================
# 4. Generasiya Olunmuş Şəkillərin Fayl Kimi Saxlanılması
# =================================================================================
# OpenAI-da `urllib.request.urlretrieve` ilə internetdən endirildiyi halda,
# Google GenAI-da birbaşa baytları `open(..., "wb")` ilə diskinizə yazırıq:

# 1-ci təsvirin saxlanılması:
with open("chicken.jpg", "wb") as f:
    f.write(img_1)
print(f" -> [UĞURLU] 1-ci şəkil saxlandı: 'chicken.jpg' ({len(img_1)/1024:.1f} KB) [{engine_name}]")

# 2-ci təsvirin saxlanılması:
with open("chicken_2.jpg", "wb") as f:
    f.write(img_2)
print(f" -> [UĞURLU] 2-ci şəkil saxlandı: 'chicken_2.jpg' ({len(img_2)/1024:.1f} KB) [{engine_name}]")


# =================================================================================
# 5. Təsvir Variasiyası (OpenAI client.images.create_variation analoqu)
# =================================================================================
# Prompt Engineering (Prompt Mühəndisliyi) vasitəsilə eyni mövzunun fərqli üslubda
# (1980-ci illər retro sintveyv elmi-fantastik yağlı boya üslubu) variasiyası yaradılır:
# =================================================================================

print("\n" + "="*70)
print("Here's a variation of the first image (1980s Retro Sci-Fi Style):")
print("="*70)

variation_prompt = f"{prompt}, 1980s retro sci-fi synthwave art style, oil painting on canvas, vibrant colors"
print(f"Variasiya promptu: \"{variation_prompt}\"")

try:
    var_result = client.models.generate_images(
        model="imagen-3.0-generate-002",
        prompt=variation_prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            output_mime_type="image/jpeg",
            aspect_ratio="1:1"
        )
    )
    var_img = var_result.generated_images[0].image.image_bytes
    var_engine = "Google Imagen 3"
except Exception:
    var_url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(variation_prompt)}?width=1024&height=1024&model=flux&seed=999&nologo=true"
    var_img = urllib.request.urlopen(urllib.request.Request(var_url, headers={"User-Agent": "Mozilla/5.0"})).read()
    var_engine = "Açıq Generativ AI (Flux SOTA)"

# Variasiya təsvirini saxlayırıq:
with open("chicken_variation.jpg", "wb") as f:
    f.write(var_img)

print(f" -> [UĞURLU] Variasiya saxlandı: 'chicken_variation.jpg' ({len(var_bytes if 'var_bytes' in locals() else var_img)/1024:.1f} KB) [{var_engine}]")

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
