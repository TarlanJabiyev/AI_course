"""
===================================================================================
4. Google Gemini ilə Mətn Vektorlaşdırması (Text Embeddings) və Semantik Oxşarlıq
===================================================================================

Bu skript Google Gemini-nin rəsmi mətn vektorlaşdırma (Embedding) modelindən
(text-embedding-004) istifadə edərək söz və cümlələrin riyazi vektorlara
çevrilməsini və onlar arasındakı semantik (məna) oxşarlığının (Cosine Similarity)
hesablanmasını rəsmi google-genai (`from google import genai`) SDK-sı ilə nümayiş etdirir.

Embedding Nədir?
- Süni intellekt mətnləri birbaşa anlaya bilmir; onları çoxölçülü fəzada ədədlər massivinə (vektora) çevirir.
- Mənaca bir-birinə yaxın olan sözlər (məsələn, 'potato' və 'carrot') bu fəzada bir-birinə yaxın koordinatlara malik olur.
- Fərqli mənalı sözlər (məsələn, 'potato' və 'The starship Enterprise') isə bir-birindən çox uzaq nöqtələrə düşür.
- text-embedding-004 vektorları artıq vahid uzunluğa (L2 norm = 1) malik olduğu üçün
  iki vektor arasındakı skalyar hasil (np.dot) birbaşa Kosinus Oxşarlığını (Cosine Similarity) verir.
===================================================================================
"""

import os
import sys
import numpy as np
from google import genai

# Əgər dotenv quraşdırılıbsa, .env faylından açarları avtomatik oxuyur
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------------
# 1. Google Gemini API Açarının Əldə Edilməsi və Client Yaradılması
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
MODEL = "text-embedding-004"


# =================================================================================
# 2. İSTİFADƏÇİ SORĞUSU ÜÇÜN VEKTOR YARADILMASI
# =================================================================================
prompt = input("Enter a string to create an embedding vector for: ")
if not prompt:
    prompt = "Süni intellekt gələcəyin texnologiyasıdır."
    print(f"Standart nümunə mətn seçildi: '{prompt}'")

response = client.models.embed_content(
    model=MODEL,
    contents=prompt
)

user_vector = response.embeddings[0].values

print("\n" + "="*70)
print(f"Model: {MODEL}")
print(f"Vektorun ölçüsü (Dimensions): {len(user_vector)} ədəd")
print("Vektorun ilk 5 koordinatı (nümunə):")
print([round(num, 5) for num in user_vector[:5]], "...")
print("="*70)
print("\nTam Cavab Obyekti (Complete Response):")
print(response)


# =================================================================================
# 3. SEMANTİK OXŞARLIQ TESTİ 1: 'potato' (kartof) və 'carrot' (kök)
# =================================================================================
print("\n" + "="*70)
print("Let's find the similarity score between 'potato' and 'carrot'.")
print("="*70)

response = client.models.embed_content(
    model=MODEL,
    contents=["potato", "carrot"]
)

potato = response.embeddings[0].values
carrot = response.embeddings[1].values

simScore = float(np.dot(potato, carrot))

print(f"\nScore is {simScore:.4f} ({simScore * 100:.2f}%)")
print("İzah: Hər iki anlayış qida və bitki aləminə aid olduğu üçün oxşarlıq yüksəkdir.\n")


# =================================================================================
# 4. SEMANTİK OXŞARLIQ TESTİ 2: 'potato' və 'The starship Enterprise'
# =================================================================================
print("="*70)
print("How about 'potato' and 'The starship Enterprise'")
print("="*70)

response = client.models.embed_content(
    model=MODEL,
    contents=["potato", "The starship Enterprise"]
)

potato = response.embeddings[0].values
enterprise = response.embeddings[1].values

simScore2 = float(np.dot(potato, enterprise))

print(f"\nScore is {simScore2:.4f} ({simScore2 * 100:.2f}%)")
print("İzah: Tərəvəz ilə fantastik kosmik gəmi arasında məna əlaqəsi olmadığı üçün xal aşağıdır.\n")


# =================================================================================
# 5. YEKUN NƏTİCƏ MÜQAYİSƏSİ
# =================================================================================
print("="*70)
print("YEKUN NƏTİCƏ:")
print("="*70)
print(f"1. 'potato' <-> 'carrot'                 : {simScore:.4f}  [Yüksək Oxşarlıq]")
print(f"2. 'potato' <-> 'The starship Enterprise' : {simScore2:.4f}  [Aşağı Oxşarlıq]")
print("-" * 70)
print("NƏTİCƏ: Model sözləri hərflərinə görə deyil, daşıdığı mənaya (semantikaya) görə əlaqələndirir.")
print("=======================================================================\n")


"""
===================================================================================
TEST ÜÇÜN DİGƏR MARAQLI MƏNA CÜTLÜKLƏRİ:
===================================================================================

Aşağıdakı söz və cümlə cütlüklərini skriptdə yoxlayaraq fərqi görə bilərsiniz:

1. Sinonimlər və Eyni Anlayışlar (Çox Yüksək Oxşarlıq: ~0.80 - 0.95):
   - "avtomobil" <-> "maşın"
   - "həkim" <-> "tibb işçisi"
   - "king" <-> "queen"

2. Əlaqəli, lakin Fərqli Anlayışlar (Orta Oxşarlıq: ~0.45 - 0.65):
   - "tələbə" <-> "kitabxana"
   - "həkim" <-> "xəstəxana"
   - "yağış" <-> "çətir"

3. Fərqli Dillərdə Eyni Məna (Çoxdilli / Multilingual Uyğunluq: ~0.75 - 0.90):
   - "kitab" (Azərbaycan) <-> "book" (İngilis)
   - "günəş" (Azərbaycan) <-> "sun" (İngilis)

4. Tamamilə Fərqli / Zidd Mənalar (Aşağı Oxşarlıq: < 0.25):
   - "pizza" <-> "kvant fizikası"
   - "dondurma" <-> "qlobal istiləşmə"
===================================================================================
"""
