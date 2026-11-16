"""
===================================================================================
3. OpenAI ilə Mətn Vektorlaşdırması (Text Embeddings) və Semantik Oxşarlıq
===================================================================================

Bu skript OpenAI-ın rəsmi mətn vektorlaşdırma (Embedding) modelindən
(`text-embedding-ada-002` və ya `text-embedding-3-small`) istifadə edərək
söz və cümlələrin çoxölçülü riyazi vektorlara çevrilməsini və onlar arasındakı
semantik (məna) oxşarlığının (Cosine Similarity) hesablanmasını nümayiş etdirir.

Embedding Nədir və Necə İşləyir?
--------------------------------
1. Süni intellekt mətnləri birbaşa anlaya bilmir; onları çoxölçülü fəzada ədədlər
   massivinə (vektora) çevirir (məsələn, 1536 ölçülü fəza).
2. Mənaca bir-birinə yaxın olan anlayışlar (məsələn: 'potato' və 'carrot') bu fəzada
   bir-birinə yaxın bucaq və koordinatlara malik olur.
3. Fərqli mənalı anlayışlar (məsələn: 'potato' və 'The starship Enterprise') isə
   fəzada bir-birindən çox uzaq nöqtələrə düşür.
4. OpenAI-ın embedding vektorları artıq standart olaraq vahid uzunluğa (L2 norm = 1)
   normallaşdırıldığı üçün iki vektor arasındakı Kosinus Oxşarlığı (Cosine Similarity)
   sadəcə onların skalyar hasilinə (`np.dot`) bərabər olur:
   
       Cosine_Similarity(A, B) = dot(A, B) / (||A|| * ||B||) = dot(A, B)
===================================================================================
"""

import os
import sys
import numpy as np
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

# İstifadə ediləcək OpenAI Embedding modeli (text-embedding-ada-002 və ya text-embedding-3-small):
MODEL = "text-embedding-ada-002"


# =================================================================================
# 2. İSTİFADƏÇİ SORĞUSU ÜÇÜN VEKTOR (EMBEDDING) YARADILMASI
# =================================================================================
prompt = input("Enter a string to create an embedding vector for: ")
if not prompt.strip():
    prompt = "Süni intellekt gələcəyin texnologiyasıdır."
    print(f"Standart nümunə mətn seçildi: '{prompt}'")

response = client.embeddings.create(
    input=prompt,
    model=MODEL
)

user_vector = response.data[0].embedding

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

# Hər iki sözü eyni sorğuda vektorlaşdırırıq:
response = client.embeddings.create(
    input=["potato", "carrot"],
    model=MODEL
)

potato = response.data[0].embedding
carrot = response.data[1].embedding

# OpenAI vektorları normallaşdırıldığı üçün Kosinus Oxşarlığı = np.dot(u, v)
simScore = float(np.dot(potato, carrot))

print(f"\nScore is {simScore:.4f} ({simScore * 100:.2f}%)")
print("İzah: Hər iki anlayış ('potato' - kartof, 'carrot' - ravənd bitkisi) qida və")
print("kənd təsərrüfatı aləminə aid olduğu üçün onların semantik oxşarlığı yüksəkdir.\n")


# =================================================================================
# 4. SEMANTİK OXŞARLIQ TESTİ 2: 'potato' və 'The starship Enterprise'
# =================================================================================
print("="*70)
print("How about 'potato' and 'The starship Enterprise'")
print("="*70)

response = client.embeddings.create(
    input=["potato", "The starship Enterprise"],
    model=MODEL
)

potato = response.data[0].embedding
enterprise = response.data[1].embedding

simScore2 = float(np.dot(potato, enterprise))

print(f"\nScore is {simScore2:.4f} ({simScore2 * 100:.2f}%)")
print("İzah: Tərəvəz ('potato') ilə Star Trek elmi-fantastik kosmik gəmisi")
print("('The starship Enterprise') arasında heç bir məna bağlılığı olmadığı üçün xal aşağıdır.\n")


# =================================================================================
# 5. YEKUN NƏTİCƏ MÜQAYİSƏSİ
# =================================================================================
print("="*70)
print("YEKUN NƏTİCƏ MÜQAYİSƏSİ:")
print("="*70)
print(f"1. 'potato' <-> 'carrot'                 : {simScore:.4f}  [Yüksək Oxşarlıq]")
print(f"2. 'potato' <-> 'The starship Enterprise' : {simScore2:.4f}  [Aşağı Oxşarlıq]")
print("-" * 70)
print("NƏTİCƏ: Model sözləri hərflərinə görə deyil, daşıdığı mənaya (semantikaya) görə fəzada qruplaşdırır.")
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
