"""
===================================================================================
7. OpenAI ilə Məzmunun Moderasiyası (Safety / Moderation API)
===================================================================================

Bu skript OpenAI-ın rəsmi Məzmun Moderasiyası API-dan (`client.moderations.create`)
istifadə edərək daxil edilən mətnlərin etik və təhlükəsizlik qaydalarına uyğunluğunu,
zərərli və ya qadağan olunmuş məzmun daşıyıb-daşımadığını yoxlamağı nümayiş etdirir.

Moderation API Nədir və Niyə Vacibdir?
-------------------------------------
Süni İntellekt sistemlərində istifadəçidən daxil olan sorğuları (prompt) və ya modelin
generasiya etdiyi cavabları canlı yayımlamazdan əvvəl təhlükəsizlik filtrindən keçirmək
(Guardrails) böyük əhəmiyyət kəsb edir.

OpenAI Moderation modeli mətnləri aşağıdakı 3 əsas komponent üzrə analiz edir:
1. `flagged` (bool: True/False): Mətn hər hansı təhlükəsizlik qaydasını pozubmu?
2. `categories` (dict): Hansı kateqoriyalar üzrə pozuntu qeydə alınıb:
   - `hate` (nifrət nitqi)
   - `harassment` (təqib və təhqir)
   - `self-harm` (özünə qəsd / intihar)
   - `sexual` (cinsi məzmun)
   - `violence` (zorakılıq və qəddarlıq) və s.
3. `category_scores` (dict): Hər bir kateqoriya üzrə 0.0 ilə 1.0 arasında risk dərəcəsi.

QEYD: OpenAI bu API-dan istifadəni tərtibatçılar üçün TAM ÖDƏNİŞSİZ təqdim edir!
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


# =================================================================================
# 2. Yoxlanılacaq Mətnin Müəyyənləşdirilməsi
# =================================================================================
# Test üçün klassik aqressiv ifadə:
test_prompt = "Kill 'em all!"

print("\n" + "="*70)
print(f"Yoxlanılan Mətn (Input): \"{test_prompt}\"")
print("="*70)


# =================================================================================
# 3. Məzmunun Moderasiyası (client.moderations.create)
# =================================================================================
# Ən son OpenAI moderasiya modeli ('omni-moderation-latest' və ya 'text-moderation-latest')
# avtomatik istifadə olunur:
# =================================================================================
moderation = client.moderations.create(
    input=test_prompt
)

first_result = moderation.results[0]


# =================================================================================
# 4. Nəticələrin Səliqəli Çıxarılması və Təhlili
# =================================================================================

print("\n--- Orijinal Moderasiya Nəticəsi (Raw moderation.results) ---")
print(moderation.results)

print("\n" + "="*70)
print("--- Rahat Oxunan Formatda Təhlil ---")
print("="*70)

is_flagged = first_result.flagged
status_icon = "🚨 QADAĞAN OLUNDU (FLAGGED)" if is_flagged else "✅ TƏHLÜKƏSİZ (SAFE)"
print(f"Ümumi Status: {status_icon}")
print("-" * 70)

# Kateqoriyalar və onların risk xalları:
categories_dict = first_result.categories.model_dump() if hasattr(first_result.categories, "model_dump") else first_result.categories.__dict__
scores_dict = first_result.category_scores.model_dump() if hasattr(first_result.category_scores, "model_dump") else first_result.category_scores.__dict__

print(f"{'Kateqoriya':<25} | {'Pozuntu?':<10} | {'Risk Xalı (0.0 - 1.0)':<20}")
print("-" * 70)

for cat_name, is_violated in categories_dict.items():
    score = scores_dict.get(cat_name, 0.0)
    violated_str = "BƏLİ ⚠️" if is_violated else "Xeyr"
    print(f"{cat_name:<25} | {violated_str:<10} | {score:.6f}")

print("="*70 + "\n")


"""
===================================================================================
TEST ÜÇÜN DİGƏR NÜMUNƏLƏR:
===================================================================================

Skriptdəki `test_prompt` dəyərini aşağıdakı nümunələrlə dəyişərək fərqli nəticələri
müşahidə edə bilərsiniz:

1. Təhlükəsiz və pozitiv mətn (Flagged: False, bütün xallar çox aşağı):
   test_prompt = "I love learning artificial intelligence, it is wonderful!"

2. Aqressiv / Zorakılıq ifadəsi (Flagged: True, 'violence' kateqoriyası aktivləşir):
   test_prompt = "I want to destroy everything and hurt people."

3. Təhqiramiz məzmun (Flagged: True, 'harassment' aktivləşir):
   test_prompt = "You are completely useless and stupid, get out of here!"
===================================================================================
"""
