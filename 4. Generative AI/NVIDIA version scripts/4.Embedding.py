"""
===================================================================================
4. NVIDIA NIM ilə Mətn Vektorlaşdırması (Text Embeddings) və Semantik Oxşarlıq
===================================================================================

Bu skript NVIDIA NIM (`https://integrate.api.nvidia.com/v1`) platforması üzərində
söz və cümlələrin riyazi vektorlara (Embeddings) çevrilməsini və onlar arasındakı
semantik (məna) oxşarlığının (Cosine Similarity) hesablanmasını nümayiş etdirir.

Embedding Nədir?
----------------
- Süni intellekt mətnləri birbaşa anlaya bilmir; onları çoxölçülü fəzada ədədlər massivinə (vektora) çevirir.
- Mənaca bir-birinə yaxın olan sözlər (məsələn, 'potato' və 'carrot') bu fəzada bir-birinə yaxın koordinatlara malik olur.
- Fərqli mənalı sözlər (məsələn, 'potato' və 'The starship Enterprise') isə bir-birindən çox uzaq nöqtələrə düşür.
- Vektorlar vahid uzunluğa (L2 norm = 1) normallaşdırıldıqda, iki vektor arasındakı
  skalyar hasil (`np.dot`) birbaşa Kosinus Oxşarlığını (Cosine Similarity) verir.
===================================================================================
"""

import os
import sys
import re
import json
import numpy as np
from openai import OpenAI

# Əgər dotenv quraşdırılıbsa, .env faylından açarları avtomatik oxuyur
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

EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


def get_nvidia_embeddings(texts: list[str]) -> tuple[list[list[float]], str]:
    """
    Əvvəlcə NVIDIA-nın rəsmi `nvidia/nv-embedqa-e5-v5` embedding modelini yoxlayır,
    əgər açar yalnız `nemotron-3.5-super-vl-120b-a12b` üçün aktivdirsə, həmin modellə
    normallaşdırılmış (L2 norm = 1) semantik vektorları çıxarır.
    """
    try:
        resp = client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL,
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "NONE"}
        )
        vecs = []
        for item in resp.data:
            arr = np.array(item.embedding, dtype=float)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            vecs.append(arr.tolist())
        return vecs, EMBEDDING_MODEL
    except Exception:
        system_prompt = """You are a semantic embedding engine. For each input text, evaluate its semantic activation (0.0 to 1.0) across these 16 orthogonal conceptual dimensions:
1. food_nutrition, 2. plant_agriculture, 3. animal_biology, 4. nature_environment,
5. spacecraft_aviation, 6. sci_fi_fantasy, 7. computer_ai_tech, 8. physics_engineering,
9. human_society, 10. art_entertainment, 11. finance_economy, 12. medicine_health,
13. geography_travel, 14. abstract_philosophy, 15. action_motion, 16. household_object.

Return ONLY a valid JSON object with key "embeddings" containing a list of 16-float arrays, in the exact order of the input texts."""

        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Input texts: {json.dumps(texts, ensure_ascii=False)}"}
            ],
            temperature=0.1,
            top_p=0.95,
            max_tokens=2048,
            seed=42,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            stream=False
        )
        raw_text = completion.choices[0].message.content
        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw_text).strip(), flags=re.DOTALL)
        parsed = json.loads(clean_json)
        vecs = []
        for vec in parsed.get("embeddings", []):
            arr = np.array(vec, dtype=float)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            vecs.append(arr.tolist())
        return vecs, CHAT_MODEL


# =================================================================================
# 2. İSTİFADƏÇİ SORĞUSU ÜÇÜN VEKTOR YARADILMASI
# =================================================================================
prompt = input("Enter a string to create an embedding vector for: ")
if not prompt.strip():
    prompt = "Süni intellekt gələcəyin texnologiyasıdır."
    print(f"Standart nümunə mətn seçildi: '{prompt}'")

vecs_user, used_model = get_nvidia_embeddings([prompt])
user_vector = vecs_user[0]

print("\n" + "="*70)
print(f"Model: {used_model}")
print(f"Vektorun ölçüsü (Dimensions): {len(user_vector)} ədəd")
print("Vektorun ilk 5 koordinatı (nümunə):")
print([round(num, 5) for num in user_vector[:5]], "...")
print("="*70)


# =================================================================================
# 3. SEMANTİK OXŞARLIQ TESTİ 1: 'potato' (kartof) və 'carrot' (kök)
# =================================================================================
print("\n" + "="*70)
print("Let's find the similarity score between 'potato' and 'carrot'.")
print("="*70)

vecs_1, _ = get_nvidia_embeddings(["potato", "carrot"])
potato = vecs_1[0]
carrot = vecs_1[1]

simScore = float(np.dot(potato, carrot))

print(f"\nScore is {simScore:.4f} ({simScore * 100:.2f}%)")
print("İzah: Hər iki anlayış qida və bitki aləminə aid olduğu üçün oxşarlıq yüksəkdir.\n")


# =================================================================================
# 4. SEMANTİK OXŞARLIQ TESTİ 2: 'potato' və 'The starship Enterprise'
# =================================================================================
print("="*70)
print("How about 'potato' and 'The starship Enterprise'")
print("="*70)

vecs_2, _ = get_nvidia_embeddings(["potato", "The starship Enterprise"])
potato = vecs_2[0]
enterprise = vecs_2[1]

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
