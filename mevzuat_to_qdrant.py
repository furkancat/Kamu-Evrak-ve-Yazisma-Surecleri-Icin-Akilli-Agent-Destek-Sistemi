"""
Mevzuat Verisi -> Qdrant Vektör Veritabanı (Yerel / Local)
TEKNOFEST Türkçe Yapay Zeka Dil Ajanları Yarışması - RAG Altyapısı

Donanım: RTX 4070 (8GB VRAM) + 32GB RAM
Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 boyut)
Vektör DB: Qdrant (yerel disk, Docker kullanmadan)
"""

import json
import os
import time
import hashlib
from uuid import uuid4

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ═══════════════════════════════════════════════════════════════
# KONFİGÜRASYON
# ═══════════════════════════════════════════════════════════════
JSON_PATH = "mevzuat_veriseti_tam.json"          # Girdi JSON
QDRANT_PATH = "./qdrant_db"                      # Yerel Qdrant dizini
COLLECTION_NAME = "mevzuat_kanunlar"             # Koleksiyon adı

# Model: 8GB VRAM'e rahat sığar (~500MB), Türkçe'de çok iyi çalışır
# Alternatif (daha güçlü, daha büyük): "BAAI/bge-m3"  -> 1024 boyut
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384                                # MiniLM-L12-v2 çıktı boyutu

BATCH_SIZE = 256                                 # GPU VRAM'e göre ayarlanabilir
# BATCH_SIZE = 128  # VRAM sorunu olursa bunu dene

# ───────────────────────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ───────────────────────────────────────────────────────────────
def deterministic_id(item: dict) -> str:
    """
    UUID yerine deterministik ID üretir.
    Aynı (tur, mevzuat_no, madde_no) her zaman aynı ID'yi verir.
    Güncelleme yaparken çok işe yarar.
    """
    raw = f"{item.get('tur','')}|{item.get('mevzuat_no','')}|{item.get('madde_no','')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def main():
    # 1) Cihaz tespiti
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  MEVZUAT -> QDRANT EMBEDDING PIPELINE")
    print("=" * 60)
    print(f"[1/6] PyTorch cihazı : {device}")
    if device == "cuda":
        print(f"       GPU           : {torch.cuda.get_device_name(0)}")
        print(f"       VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 2) Model yükleme (GPU'ya)
    print(f"[2/6] Model yükleniyor : {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"       Model boyutu    : ~{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parametre")

    # 3) Qdrant yerel başlatma
    print(f"[3/6] Qdrant başlatılıyor : {os.path.abspath(QDRANT_PATH)}")
    client = QdrantClient(path=QDRANT_PATH)

    # 4) Koleksiyon oluşturma (varsa silip yeniden)
    print(f"[4/6] Koleksiyon oluşturuluyor : '{COLLECTION_NAME}'")
    if client.collection_exists(COLLECTION_NAME):
        print("       Eski koleksiyon siliniyor...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f"       Vektör boyutu   : {VECTOR_SIZE}")
    print(f"       Mesafe metriği  : COSINE")

    # 5) JSON okuma
    print(f"[5/6] JSON okunuyor : {JSON_PATH}")
    if not os.path.exists(JSON_PATH):
        print(f"❌ HATA: '{JSON_PATH}' dosyası bulunamadı!")
        print("   Lütfen JSON dosyasını bu script ile aynı dizine koyun.")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"       Toplam kayıt    : {total:,}")

    # 6) Batch halinde embedding + Qdrant upsert
    print(f"[6/6] Embedding üretiliyor & Qdrant'a yazılıyor (batch={BATCH_SIZE})...")
    print("-" * 60)

    start_time = time.time()
    processed = 0

    for i in range(0, total, BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]
        texts = [item["metin"] for item in batch]

        # ── GPU'da embedding üret ──
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,   # Cosine similarity için normalize
            show_progress_bar=False,
            device=device
        )

        # ── PointStruct hazırla ──
        points = []
        for j, item in enumerate(batch):
            point_id = deterministic_id(item)  # veya str(uuid4())

            payload = {
                "tur": item.get("tur", ""),
                "mevzuat_no": item.get("mevzuat_no", ""),
                "madde_no": item.get("madde_no", ""),
                "metin": item.get("metin", "")
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[j].tolist(),
                    payload=payload
                )
            )

        # ── Qdrant'a yaz ──
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=False   # Asenkron yazım, çok daha hızlı
        )

        processed += len(batch)

        # İlerleme raporu
        if (i // BATCH_SIZE) % 10 == 0 or processed >= total:
            elapsed = time.time() - start_time
            percent = (processed / total) * 100
            speed = processed / elapsed if elapsed > 0 else 0
            print(
                f"   [{processed:>8,} / {total:,}]  %{percent:>5.1f}  |  "
                f"{speed:>6.1f} kayıt/s  |  {elapsed:>6.1f}s"
            )

    # ── Özet ──
    total_time = time.time() - start_time
    print("-" * 60)
    print("✅ TÜM İŞLEMLER TAMAMLANDI!")
    print(f"   • İşlenen kayıt     : {total:,}")
    print(f"   • Toplam süre       : {total_time:.1f} saniye")
    print(f"   • Ortalama hız      : {total/total_time:.1f} kayıt/saniye")
    print(f"   • Qdrant dizini     : {os.path.abspath(QDRANT_PATH)}")
    print(f"   • Koleksiyon adı    : {COLLECTION_NAME}")
    print("=" * 60)
    print("💡 Sorgu örneği (başka bir script'te kullanmak için):")
    print("""
    from qdrant_client import QdrantClient
    client = QdrantClient(path="./qdrant_db")

    results = client.search(
        collection_name="mevzuat_kanunlar",
        query_vector=model.encode("üniversite yönetmeliği öğrenci hakları").tolist(),
        limit=5
    )
    for r in results:
        print(r.payload["madde_no"], r.payload["tur"], r.score)
    """)


if __name__ == "__main__":
    main()