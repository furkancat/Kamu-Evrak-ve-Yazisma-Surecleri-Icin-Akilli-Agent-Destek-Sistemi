"""
langgraph_multi_agent.py
LangGraph Çok Ajanlı (Multi-Agent) RAG Pipeline

Ajan 1: Evrak Analizcisi  -> Evrakı okur, sınıflandırır, temiz arama sorgusu üretir
Ajan 2: Mevzuat Uzmanı    -> Qdrant'ta arar, mevzuatla kıyaslar, eksik denetler
Ajan 3: Raportör          -> Önceki çıktıları alır, resmî taslaklar üretir

Donanım: RTX 4070 (8GB VRAM) + 32GB RAM
LLM: Qwen 2.5 (Ollama)
Embedding: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Vektör DB: Qdrant (yerel disk)
"""

import os
import json
from typing import List, Dict, Any, TypedDict, Annotated
from operator import add

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# LangChain & LangGraph
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END


# KONFİGÜRASYON
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "mevzuat_kanunlar"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "qwen2.5"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 2048

device = "cuda" if torch.cuda.is_available() else "cpu"


# 0. MODÜLER LLM BAĞLANTISI
class LLMProvider:
    """Modüler LLM arayüzü. İleride farklı API'lere geçiş kolay."""
    def __init__(self, model_name: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        self.model_name = model_name
        self.temperature = temperature
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=LLM_MAX_TOKENS,
            )
        return self._llm

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        llm = self._get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        return response.content


# 0. QDRANT RETRIEVER
class MevzuatRetriever:
    """Qdrant üzerinden mevzuat araması yapar."""
    def __init__(self, qdrant_path: str = QDRANT_PATH, model_name: str = EMBED_MODEL):
        self.embed_model = SentenceTransformer(model_name, device=device)
        self.client = QdrantClient(path=qdrant_path)
        self.collection = COLLECTION_NAME

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        embedding = self.embed_model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device
        ).tolist()

        results = self.client.query_points(
            collection_name=self.collection,
            query=embedding,
            limit=top_k,
            with_payload=True
        )

        docs = []
        for r in results.points:
            docs.append({
                "score": round(r.score, 4),
                "tur": r.payload.get("tur", ""),
                "mevzuat_no": r.payload.get("mevzuat_no", ""),
                "madde_no": r.payload.get("madde_no", ""),
                "metin": r.payload.get("metin", "")[:1000]  # token sınırı için kısalt
            })
        return docs


# Global instance'lar (node fonksiyonları tarafından kullanılır)
llm_provider = LLMProvider()
retriever = MevzuatRetriever()


# STATE TANIMI (LangGraph TypedDict)
class AgentState(TypedDict):
    """Tüm ajanlar arasında paylaşılan durum (state)."""
    evrak_metni: str           # Gelen evrakın tam metni
    search_query: str          # Ajan 1'in ürettiği temiz arama sorgusu
    siniflandirma: str         # Ajan 1'in sınıflandırma çıktısı
    retrieved_docs: List[Dict] # Ajan 2'nin Qdrant'tan çektiği maddeler
    mevzuat_analizi: str       # Ajan 2'nin eksiklik raporu
    taslaklar: str             # Ajan 3'ün ürettiği resmî yazılar
    final_output: str          # Birleştirilmiş nihai rapor


# AJAN 1: EVRAK ANALİZCİSİ
PROMPT_AJAN1 = """Sen, Türkiye Cumhuriyeti kamu kurumlarında 20 yıllık deneyime sahip bir evrak analiz uzmanısın. 

KURALLAR:
1. Birimi SADECE şu listeden seç: Öğrenci İşleri Daire Başkanlığı, İlgili Fakülte Dekanlığı, Bilgi İşlem Daire Başkanlığı, Hukuk Müşavirliği, İmar ve Şehircilik Müdürlüğü, Zabıta Müdürlüğü, İnsan Kaynakları Müdürlüğü, Emeklilik Hizmetleri Daire Başkanlığı, Vergi Dairesi Müdürlüğü, Tapu Sicil Müdürlüğü, İlgili Mahkeme Kalemi, Yurt İdare Müdürlüğü, Veteriner İşleri Müdürlüğü.
2. Arama sorgusunda özel isim (Ahmet, Mehmet) kullanma, sadece resmi/hukuki terimler yaz.
3. KESİNLİKLE aşağıdaki 4 başlığı sırasıyla ve eksiksiz kullan.

LÜTFEN YANITINI TAM OLARAK AŞAĞIDAKİ ŞABLONA GÖRE ÜRET:

### 1. EVRAK ÖZETİ
[Buraya evrakın 2-3 cümlelik özetini yaz.]

### 2. AYIKLANAN BİLGİLER
Ad Soyad: [Bulunamazsa: Belirtilmemiş]
T.C. Kimlik No: [Bulunamazsa: Belirtilmemiş]
Tarih: [Bulunamazsa: Belirtilmemiş]
Konu: [Bulunamazsa: Belirtilmemiş]
Sayı: [Bulunamazsa: Belirtilmemiş]
Öğrenci No: [Bulunamazsa: Belirtilmemiş]
Parsel veya Ada No: [Bulunamazsa: Belirtilmemiş]

### 3. SINIFLANDIRMA
Evrak Türü: [Dilekçe / Başvuru / Bilgi Edinme vb.]
İşlem Türü: [Talep / Şikayet / Başvuru vb.]
Önerilen Birim: [Yukarıdaki listeden SADECE birini seç]
Birim Yönlendirme Gerekçesi: [1-2 cümlelik açıklama]

### 4. ARAMA KELİMELERİ
[Hukuki ve idari terimlerden oluşan 3-7 kelimelik net arama sorgusunu buraya yaz. BU KISMI KESİNLİKLE BOŞ BIRAKMA.]
"""


def evrak_analizcisi(state: AgentState) -> AgentState:
    """Ajan 1: Evrakı okur, sınıflandırır, temiz arama sorgusu üretir."""
    print("\n" + "━" * 60)
    print("  [AJAN 1] EVRAK ANALİZCİSİ çalışıyor...")
    print("━" * 60)

    evrak = state["evrak_metni"]
    user_prompt = f"Aşağıdaki evrakı analiz et:\n\n{evrak}"

    response = llm_provider.generate(PROMPT_AJAN1, user_prompt)

    # Çıktıyı parse et: SINIFLANDIRMA ve ARAMA SORGUSU ayır (regex ile esnek)
    import re
    
    # KRİTİK DÜZELTME: Modelin koyduğu kalın (**) ve tırnak ("") işaretlerini temizle
    clean_resp = response.replace("**", "").replace("\"", "")
    
    siniflandirma = clean_resp
    search_query = evrak  # fallback

    match = re.search(
        r'(?:---+\s*)?[\#\*\-]*\s*[Aa]rama\s+[Ss]orgusu\s*(?:---+)?\s*:?\s*\n?(.*?)(?:\n|$)',
        clean_resp, re.DOTALL | re.IGNORECASE
    )
    if match:
        siniflandirma = clean_resp[:match.start()].strip()
        search_query = match.group(1).strip().split("\n")[0].strip()
        # Eğer sorgu boşsa veya çok kısaysa, daha geniş bir alan dene
        if len(search_query) < 5:
            lines_after = match.group(1).strip().split("\n")
            for line in lines_after:
                line = line.strip()
                if len(line) >= 5:
                    search_query = line
                    break

    print(f"  ✓ Sınıflandırma tamamlandı.")
    print(f"  ✓ Arama sorgusu: \"{search_query}\"")

    return {
        **state,
        "siniflandirma": siniflandirma,
        "search_query": search_query,
    }


# AJAN 2: MEVZUAT UZMANI & EKSİK DENETLEYİCİ
PROMPT_AJAN2 = """Sen kıdemli bir hukuk müşavirisin. 
Evrak metni ile mevzuat maddelerini karşılaştırarak eksik belge denetimi yapacaksın.

KESİN VE İHLAL EDİLEMEZ KURALLAR:
1. Evrakta "ekte sunuyorum", "ektedir" denilen belgeleri ASLA eksik listesine yazma!
2. Vatandaş bir belgesinin olmadığını beyan edip bunun için hak talep ediyorsa, o belgeyi eksik olarak İSTEME.
3. Gelen mevzuat maddeleri konuyla ilgisizse "Konuyla ilgili uygun mevzuat maddesi bulunamamıştır" de.
4. ÇIKTI FORMATINDAKİ BAŞLIKLARI KESİNLİKLE DEĞİŞTİRME VE YANLARINA PARANTEZ İÇİNDE NOT EKLEME.

--- MEVZUAT ÖZETİ ---
[Mevzuat değerlendirmeni buraya yaz]

--- EKSİK BİLGİ / BELGE RAPORU ---
Eksiklik Durumu: [Sadece VAR veya YOK yaz]
Eksiklik Listesi: [Eksik belgeleri yaz veya 'Tespit edilememiştir' yaz]
Tamamlanması İçin Süre: [Sadece gün sayısı yaz]
"""

def mevzuat_uzmani(state: AgentState) -> AgentState:
    """Ajan 2: Qdrant'ta arama yapar, mevzuatla kıyaslar, eksik denetler."""
    print("\n" + "━" * 60)
    print("  [AJAN 2] MEVZUAT UZMANI çalışıyor...")
    print("━" * 60)

    sorgu = state["search_query"]
    evrak = state["evrak_metni"]

    # 1) Qdrant araması
    print(f"  → Qdrant araması: \"{sorgu}\" (limit=5)")
    docs = retriever.search(sorgu, top_k=5)
    print(f"  → {len(docs)} mevzuat maddesi bulundu.")
    for d in docs:
        print(f"     • {d['tur']} | No:{d['mevzuat_no']} | {d['madde_no']} | skor:{d['score']}")

    # 2) Mevzuat context'ini formatla
    context = "\n\n".join([
        f"[KAYNAK {i+1}] {doc['tur']} | No:{doc['mevzuat_no']} | {doc['madde_no']}\n{doc['metin']}"
        for i, doc in enumerate(docs)
    ])

    user_prompt = f"""EVRAK METNİ:
{evrak}

İLGİLİ MEVZUAT MADDELERİ:
{context}

Lütfen yukarıdaki kurallara göre eksiklik analizi yap."""

    response = llm_provider.generate(PROMPT_AJAN2, user_prompt)

    print(f"  ✓ Mevzuat analizi ve eksiklik raporu tamamlandı.")

    return {
        **state,
        "retrieved_docs": docs,
        "mevzuat_analizi": response,
    }


# AJAN 3: RAPORTÖR (TASLAK ÜRETİCİ)
PROMPT_AJAN3 = """Sen kurumsal bir raportörsün.
Amacın önceki raporları birleştirip resmi yazılar oluşturmaktır.

KURALLAR:
1. Üst Yazı (Metin) kısmı GERÇEKÇİ, RESMİ ve UZUN olmalıdır. Tek cümlelik özetler yazma. Vatandaşa hitaben düzgün bir bürokratik dille durumu anlat.
2. İmza kısmına evrakın geldiği kuruma uygun bir makam yaz.
3. Ajan 2 eksiklik "YOK" dediyse 3. Bölüme SADECE "Eksik evrak bulunmadığından talep yazısına gerek duyulmamıştır." yaz. 
4. Çıktıyı SADECE BİR KERE YAZ. "Aşağıdaki gibidir" diyerek metni tekrar etme. Şablon başlıklarının yanına kendi yorumunu veya parantez ekleme.
5. Eğer Ajan 2 eksiklik 'YOK' dediyse, 1. Bölümdeki (Üst Yazı) metin içinde KESİNLİKLE vatandaştan yeni belge veya evrak talep etme. Vatandaşa kurumun vereceği olağan cevabı, onay yazısını veya bilgilendirmeyi detaylı ve resmi bir dille anlat.

--- 1. ÜST YAZI (Kurumdan Vatandaşa) ---
Sayı: [Sayı]
Konu: [Konu]
Metin: [En az 3-4 cümlelik detaylı, resmi kurum cevabı]
İmza: [İlgili Kurum Birim Amiri Unvanı]

--- 2. BİLGİLENDİRME NOTU ---
[Kısa süreç açıklaması]

--- 3. EKSİK BELGE TALEP YAZISI (Varsa) ---
[Talep yazısı veya 'Gerek duyulmamıştır']
"""

def raportor(state: AgentState) -> AgentState:
    """Ajan 3: Önceki çıktıları alır, resmî taslaklar üretir."""
    print("\n" + "━" * 60)
    print("  [AJAN 3] RAPORTÖR çalışıyor...")
    print("━" * 60)

    siniflandirma = state["siniflandirma"]
    mevzuat_analizi = state["mevzuat_analizi"]

    user_prompt = f"""AJAN 1 - SINIFLANDIRMA RAPORU:
{siniflandirma}

AJAN 2 - MEVZUAT VE EKSİKLİK RAPORU:
{mevzuat_analizi}

Yukarıdaki uzman raporlarını temel alarak, kurumun vatandaşa vereceği resmî yazı 
taslaklarını hazırla."""

    response = llm_provider.generate(PROMPT_AJAN3, user_prompt)

    print(f"  ✓ Resmî yazı taslakları tamamlandı.")

    # Nihai raporu birleştir
    final = f"""
{'='*60}
  TEKNOFEST RAG AGENT - NİHAİ RAPOR
{'='*60}

[SINIFLANDIRMA]
{siniflandirma}

[MEVZUAT ANALİZİ]
{mevzuat_analizi}

[TASLAKLAR]
{response}

{'='*60}
"""

    return {
        **state,
        "taslaklar": response,
        "final_output": final,
    }


# LANGGRAPH STATEGRAPH YAPISI
def build_graph() -> StateGraph:
    """LangGraph StateGraph'ini oluşturur ve derler."""
    graph = StateGraph(AgentState)

    # Node'ları ekle
    graph.add_node("evrak_analizcisi", evrak_analizcisi)
    graph.add_node("mevzuat_uzmani", mevzuat_uzmani)
    graph.add_node("raportor", raportor)

    # Başlangıç noktası
    graph.set_entry_point("evrak_analizcisi")

    # Geçişler (linear pipeline)
    graph.add_edge("evrak_analizcisi", "mevzuat_uzmani")
    graph.add_edge("mevzuat_uzmani", "raportor")
    graph.add_edge("raportor", END)

    return graph.compile()


# TEST EVRAKLARI
TEST_EVRAGI_1 = """T.C.
ÖRNEK ÜNİVERSİTESİ REKTÖRLÜĞÜNE

Sayı    : ...
Konu    : Yatay Geçiş Başvurusu

Sayın Yetkililer,

Ben, Ahmet Yılmaz, T.C. kimlik numaram 12345678901. 2023-2024 eğitim-öğretim 
yılında X Üniversitesi Bilgisayar Mühendisliği 2. sınıf öğrencisiyim. 
Öğrenci numaram 20230056'dır.

Kardeşim de aynı şehirde üniversite okuduğu için ailemiz maddi zorluk 
çekmektedir. Bu nedenle Örnek Üniversitesi Bilgisayar Mühendisliği bölümüne 
yatay geçiş yapmak istiyorum.

Transkriptimi ve ders içeriklerini ekte sunuyorum. Lütfen başvurumu 
değerlendiriniz.

İmza
Ahmet Yılmaz"""

TEST_EVRAGI_2 = """T.C.
BELEDİYE BAŞKANLIĞINA

Sayı    : ...
Konu    : İmar Barışı Başvurusu

Sayın Başkan,

İlçenizde bulunan 1234 ada, 5 parsel sayılı taşınmazım için imar barışı 
başvurusu yapmak istiyorum. Yapı kayıt belgesi almak üzere gerekli 
ödemeyi yapacağımı taahhüt ederim.

Tapu fotokopisi ve yapı fotoğrafları ekte iletilmektedir. 
Yapı kullanma izin belgesi mevcut değildir.

Bilgilerinize arz ederim.

İmza
Mehmet Kaya"""


# ANA ÇALIŞMA AKIŞI
def main():
    print("=" * 65)
    print("  LANGGRAPH ÇOK AJANLI RAG PIPELINE")
    print("  TEKNOFEST Türkçe Yapay Zeka Dil Ajanları")
    print("=" * 65)
    print(f"  LLM: {LLM_MODEL} | Embedding: {EMBED_MODEL}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 65)

    # StateGraph'i derle
    app = build_graph()

    # Test evrakı seç
    evrak = TEST_EVRAGI_1

    print("\n📄 GELEN EVRAK:")
    print("-" * 65)
    print(evrak[:300] + "...")
    print("-" * 65)

    # Başlangıç state'i
    initial_state: AgentState = {
        "evrak_metni": evrak,
        "search_query": "",
        "siniflandirma": "",
        "retrieved_docs": [],
        "mevzuat_analizi": "",
        "taslaklar": "",
        "final_output": "",
    }

    # Graph'i çalıştır
    print("\n🚀 StateGraph çalıştırılıyor...")
    print("   Sıra: Ajan 1 → Ajan 2 → Ajan 3 → SON\n")

    final_state = app.invoke(initial_state)

    # Sonucu yazdır
    print(final_state["final_output"])

    # Detaylı çıktıları da kaydet (opsiyonel)
    with open("rapor_output.txt", "w", encoding="utf-8") as f:
        f.write(final_state["final_output"])
    print("\n💾 Rapor 'rapor_output.txt' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
