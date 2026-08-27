"""
rag_agent.py
TEKNOFEST Türkçe Yapay Zeka Dil Ajanları Yarışması
Uçtan Uça RAG Pipeline: Evrak Sınıflandırma + Eksik Bilgi Tespiti + Resmî Yazı Taslaklama

Donanım: RTX 4070 (8GB VRAM) + 32GB RAM
LLM: Qwen 2.5 (Ollama üzerinden)
Embedding: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Vektör DB: Qdrant (yerel disk)
"""

import os
import json
import textwrap
from typing import List, Dict, Any, Optional

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ── LangChain Ollama ──
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# ═══════════════════════════════════════════════════════════════
# KONFİGÜRASYON
# ═══════════════════════════════════════════════════════════════
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "mevzuat_kanunlar"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "qwen2.5"          # Ollama'da kurulu model adı
LLM_TEMPERATURE = 0.2          # Düşük sıcaklık = daha deterministik, kurumsal
LLM_MAX_TOKENS = 2048

# Cihaz tespiti
device = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════
# 1. MODÜLER LLM BAĞLANTISI
# ═══════════════════════════════════════════════════════════════
class LLMProvider:
    """
    Modüler LLM arayüzü. İleride OpenAI, Gemini, vs. geçişi kolaylaştırır.
    Şu an Ollama (Qwen 2.5) kullanıyor.
    """
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
        """Sistem + kullanıcı promptunu alır, LLM yanıtını döndürür."""
        llm = self._get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        return response.content

    def generate_structured(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """JSON formatında yanıt almayı dener. Başarısız olursa metin döner."""
        json_system = system_prompt + "\n\nYanıtını SADECE geçerli JSON formatında ver. Başka açıklama ekleme."
        raw = self.generate(json_system, user_prompt)
        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_output": raw, "parse_error": True}


# ═══════════════════════════════════════════════════════════════
# 2. EMBEDDING & RETRIEVAL (Qdrant)
# ═══════════════════════════════════════════════════════════════
class MevzuatRetriever:
    """Qdrant üzerinden mevzuat araması yapar."""
    def __init__(self, qdrant_path: str = QDRANT_PATH, model_name: str = EMBED_MODEL):
        print(f"[Retriever] Embedding modeli yükleniyor: {model_name} ({device})")
        self.embed_model = SentenceTransformer(model_name, device=device)
        print(f"[Retriever] Qdrant bağlanıyor: {os.path.abspath(qdrant_path)}")
        self.client = QdrantClient(path=qdrant_path)
        self.collection = COLLECTION_NAME

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Metni vektöre çevirip en alakalı mevzuat maddelerini getirir."""
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
                "metin": r.payload.get("metin", "")
            })
        return docs


# ═══════════════════════════════════════════════════════════════
# 3. GELİŞMİŞ PROMPT MÜHENDİSLİĞİ
# ═══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """Sen, Türkiye Cumhuriyeti kamu kurumlarında (özellikle üniversitelerde) çalışan son derece dikkatli, analitik ve deneyimli bir idari uzman ve hukuk danışmanısın.
Görevin, kuruma gelen evrakları (dilekçe, başvuru vb.) analiz etmek ve SIFIR mantık hatasıyla aşağıdaki 3 adımı uygulamaktır.

## KESİN VE İHLAL EDİLEMEZ KURALLAR (DİKKATLE OKU):
1. **EKSİK BELGE KURALI (ÇOK KRİTİK):** Eğer başvuru sahibi metin içinde "ekte sunuyorum", "ektedir", "ekledim", "fotokopisi ektedir" gibi ifadeler kullanmışsa, o belgeler KESİNLİKLE "Eksik" DEĞİLDİR. Mantıklı düşün; kişi belgeyi eklediğini söylüyorsa eksik olamaz! Bu belgeleri eksik listesine ASLA yazma.
2. **SINIFLANDIRMA KURALI:** Bir kişi bir hak istiyorsa, başvuru yapıyorsa veya bir işlem yapılmasını bekliyorsa bu bir "Talep"tir. "Öneri" değildir. İşlem türünü mantıklı seç.
3. **BİRİM YÖNLENDİRMESİ:** Üniversitelerdeki öğrenci başvuruları (yatay geçiş vb.) doğrudan "Rektör" şahsına değil, "Öğrenci İşleri Daire Başkanlığı"na, "İlgili Fakülte Dekanlığı"na veya "Enstitü Müdürlüğü"ne yönlendirilir. Gerçekçi bir kurum içi birim bul.
4. **MEVZUAT FİLTRESİ:** Sana verilen [KAYNAK] metinlerini dikkatle oku. Eğer kaynak metin "lisansüstü (yüksek lisans)" diyorsa ama dilekçe "lisans (2. sınıf)" öğrencisine aitse, o maddeyi dayanak olarak GÖSTERME. Alakasız veya yetersiz kaynak gelmişse uydurma, "Sağlanan mevzuatta bu duruma birebir uygun madde bulunamamıştır" de.
5. **ÜST YAZI VE HİYERARŞİ MANTIĞI:** Yazıları oluştururken kimlik karmaşası yaşama. Yazıyı gönderen makam kurumu (Örn: Örnek Üniversitesi Öğrenci İşleri), alıcı ise ya vatandaşın kendisi (Ahmet Yılmaz) ya da evrakın havale edileceği başka bir alt birimdir. Kurumun kendisine dilekçe yazdığı absürt metinler oluşturma.

## YANIT YAPISI (Lütfen tam olarak bu başlıkları kullan)

### 1. EVRAK SINIFLANDIRMASI VE BİRİM YÖNLENDİRMESİ
- Evrak Türü: [Dilekçe / Başvuru / Bilgi Edinme / Şikayet]
- İşlem Türü: [Talep / Şikayet / Bilgi Edinme / İtiraz] 
- Önerilen Birim: [Kurum içindeki ilgili daire başkanlığı veya fakülte]
- Birim Yönlendirme Gerekçesi: [Neden bu birime yönlendirildiğini idari bir dille açıkla]

### 2. EKSİK BİLGİ / BELGE TESPİTİ
- Eksiklik Durumu: [VAR / YOK] (Ekte sunulduğu belirtilen belgeleri eksik sayma!)
- Eksiklik Listesi: [Sadece gerçekten bahsedilmeyen/eksik belgeleri yaz, eğer her şey tamsa "Tespit edilememiştir" yaz]
- Mevzuat Dayanağı: [Alakalı kaynak varsa yaz, yoksa "Uygun mevzuat bulunamadı" de]
- Tamamlanması İçin Süre: [Süre belirtilmemişse standart 15 iş günü yaz]

### 3. RESMİ YAZI TASLAKLARI

#### A) ÜST YAZI (Kurum İçi Havale veya Kurumdan Vatandaşa)
Sayı: [Kurum Kodu/Yıl/Evrak No]
Konu: [Evrakın Konusu]
İlgi: [Başvuru sahibinin dilekçesi/tarihi]
Metin: [Başvurunun alındığı ve gereğinin yapılması/yapıldığına dair resmi, ciddi idari metin]
İmza: [Kurum Yetkilisi / Daire Başkanı Unvanı]

#### B) BİLGİLENDİRME NOTU (Vatandaşa E-posta/SMS formatında)
[Sayın Başvuru Sahibi, başvurunuz kurumumuz kayıtlarına alınmış olup... şeklinde kısa, nazik ve resmi bilgilendirme]

#### C) EKSİK BELGE TALEP YAZISI
[Eğer eksiklik durumu YOK ise buraya sadece "Başvuruda eksik evrak bulunmadığından bu yazı taslağına gerek duyulmamıştır." yaz. Uydurma belge isteme!]
"""


def build_rag_prompt(evrak_metni: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """Kullanıcı promptunu (RAG context + evrak) oluşturur."""
    # Mevzuat bağlamını formatla
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        block = (
            f"[KAYNAK {i}]\n"
            f"Tür: {doc['tur']} | No: {doc['mevzuat_no']} | Madde: {doc['madde_no']}\n"
            f"Metin: {doc['metin'][:800]}...\n"  # İlk 800 karakter (token sınırı için)
        )
        context_blocks.append(block)

    context = "\n".join(context_blocks)

    prompt = f"""Aşağıda kurumumuza gelen bir evrakın tam metni ve ilgili mevzuat maddeleri verilmiştir.
Lütfen yukarıdaki sistem talimatlarına göre analiz yap ve üç görevi tek yanıtta tamamla.

--- GELEN EVRAK METNİ ---
{evrak_metni}

--- İLGİLİ MEVZUAT MADDELERİ (Vektör Arama Sonucu) ---
{context}

--- ANALİZ SONUCU ---
"""
    return prompt


# ═══════════════════════════════════════════════════════════════
# 4. RAG AGENT (ANA MOTOR)
# ═══════════════════════════════════════════════════════════════
class RAGAgent:
    """Uçtan uca RAG pipeline: Retrieve -> Augment -> Generate"""
    def __init__(self):
        print("=" * 65)
        print("  RAG AGENT BAŞLATILIYOR")
        print("=" * 65)
        self.retriever = MevzuatRetriever()
        self.llm = LLMProvider()
        print("[Agent] Tüm bileşenler hazır.\n")

    def process(self, evrak_metni: str, top_k: int = 5, verbose: bool = True) -> Dict[str, Any]:
        """
        Evrakı alır, mevzuat arar, LLM ile analiz eder, sonucu döndürür.
        """
        # 1) RETRIEVE
        if verbose:
            print(f"[1/3] Vektör araması yapılıyor (top_k={top_k})...")
        docs = self.retriever.search(evrak_metni, top_k=top_k)

        if verbose:
            print(f"       {len(docs)} mevzuat maddesi bulundu.")
            for d in docs:
                print(f"       • {d['tur']} | No:{d['mevzuat_no']} | {d['madde_no']} | skor:{d['score']}")

        # 2) AUGMENT (Prompt oluştur)
        if verbose:
            print("\n[2/3] RAG promptu oluşturuluyor...")
        user_prompt = build_rag_prompt(evrak_metni, docs)

        # 3) GENERATE (LLM çağrısı)
        if verbose:
            print("[3/3] LLM (Qwen 2.5) analiz ediyor...")
        response = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        if verbose:
            print("       ✓ Analiz tamamlandı.\n")

        return {
            "evrak_metni": evrak_metni,
            "retrieved_docs": docs,
            "llm_response": response,
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE
        }

    def process_and_print(self, evrak_metni: str, top_k: int = 5):
        """İşlemi yapar ve sonucu şık bir şekilde terminale basar."""
        result = self.process(evrak_metni, top_k=top_k, verbose=True)

        print("=" * 65)
        print("  RAG AGENT - ANALİZ SONUCU")
        print("=" * 65)
        print(result["llm_response"])
        print("=" * 65)
        print(f"\nKullanılan Model: {result['model']} | Sıcaklık: {result['temperature']}")
        print(f"Mevzuat Kaynağı: {len(result['retrieved_docs'])} madde")
        return result


# ═══════════════════════════════════════════════════════════════
# 5. KURGUSAL TEST EVRAKLARI
# ═══════════════════════════════════════════════════════════════
TEST_EVRAGI_1 = """
T.C.
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
Ahmet Yılmaz
"""

TEST_EVRAGI_2 = """
T.C.
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
Mehmet Kaya
"""

TEST_EVRAGI_3 = """
T.C.
SAĞLIK BAKANLIĞI GENEL MÜDÜRLÜĞÜNE

Sayı    : ...
Konu    : E-Nabız Bilgi Edinme Başvurusu

Sayın Yetkililer,

E-Nabız sisteminde kayıtlı sağlık verilerimin bir kısmının eksik olduğunu 
fark ettim. 2024 yılına ait aile hekimi ziyaretlerim sistemde görünmüyor. 
Bu verilerin neden eksik olduğunu ve nasıl düzeltileceğini öğrenmek istiyorum.

Ayrıca, bu verilerin üçüncü taraflarla paylaşılıp paylaşılmadığını da 
bilgi edinmek istiyorum.

Bilgilerinize rica ederim.

İmza
Elif Demir
"""


# ═══════════════════════════════════════════════════════════════
# 6. ANA ÇALIŞMA AKIŞI
# ═══════════════════════════════════════════════════════════════
def main():
    agent = RAGAgent()

    print("\n" + "━" * 65)
    print("  TEST SENARYOSU 1: Üniversite Yatay Geçiş Başvurusu")
    print("━" * 65 + "\n")
    agent.process_and_print(TEST_EVRAGI_1, top_k=5)

    # İstersen diğer test evraklarını da çalıştırabilirsin:
    # print("\n" + "━" * 65)
    # print("  TEST SENARYOSU 2: İmar Barışı Başvurusu")
    # print("━" * 65 + "\n")
    # agent.process_and_print(TEST_EVRAGI_2, top_k=5)

    # print("\n" + "━" * 65)
    # print("  TEST SENARYOSU 3: E-Nabız Bilgi Edinme")
    # print("━" * 65 + "\n")
    # agent.process_and_print(TEST_EVRAGI_3, top_k=5)


if __name__ == "__main__":
    main()