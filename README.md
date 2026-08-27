# Kamu Evrak ve Yazışma Süreçleri İçin Akıllı Agent Destek Sistemi

> Türkçe doğal dil işleme ve çok ajanlı (multi-agent) yapay zeka mimarisi ile kamu kurumlarındaki evrak sınıflandırma, mevzuat analizi, eksik belge tespiti ve resmî yazı taslağı üretimi süreçlerini otomatikleştiren uçtan uca RAG (Retrieval-Augmented Generation) tabanlı akıllı destek sistemi.

---

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Teknoloji Yığını](#teknoloji-yığını)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Dosya Yapısı](#dosya-yapısı)
- [Ekran Görüntüleri / Demo Akışı](#ekran-görüntüleri--demo-akışı)
- [Donanım ve Performans](#donanım-ve-performans)
- [Geliştirme Notları](#geliştirme-notları)
- [Lisans](#lisans)

---

## 🎯 Proje Hakkında

Kamu kurumlarında günlük işleyişin önemli bir bölümü; evrak hazırlama, inceleme, yönlendirme, yazışma üretimi ve arşivleme süreçlerinden oluşur. Bu süreçler çoğu zaman çok adımlı, tekrarlı, manuel müdahale gerektiren ve zaman baskısı altında yürütülen işlemlerdir.

Bu proje, söz konusu süreçleri yapay zeka destekli akıllı ajan sistemleriyle yeniden tasarlayarak daha verimli, daha güvenilir ve daha hızlı hale getirmeyi amaçlar. Sistem; gelen evrakı okur, anlamlandırır, sınıflandırır, ilgili mevzuat ve yönetmelikleri eşleştirir, eksik belgeleri tespit eder ve kurumun vatandaşa vereceği resmî yazı taslağını otomatik olarak üretir.

---

## ✨ Özellikler

### Görev 1: Evrak Sınıflandırma ve İçerik Analizi
- **OCR Desteği**: PDF (PyMuPDF) ve görüntü (Tesseract OCR) dosyalarından Türkçe metin çıkarımı
- **Akıllı Görüntü Filtresi**: Karanlık mod (siyah arka plan) evrakları otomatik tespit edip tersine çevirir ve netleştirir
- **Bilgi Çıkarımı**: Ad soyad, T.C. kimlik no, tarih, konu, sayı, öğrenci no, parsel/ada no gibi anahtar bilgileri ayıklar
- **Evrak Sınıflandırma**: Evrak türü (dilekçe, başvuru, bilgi edinme, şikayet) ve işlem türü (talep, şikayet, itiraz) belirleme
- **Birim Yönlendirme**: Evrakın hangi kurum birimine yönlendirilmesi gerektiğine dair tutarlı öneri
- **Özetleme**: Evrakın kısa ve öz özetini üretme

### Görev 2: Mevzuat Analizi ve Eksik Belge Tespiti
- **Vektör Tabanlı Mevzuat Araması**: Qdrant vektör veritabanı üzerinde kosinüs benzerliği ile en alakalı mevzuat maddelerini getirme
- **Eksik Bilgi/Belge Raporu**: Evrak metni ile mevzuat maddelerini karşılaştırarak eksik belge denetimi
- **Akıllı Eksiklik Filtresi**: "Ekte sunuyorum", "ektedir" gibi ifadelerle beyan edilen belgeleri eksik listesine almama mantığı
- **Mevzuat Dayanağı**: İşleme konu evraka ilişkin ilgili kanun, yönetmelik ve genelge maddelerini gösterme

### Görev 3: Resmî Yazı Taslaklama
- **Üst Yazı Üretimi**: Kurumdan vatandaşa hitaben resmî, detaylı ve bürokratik dilde cevap yazısı
- **Bilgilendirme Notu**: Süreç hakkında kısa açıklama
- **Eksik Belge Talep Yazısı**: Eksiklik varsa talep yazısı, yoksa "gerek duyulmamıştır" notu
- **PDF Çıktısı**: Üretilen resmî yazıların PDF olarak indirilebilir hale getirilmesi

---

## 🏗️ Sistem Mimarisi

Sistem, **LangGraph** tabanlı çok ajanlı (multi-agent) bir mimari üzerine kuruludur. Üç uzman ajan sıralı bir pipeline içinde çalışır:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     KAMU EVRAK AKILLI ANALİZ SİSTEMİ                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│   │   AJAN 1     │───▶│   AJAN 2     │───▶│   AJAN 3     │          │
│   │ Evrak        │    │ Mevzuat      │    │ Raportör     │          │
│   │ Analizcisi   │    │ Uzmanı       │    │ (Taslak)     │          │
│   └──────────────┘    └──────────────┘    └──────────────┘          │
│          │                   │                   │                  │
│          ▼                   ▼                   ▼                  │
│   • Sınıflandırma      • Qdrant Arama      • Üst Yazı               │
│   • Bilgi Çıkarımı     • Eksiklik Raporu   • Bilgilendirme Notu    │
│   • Arama Sorgusu      • Mevzuat Eşleşme   • Talep Yazısı          │
│                                                                     │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │              Paylaşılan State (AgentState)                  │   │
│   │  evrak_metni | search_query | siniflandirma | retrieved_docs │   │
│   │  mevzuat_analizi | taslaklar | final_output                  │   │
│   └────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Ajan Akışı

1. **Ajan 1 — Evrak Analizcisi**: Gelen evrak metnini okur, sınıflandırır, bilgileri ayıklar ve mevzuat araması için optimize edilmiş temiz arama sorgusu üretir.
2. **Ajan 2 — Mevzuat Uzmanı**: Ajan 1'in ürettiği sorgu ile Qdrant vektör veritabanında arama yapar, bulunan mevzuat maddelerini evrak metniyle karşılaştırır ve eksik belge raporu hazırlar.
3. **Ajan 3 — Raportör**: Önceki iki ajanın çıktılarını birleştirir, kurumun vatandaşa vereceği resmî yazı taslaklarını (üst yazı, bilgilendirme notu, eksik belge talep yazısı) üretir.

---

## 🛠️ Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| **LLM** | Qwen 2.5 (Ollama üzerinden yerel çalıştırma) |
| **Embedding** | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 boyut) |
| **Vektör DB** | Qdrant (yerel disk, Docker gerektirmez) |
| **Agent Framework** | LangGraph (StateGraph) |
| **Web Arayüzü** | Gradio |
| **OCR** | PyMuPDF (PDF) + Tesseract OCR (Görüntü) |
| **Görüntü İşleme** | OpenCV + Pillow + NumPy |
| **PDF Çıktı** | FPDF2 |
| **Dil** | Python 3.10+ |

---

## 📦 Kurulum

### 1. Gereksinimler

- Python 3.10 veya üzeri
- CUDA destekli GPU (tercihen, CPU'da da çalışır)
- [Ollama](https://ollama.com/) kurulu ve `qwen2.5` modeli çekilmiş olmalı
- Tesseract OCR (sistem kurulumu)

### 2. Tesseract OCR Kurulumu

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-tur
```

**Windows:**
- [Tesseract Windows Installer](https://github.com/UB-Mannheim/tesseract/wiki) indirin ve kurun.
- `C:\Program Files\Tesseract-OCR\tesseract.exe` yolunu ortam değişkenlerine ekleyin.

**macOS:**
```bash
brew install tesseract
```

### 3. Python Bağımlılıkları

```bash
pip install gradio langchain-ollama langgraph qdrant-client sentence-transformers
pip install pymupdf pytesseract pillow opencv-python numpy fpdf2 beautifulsoup4 playwright
```

### 4. Ollama Modelini İndirme

```bash
ollama pull qwen2.5
```

### 5. Mevzuat Veritabanını Hazırlama

Mevzuat verilerini kazıyın:
```bash
python mevzuat_scraper.py
```

Kazınan verileri Qdrant vektör veritabanına aktarın:
```bash
python mevzuat_to_qdrant.py
```

> **Not:** `mevzuat_veriseti_tam.json` dosyası oluşturulduktan sonra Qdrant embedding pipeline'ı çalıştırılır. İlk çalıştırmada ~5000-10000 mevzuat maddesinin embedding'i GPU üzerinde üretilecektir.

---

## 🚀 Kullanım

### Web Arayüzü (Gradio)

```bash
python app.py
```

Tarayıcınızda `http://localhost:7860` adresini açın.

**Arayüz Özellikleri:**
- **Dosya Yükleme**: PDF, PNG, JPG dosyalarını sürükleyip bırakarak OCR ile metin çıkarabilirsiniz.
- **Metin Girişi**: Doğrudan evrak metni yapıştırabilir veya hazır örnek evraklardan seçebilirsiniz.
- **Sekmeli Çıktılar**: Sınıflandırma, Eksik Belge Raporu ve Resmî Yazı Taslakları ayrı sekmelerde görüntülenir.
- **PDF İndirme**: Üretilen resmî yazıyı PDF olarak indirebilirsiniz.

### Komut Satırı (CLI)

Tek ajanlı RAG pipeline:
```bash
python rag_agent.py
```

Çok ajanlı LangGraph pipeline:
```bash
python langgraph_multi_agent.py
```

---

## 📁 Dosya Yapısı

```
.
├── app.py                      # Gradio web arayüzü + OCR entegrasyonu
├── langgraph_multi_agent.py    # LangGraph çok ajanlı RAG pipeline (3 ajan)
├── rag_agent.py                # Tek ajanlı uçtan uca RAG pipeline (alternatif)
├── mevzuat_scraper.py          # mevzuat.gov.tr'den mevzuat kazıma
├── mevzuat_to_qdrant.py        # Mevzuat verisini Qdrant'a embedding
├── mevzuat_veriseti_tam.json   # Kazınan mevzuat verisi (üretilir)
├── qdrant_db/                  # Yerel Qdrant vektör veritabanı (üretilir)
└── README.md                   # Bu dosya
```

| Dosya | Açıklama |
|-------|----------|
| `app.py` | Gradio tabanlı web arayüzü. OCR, örnek evrak seçimi, PDF çıktısı ve 3 sekmeli analiz sonuçları sunar. |
| `langgraph_multi_agent.py` | LangGraph StateGraph ile 3 ajanın sıralı çalıştığı ana pipeline. Her ajan farklı bir sorumluluk alanına sahiptir. |
| `rag_agent.py` | Tek LLM çağrısında tüm analizi yapan monolitik RAG pipeline (karşılaştırma ve test amaçlı). |
| `mevzuat_scraper.py` | Playwright + BeautifulSoup ile mevzuat.gov.tr'den kanun, yönetmelik ve genelge metinlerini kazır. |
| `mevzuat_to_qdrant.py` | SentenceTransformers ile metinleri vektöre çevirip Qdrant yerel veritabanına yazar. |

---

## 🖥️ Ekran Görüntüleri / Demo Akışı

### 1. Evrak Yükleme ve OCR
Kullanıcı PDF veya resim dosyası yükler. Sistem:
- PyMuPDF ile PDF'ten metin çıkarır.
- Tesseract OCR (Türkçe dil paketi) ile görüntüden metin çıkarır.
- Akıllı renk filtresi ile karanlık tema evrakları otomatik düzeltir.

### 2. Üç Ajanlı Analiz
**Sekme 1 — Sınıflandırma:**
- Evrak özeti
- Ayıklanan bilgiler (ad, TCKN, konu vb.)
- Evrak türü, işlem türü, önerilen birim

**Sekme 2 — Eksik Belge Raporu:**
- İlgili mevzuat maddeleri (Qdrant skorları ile)
- Eksiklik durumu (VAR/YOK)
- Eksiklik listesi (mantıksal filtreleme ile)
- Tamamlanma süresi

**Sekme 3 — Resmî Yazı Taslakları:**
- Üst yazı (kurumdan vatandaşa)
- Bilgilendirme notu
- Eksik belge talep yazısı (varsa)

### 3. PDF İndirme
Üretilen resmî yazı, Arial Türkçe font desteği ile PDF'e dönüştürülür ve indirilebilir.

---

## ⚙️ Donanım ve Performans

| Bileşen | Önerilen |
|---------|----------|
| GPU | NVIDIA RTX 4070 (8GB VRAM) veya üzeri |
| RAM | 32 GB |
| Disk | 10 GB (Qdrant + modeller için) |

**Embedding Hızı:** ~500-1000 kayıt/saniye (RTX 4070 ile)
**LLM Yanıt Süresi:** ~5-15 saniye (Qwen 2.5, 2048 token)

---

## 📝 Geliştirme Notları

### Çok Ajanlı vs. Tek Ajanlı Yaklaşım
- **LangGraph (Çok Ajanlı)**: Her ajan farklı bir sistem promptu ve sorumluluk alanına sahiptir. Ajanlar arası state paylaşımı ile daha modüler ve yönetilebilir bir yapı sunar.
- **RAGAgent (Tek Ajanlı)**: Tek LLM çağrısında tüm analizi yapar. Daha hızlıdır ancak karmaşık evraklarda ajanların uzmanlaşması kadar tutarlı olmayabilir.

### Önemli Kurallar
- Evrakta "ekte sunuyorum", "ektedir" denilen belgeler **asla** eksik listesine alınmaz.
- Mevzuat maddeleri evrak içeriğiyle alakasızsa "Uygun mevzuat bulunamadı" denir; uydurma madde gösterilmez.
- Üst yazılar gerçekçi, resmî ve en az 3-4 cümle uzunluğunda üretilir.
- Eksiklik yoksa vatandaştan yeni belge talep edilmez.

### Gelecek Geliştirmeler
- [ ] Daha büyük embedding modeli (BAAI/bge-m3) desteği
- [ ] Çoklu evrak toplu işleme (batch processing)
- [ ] Redis tabanlı oturum yönetimi
- [ ] REST API katmanı (FastAPI)
- [ ] Farklı LLM sağlayıcılarına geçiş (modüler LLMProvider yapısı hazır)

---

## 📜 Lisans

Bu proje açık kaynak olarak paylaşılmıştır. Kod bileşenleri /Apache 2.0 uyumlu lisanslar altında kullanılabilir. Üçüncü taraf açık-ağırlık (open-weight) modelleri (Qwen 2.5 vb.) için ilgili lisans metinleri ve kullanım koşullarına uyulması gerekmektedir.

---
