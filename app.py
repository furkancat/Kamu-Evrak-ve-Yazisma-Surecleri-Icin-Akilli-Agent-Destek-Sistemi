"""
app.py
TEKNOFEST Türkçe Yapay Zeka Dil Ajanları Yarışması
Gradio Web Arayüzü - LangGraph Çok Ajanlı RAG Pipeline + OCR

Desteklenen Dosyalar: PDF, PNG, JPG, JPEG
OCR: PyMuPDF (PDF) + Tesseract/pytesseract (Resim)

Kullanım:
    pip install gradio pymupdf pytesseract pillow
    # Tesseract OCR sistem kurulumu:
    #   Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-tur
    #   Windows: https://github.com/UB-Mannheim/tesseract/wiki
    #   Mac: brew install tesseract
    python app.py
    → Tarayıcıda http://localhost:7860 açılır
"""

import os
import gradio as gr
from langgraph_multi_agent import build_graph, AgentState
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# ── OCR Kütüphaneleri ──
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
    import pytesseract
    import cv2
    import numpy as np
    # Windows için Tesseract yolunu zorunlu olarak belirtiyoruz!
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    Image = None
    pytesseract = None
    cv2 = None
    np = None


# ═══════════════════════════════════════════════════════════════
# OCR FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════

def akilli_resim_filtresi(resim_yolu):
    """Karanlık mod (siyah arka plan) resimleri tespit edip tersine çevirir ve netleştirir."""
    # Kütüphaneler eksikse veya okuma hatası olursa varsayılan (normal) resmi döndür
    if cv2 is None or np is None:
        return Image.open(resim_yolu)
        
    # 1. Görüntüyü gri tonlamalı (siyah-beyaz) olarak yükle
    img_array = np.fromfile(resim_yolu, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return Image.open(resim_yolu)
        
    # 2. Görüntünün ortalama parlaklığını hesapla
    ortalama_parlaklik = np.mean(img)
    
    # 3. Eğer görüntü karanlıksa (karanlık tema/siyah arka plan), renkleri tersine çevir
    if ortalama_parlaklik < 127:
        img = cv2.bitwise_not(img)
        
    # 4. Yazıları daha keskin hale getirmek için Eşikleme (Otsu Thresholding) uygula
    _, temiz_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Pytesseract'ın okuyabilmesi için PIL Image formatına dönüştür
    return Image.fromarray(temiz_img)


def extract_text_from_pdf(file_path: str) -> str:
    """PyMuPDF (fitz) ile PDF'ten metin çıkarır."""
    if fitz is None:
        return "❌ HATA: PyMuPDF (pymupdf) kurulu değil. Kurulum: pip install pymupdf"

    text_parts = []
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc, 1):
            txt = page.get_text()
            if txt.strip():
                text_parts.append(f"--- Sayfa {page_num} ---\n{txt.strip()}")
        doc.close()

        if not text_parts:
            return "⚠️ PDF'ten metin çıkarılamadı. Dosya taranmış görüntü (image-based) PDF olabilir."
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"❌ PDF okuma hatası: {str(e)}"


def extract_text_from_image(file_path: str) -> str:
    """Tesseract OCR ile resimden metin çıkarır (Akıllı Renk Filtresi ve Türkçe destekli)."""
    if Image is None or pytesseract is None:
        return "❌ HATA: pytesseract veya Pillow kurulu değil. Kurulum: pip install pytesseract pillow opencv-python numpy"

    try:
        # ESKİ HALİ: img = Image.open(file_path)
        # YENİ HALİ: Resmi önce akıllı filtreden geçirerek siyah/beyaz ayarını yapıyoruz
        img = akilli_resim_filtresi(file_path)
        
        # Türkçe karakterler için lang='tur' parametresi
        text = pytesseract.image_to_string(img, lang="tur")
        if not text.strip():
            return "⚠️ Resimden metin çıkarılamadı. Görüntü kalitesi düşük veya Tesseract Türkçe dil paketi eksik olabilir."
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        return (
            "❌ HATA: Tesseract OCR sistemde kurulu değil!\n\n"
            "Kurulum:\n"
            "  • Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-tur\n"
            "  • Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  • Mac: brew install tesseract\n\n"
            "Kurulumdan sonra Tesseract'ın PATH'de olduğundan emin olun."
        )
    except Exception as e:
        return f"❌ Resim OCR hatası: {str(e)}"


def process_uploaded_file(file_obj) -> str:
    """
    Yüklenen dosyayı tespit eder (PDF/Resim) ve uygun OCR yöntemiyle metin çıkarır.
    Gradio File component'inden gelen file_obj bir tempfile.NamedTemporaryFile benzeri objedir.
    """
    if file_obj is None:
        return ""

    # Gradio'dan gelen dosya objesi farklı formatlarda olabilir
    # Yeni Gradio sürümlerinde: file_obj.name veya file_obj bir dict olabilir
    if hasattr(file_obj, 'name'):
        file_path = file_obj.name
    elif isinstance(file_obj, str):
        file_path = file_obj
    elif isinstance(file_obj, dict) and 'name' in file_obj:
        file_path = file_obj['name']
    else:
        return f"❌ HATA: Dosya formatı tanınamadı. Tip: {type(file_obj)}"

    if not os.path.exists(file_path):
        return "❌ HATA: Dosya bulunamadı."

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
        return extract_text_from_image(file_path)
    else:
        return f"❌ HATA: Desteklenmeyen dosya formatı: {ext}. Lütfen PDF, PNG veya JPG yükleyin."


# ═══════════════════════════════════════════════════════════════
# ÖRNEK EVRAKLAR
# ═══════════════════════════════════════════════════════════════
ORNEK_EVRAKLAR = {
    "(Seçiniz)": "",
    "🎓 Yatay Geçiş Başvurusu (Üniversite)": """T.C.
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
Ahmet Yılmaz""",

    "🏗️ İmar Barışı Başvurusu (Belediye)": """T.C.
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
Mehmet Kaya""",

    "🏥 E-Nabız Bilgi Edinme (Sağlık Bakanlığı)": """T.C.
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
Elif Demir""",

    "📄 Emeklilik Talebi (SGK)": """T.C.
SOSYAL GÜVENLİK KURUMU İL MÜDÜRLÜĞÜNE

Sayı    : ...
Konu    : Emeklilik Talebi

Sayın Yetkililer,

Adım Fatma Şahin, T.C. kimlik numaram 98765432109. 1985 yılından bu yana 
özel sektörde çalışmaktayım. Son çalıştığım iş yerinden ayrıldım ve 
emeklilik hakkımın doğduğunu düşünüyorum.

Hizmet dökümü ve sigorta prim belgelerini ekte sunuyorum. 
Lütfen emeklilik başvurumun sonuçlandırılmasını talep ederim.

İmza
Fatma Şahin""",

    "💰 Borç Yapılandırma Talebi (Vergi Dairesi)": """T.C.
İSTANBUL VERGİ DAİRESİ BAŞKANLIĞINA

Sayı    : ...
Konu    : 6183 Sayılı Kanun Kapsamında Borç Yapılandırma Talebi

Sayın Yetkililer,

Adım Hasan Demir, T.C. kimlik numaram 55667788991, vergi kimlik numaram 1234567890. 
2023 ve 2024 yıllarına ait gelir vergisi borçlarımın toplam tutarı 45.750 TL'dir. 
Ekonomik zorluklar nedeniyle bu borçlarımı peşin ödeyemeyeceğimi belirtmek 
isterim.

6183 sayılı Amme Alacaklarının Tahsil Usulü Hakkında Kanun'un ilgili 
hükümleri uyarınca borçlarımın yapılandırılmasını talep ediyorum. 
Gelir beyannamelerinin örnekleri ve banka hesap ekstreleri ekte sunulmaktadır.

İmza
Hasan Demir""",

    "🏠 İpotek Fekki Talebi (Tapu Müdürlüğü)": """T.C.
KADIKÖY TAPU MÜDÜRLÜĞÜNE

Sayı    : ...
Konu    : İpotek Fekki Talebi

Sayın Yetkililer,

Adım Zeynep Korkmaz, T.C. kimlik numaram 99887766554. İlçenizde bulunan 
5678 ada, 12 parsel sayılı taşınmaz üzerinde kayıtlı olan ipoteğin, 
kredi borcunun tamamen kapatılmış olması nedeniyle fekkinin talep edilmesidir.

İlgili bankadan alınmış olan borç bitim yazısı ve tapu fotokopisi ekte 
iletilmektedir. Gerekli harç ve ücretleri ödemeye hazırım.

Bilgilerinize arz ederim.

İmza
Zeynep Korkmaz""",

    "⚖️ Bilirkişi Raporuna İtiraz (Adliye)": """T.C.
ANKARA 5. ASLİYE HUKUK MAHKEMESİ BAŞKANLIĞINA

Sayı    : ...
Konu    : Bilirkişi Raporuna İtiraz

Sayın Mahkeme,

Dava konusu 2023/456 E. sayılı dosyanızda bilirkişi tarafından düzenlenen 
rapora itiraz ediyorum. Bilirkişi raporunda taşınmazın değeri eksik 
hesaplanmış ve rayiç bedel göz ardı edilmiştir. Ayrıca raporda belirtilen 
hasar miktarı gerçeği yansıtmamaktadır.

Adım Selim Aydın, T.C. kimlik numaram 11223344556. Ekte sunulan 
tapu kayıt örnekleri ve ekspertiz raporları ile bilirkişi raporunun 
gözden geçirilmesini talep ederim.

İmza
Selim Aydın""",

    "🏫 Yurt Nakil Talebi (KYK)": """T.C.
KYK YÜKSEKÖĞRENİM KREDİ VE YURTLAR KURUMU MÜDÜRLÜĞÜNE

Sayı    : ...
Konu    : Yurt Nakil Talebi

Sayın Yetkililer,

Ben, Ayşe Yıldız, T.C. kimlik numaram 55443322110. 2024-2025 eğitim-öğretim 
yılında Ankara'da KYK yurdunda kalmaktayım. Öğrenci numaram 20241078'dir.

Ailemin İstanbul'a taşınması nedeniyle İstanbul'daki bir KYK yurduna 
nakil talep ediyorum. E-Devlet üzerinden alınan ikametgah belgesi ve 
aile durumunu gösteren belgeler ekte sunulmuştur.

Lütfen başvurumun değerlendirilmesini rica ederim.

İmza
Ayşe Yıldız""",

    "🐕 Sokak Hayvanları Şikayeti (CİMER / Valilik)": """T.C.
İSTANBUL VALİLİĞİ / CİMER BAŞVURUSU

Sayı    : ...
Konu    : Sokak Hayvanları ile İlgili Şikayet ve Talep

Sayın Valim,

İkamet ettiğimiz Kadıköy, Caferağa Mahallesi, Moda Caddesi üzerinde 
son zamanlarda çok sayıda başıboş köpek ve kedi görülmektedir. Hayvanlar 
yiyecek ararken çöp konteynerlerini dağıtmakta ve çevre kirliliğine 
neden olmaktadır. Ayrıca çocukların oyun alanlarına girmesi tehlike 
oluşturmaktadır.

Adım Burak Tan, T.C. kimlik numaram 66778899001. Belediye ekiplerinin 
bölgede düzenli kontrol ve besleme noktası oluşturmasını talep ediyorum. 
Fotoğraflar ve mahalle muhtarlığından alınan şikayet dilekçesi ekte 
iletilmektedir.

Bilgilerinize arz ederim.

İmza
Burak Tan""",
}

# ═══════════════════════════════════════════════════════════════
# PDF OLUŞTURMA FONKSİYONU
# ═══════════════════════════════════════════════════════════════
def create_pdf(metin):
    if FPDF is None:
        return None
    if not metin or metin.startswith("⚠️") or "Sonuç bulunamadı" in metin:
        return None
        
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Türkçe karakter desteği için Windows font yolları
        font_path = r"C:\Windows\Fonts\arial.ttf"
        font_path_bold = r"C:\Windows\Fonts\arialbd.ttf" # Kalın font dosyası eklendi
        has_font = os.path.exists(font_path)
        
        if has_font:
            # uni=True kaldırıldı, güncel fpdf2 standartlarına uyarlandı
            pdf.add_font("ArialTR", "", font_path)
            
            # Kalın font (Bold) dosyasını da kütüphaneye tanıtıyoruz
            if os.path.exists(font_path_bold):
                pdf.add_font("ArialTR", "B", font_path_bold)
                pdf.set_font("ArialTR", style="B", size=14)
            else:
                # Kalın dosya bulunamazsa normal boyutta büyütülmüş font kullan
                pdf.set_font("ArialTR", style="", size=14)
                
            # Başlık (Antet)
            pdf.cell(0, 10, txt="T.C.", ln=True, align="C")
            pdf.cell(0, 10, txt="KAMU EVRAK ANALİZ SİSTEMİ", ln=True, align="C")
            pdf.line(10, 30, 200, 30) # Şık bir alt çizgi
            pdf.cell(0, 15, txt="", ln=True) # Boşluk
            
            # İçerik metni için normal fonta (style="") geri dönüyoruz
            pdf.set_font("ArialTR", style="", size=11)
        else:
            pdf.set_font("Helvetica", size=11)
            
        # Metni satır satır pdf'e basma
        pdf.multi_cell(0, 6, txt=metin)
        
        dosya_adi = "Resmi_Yazi_Taslagi.pdf"
        pdf.output(dosya_adi)
        return dosya_adi
    except Exception as e:
        print(f"PDF Hatası: {e}")
        return None

def handle_pdf_export(metin):
    """Gradio butonuna bağlanacak yardımcı fonksiyon"""
    dosya_yolu = create_pdf(metin)
    if dosya_yolu:
        return gr.update(value=dosya_yolu, visible=True)
    return gr.update(visible=False)

# ═══════════════════════════════════════════════════════════════
# ANALİZ FONKSİYONU
# ═══════════════════════════════════════════════════════════════
def analyze_document(evrak_metni: str):
    """LangGraph pipeline'ını çalıştırır, 3 ajan çıktısını döndürür."""
    if not evrak_metni or len(evrak_metni.strip()) < 10:
        return (
            "⚠️ Lütfen analiz edilecek bir evrak metni girin veya dosya yükleyin.",
            "⚠️ Evrak metni girilmedi.",
            "⚠️ Evrak metni girilmedi."
        )

    initial_state: AgentState = {
        "evrak_metni": evrak_metni,
        "search_query": "",
        "siniflandirma": "",
        "retrieved_docs": [],
        "mevzuat_analizi": "",
        "taslaklar": "",
        "final_output": "",
    }

    graph = build_graph()
    result = graph.invoke(initial_state)

    # Ekrana basmadan önce Markdown (yıldız, diyez ve tire) işaretlerini temizleyen filtre
    def temizle(metin):
        if not isinstance(metin, str): return "Sonuç bulunamadı."
        return metin.replace("**", "").replace("### ", "").replace("## ", "").replace("*", "").replace("---", "")

    return (
        temizle(result.get("siniflandirma", "Sonuç bulunamadı.")),
        temizle(result.get("mevzuat_analizi", "Sonuç bulunamadı.")),
        temizle(result.get("taslaklar", "Sonuç bulunamadı."))
    )


def ornek_secim(secim: str):
    """Dropdown'dan seçilen örnek evrakı metin kutusuna doldurur."""
    return ORNEK_EVRAKLAR.get(secim, "")


# ═══════════════════════════════════════════════════════════════
# GRADIO ARAYÜZÜ
# ═══════════════════════════════════════════════════════════════
with gr.Blocks(
    title="TEKNOFEST RAG Agent - Kamu Evrak Analiz Sistemi",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
    ),
    css="""
        .container { max-width: 1400px !important; margin: auto; }
        .title { text-align: center; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }
        .subtitle { text-align: center; font-size: 0.95rem; color: #64748b; margin-bottom: 1.5rem; }
        .panel-header { font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem; }
        .analyze-btn { background: linear-gradient(135deg, #4f46e5, #7c3aed) !important; }
        .analyze-btn:hover { background: linear-gradient(135deg, #4338ca, #6d28d9) !important; }
        .output-box { font-family: 'Inter', system-ui, sans-serif; line-height: 1.7; max-height: 550px !important; overflow-y: auto !important; }
        .output-box textarea { overflow-y: auto !important; max-height: 550px !important; }
        .upload-box { border: 2px dashed #cbd5e1 !important; border-radius: 8px; }
        .upload-box:hover { border-color: #6366f1 !important; background: #f8fafc; }
    """
) as demo:

    # ── Başlık ──
    gr.HTML("""
        <div class="title">📋 Kamu Evrak Akıllı Analiz Sistemi</div>
        <div class="subtitle">
            TEKNOFEST Türkçe Yapay Zeka Dil Ajanları Yarışması | 
            LangGraph Çok Ajanlı RAG Pipeline | OCR + Vektör Arama + LLM
        </div>
    """)

    with gr.Row():
        # ═══════════════════════════════════════════════════════
        # SOL PANEL: Giriş (OCR + Metin + Örnekler)
        # ═══════════════════════════════════════════════════════
        with gr.Column(scale=1, min_width=450):
            # ── DOSYA YÜKLEME (OCR) ──
            gr.Markdown("### 📎 Evrak Yükle (OCR)")
            file_upload = gr.File(
                label="PDF, PNG veya JPG dosyası yükleyin",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                type="filepath",
                elem_classes="upload-box"
            )
            ocr_status = gr.Textbox(
                label="",
                placeholder="OCR durumu burada görünecek...",
                interactive=False,
                show_label=False,
                lines=1,
                visible=False
            )

            gr.Markdown("<div style='text-align:center; color:#94a3b8; font-size:0.8rem; margin-bottom:0.5rem;'>— VEYA —</div>")

            # ── METİN GİRİŞİ ──
            gr.Markdown("### 📝 Gelen Evrak Metni")
            evrak_input = gr.Textbox(
                label="",
                placeholder="Kuruma gelen dilekçe, başvuru veya resmî yazı metnini buraya yapıştırın...\n\nAyrıca yukarıdan PDF/Resim dosyası yükleyerek otomatik metin çıkarabilirsiniz.",
                lines=18,
                max_lines=30,
                show_label=False,
                elem_classes="output-box"
            )

            # ── ÖRNEK SEÇİMİ ──
            gr.Markdown("### 📂 Hızlı Örnek Evrak Seçimi")
            ornek_dropdown = gr.Dropdown(
                choices=list(ORNEK_EVRAKLAR.keys()),
                value="(Seçiniz)",
                label="",
                show_label=False,
                interactive=True
            )
            ornek_dropdown.change(
                fn=ornek_secim,
                inputs=ornek_dropdown,
                outputs=evrak_input
            )

            # ── ANALİZ BUTONU ──
            gr.Markdown("<br>")
            analyze_btn = gr.Button(
                "🔍 Evrakı Analiz Et",
                variant="primary",
                size="lg",
                elem_classes="analyze-btn"
            )

            # ── BİLGİ KUTUSU ──
            gr.Markdown("""
                <div style="margin-top:1rem; padding:0.75rem; background: var(--input-background-fill); border: 1px solid var(--border-color-primary); border-radius:8px; font-size:0.85rem; color: var(--body-text-color);">
                    <b>💡 Nasıl çalışır?</b><br>
                    1. <b>Dosya Yükle:</b> PDF/Resim yükleyin → OCR ile metin çıkarılır<br>
                    2. <b>Metin Gir:</b> Veya doğrudan evrak metni yazın / örnek seçin<br>
                    3. <b>Analiz Et:</b> 3 uzman ajan sırayla çalışır<br>
                    4. <b>Sonuçlar:</b> Sağ panelde sekme sekme inceleyin
                </div>
            """)

        # ═══════════════════════════════════════════════════════
        # SAĞ PANEL: Sekmeler (Çıktılar)
        # ═══════════════════════════════════════════════════════
        with gr.Column(scale=2, min_width=700):
            gr.Markdown("### 📊 Analiz Sonuçları")

            with gr.Tabs() as tabs:
                # ── Sekme 1: Sınıflandırma ──
                with gr.TabItem("🏷️ Sınıflandırma (Ajan 1)"):
                    gr.Markdown("*Evrak Analizcisi: Evrak türü, birim yönlendirmesi ve optimize edilmiş arama sorgusu*")
                    out_sinif = gr.Textbox(
                        label="",
                        lines=18,
                        max_lines=25,
                        show_label=False,
                        interactive=False,
                        elem_classes="output-box"
                    )

                # ── Sekme 2: Eksik Belge Raporu ──
                with gr.TabItem("📋 Eksik Belge Raporu (Ajan 2)"):
                    gr.Markdown("*Mevzuat Uzmanı: Qdrant vektör arama sonuçları ve eksiklik tespiti*")
                    out_eksik = gr.Textbox(
                        label="",
                        lines=18,
                        max_lines=25,
                        show_label=False,
                        interactive=False,
                        elem_classes="output-box"
                    )

                # ── Sekme 3: Resmî Yazı Taslakları ──
                with gr.TabItem("📜 Resmî Yazı Taslakları (Ajan 3)"):
                    gr.Markdown("*Raportör: Üst yazı, bilgilendirme notu ve eksik belge talep yazısı*")
                    out_taslak = gr.Textbox(
                        label="",
                        lines=18,
                        max_lines=25,
                        show_label=False,
                        interactive=False,
                        elem_classes="output-box"
                    )
                    gr.Markdown("<br>")
                    btn_pdf = gr.Button("📥 Bu Resmi Yazıyı PDF Olarak İndir", variant="secondary")
                    out_pdf_file = gr.File(label="📄 PDF Dosyanız Hazır", interactive=False, visible=False)

    # ── Event Bağlantıları ──
    # Dosya yüklendiğinde OCR çalıştır ve metin kutusuna yaz
    file_upload.change(
        fn=process_uploaded_file,
        inputs=file_upload,
        outputs=evrak_input
    )

    # Analiz butonu
    analyze_btn.click(
        fn=analyze_document,
        inputs=evrak_input,
        outputs=[out_sinif, out_eksik, out_taslak]
    )

    # PDF İndirme Butonu Aksiyonu
    btn_pdf.click(
        fn=handle_pdf_export,
        inputs=out_taslak,
        outputs=out_pdf_file
    )

    # ── Footer ──
    gr.HTML("""
        <div style="text-align:center; margin-top:1.5rem; padding-top:1rem; border-top:1px solid #e2e8f0; font-size:0.8rem; color:#94a3b8;">
            TEKNOFEST 2026 | Türkçe Yapay Zeka Dil Ajanları Yarışması | 
            Teknoloji: LangGraph + Qdrant + Ollama (Qwen 2.5) + SentenceTransformers + Tesseract OCR + PyMuPDF
        </div>
    """)


# ═══════════════════════════════════════════════════════════════
# BAŞLAT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
    )