import json
import time
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

def parse_mevzuat_metni(html_content, kanun_no, tur):
    soup = BeautifulSoup(html_content, 'html.parser')
    maddeler = []
    current_madde = None
    current_metin = []
    paragraphs = soup.find_all('p', class_='MsoNormal')
    
    for p in paragraphs:
        text = p.get_text(strip=True)
        if not text:
            continue
        if text.startswith("MADDE"):
            if current_madde:
                maddeler.append({
                    "tur": tur,
                    "mevzuat_no": kanun_no,
                    "madde_no": current_madde,
                    "metin": " ".join(current_metin)
                })
            current_madde = text.split("-")[0].strip().replace('\n', ' ').replace('\r', ' ')
            current_metin = [text] 
        elif current_madde:
            current_metin.append(text)

    if current_madde:
        maddeler.append({
            "tur": tur,
            "mevzuat_no": kanun_no,
            "madde_no": current_madde,
            "metin": " ".join(current_metin)
        })
    return maddeler

def run_scraper():
    tum_mevzuat_verisi = []
    linkler_ve_nolar = []
    
    # Doğru hash'leri sitenin HTML'inden veya adres çubuğundan almalısın.
    # Önceki loglarında Yönetmelikler için "cumhurbaskanligiBakanlarKuruluYonetmelikleri" kullandığını görmüştüm.
    hedef_kategoriler = [
        "#kanunlar", 
        "#cumhurbaskanligiBakanlarKuruluYonetmelikleri", 
        "#kurumKurulusVeUniversiteYonetmelikleri", 
        "#cumhurbaskaniGenelgeleri"
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        for kategori in hedef_kategoriler:
            kategori_ismi = kategori.replace('#', '')
            print(f"\n--- {kategori_ismi} taranıyor ---")
            try:
                page.goto(f"https://mevzuat.gov.tr/{kategori}", timeout=60000)
                page.wait_for_timeout(2000) 
                
                # 14 buton arasından sadece 'görünür' olana tıkla
                page.locator("button#btnSearch:visible").click()
                
                # Tablo id'si ne olursa olsun, DataTables sınıfına sahip görünür tabloyu bekle
                page.wait_for_selector("table.dataTable:visible tbody tr", timeout=15000)
                
                # İsmi "_length" ile biten görünür menüden 100 satırı seç
                page.locator("select[name$='_length']:visible").select_option("100")
                page.wait_for_timeout(2000)
                
                sayfa_no = 1
                while True:
                    print(f"{kategori_ismi} - Tablo Sayfası {sayfa_no} okunuyor...")
                    
                    # Sadece görünür olan tablodan satırları çek
                    satirlar = page.query_selector_all("table.dataTable:visible tbody tr")
                    
                    for satir in satirlar:
                        link_element = satir.query_selector("a")
                        if link_element:
                            href = link_element.get_attribute("href")
                            if href and "mevzuat?" in href:
                                kanun_no_element = satir.query_selector("td")
                                kanun_no = kanun_no_element.inner_text().strip() if kanun_no_element else "Bilinmiyor"
                                linkler_ve_nolar.append({"href": href, "no": kanun_no, "tur": kategori_ismi})

                    hedef_sayfa = str(sayfa_no + 1)
                    
                    sonraki_sayfa_butonu = page.locator(f"a.page-link:visible:text-is('{hedef_sayfa}')")
                    
                    if sonraki_sayfa_butonu.count() == 0:
                        print(f"{kategori_ismi} kategorisinde son sayfaya ulaşıldı.")
                        break 
                        
                    sonraki_sayfa_butonu.click()
                    page.wait_for_timeout(3000)
                    sayfa_no += 1
            except Exception as e:
                print(f"{kategori_ismi} işlenirken hata oluştu veya bu sayfa yok: {e}")

        toplam_link = len(linkler_ve_nolar)
        print(f"\nToplam {toplam_link} adet mevzuat linki bulundu. Metinler çekiliyor...\n")

        # --- 2. AŞAMA: METİNLERİ KAZI ---
        for i, item in enumerate(linkler_ve_nolar, 1):
            iframe_url = f"https://mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe?{item['href'].split('?')[1]}"
            print(f"[{i}/{toplam_link}] İşleniyor: {item['tur']} - No {item['no']}")
            
            try:
                page.goto(iframe_url, timeout=30000)
                page.wait_for_selector("div.WordSection1", timeout=10000) 
                
                html_content = page.content()
                parcalanmis_maddeler = parse_mevzuat_metni(html_content, item['no'], item['tur'])
                tum_mevzuat_verisi.extend(parcalanmis_maddeler)
            except Exception as e:
                print(f"Hata oluştu (No {item['no']} atlanıyor)")
                
            time.sleep(0.3) 

        browser.close()

    # --- 3. AŞAMA: VERİYİ KAYDET ---
    with open('mevzuat_veriseti_tam.json', 'w', encoding='utf-8') as f:
        json.dump(tum_mevzuat_verisi, f, ensure_ascii=False, indent=4)
        
    print("\nŞAHANE! Tüm işlemler tamamlandı. Veriler kaydedildi.")

if __name__ == "__main__":
    run_scraper()