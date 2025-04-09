import hashlib
import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ------------------ Config ------------------

keyword = "Vergi Usul Kanunu"
download_dir = os.path.join(
    r"D:\AITTORNEY", f"kanunlar_{keyword.replace(' ', '')}_pdfler"
)
os.makedirs(download_dir, exist_ok=True)

# ------------------ Hash Functions ------------------


def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def build_existing_hashes(root_dir):
    hashes = set()
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pdf"):
                full_path = os.path.join(root, file)
                try:
                    file_hash = calculate_md5(full_path)
                    hashes.add(file_hash)
                except Exception as e:
                    print(f"⚠️ Hash alınamadı: {full_path} | {e}")
    return hashes


existing_hashes = build_existing_hashes("D:\\AITTORNEY")

# ------------------ Wait for PDF ------------------


def wait_for_new_pdf(timeout=60):
    start_time = time.time()
    existing_files = set(os.listdir(download_dir))

    while time.time() - start_time < timeout:
        current_files = set(os.listdir(download_dir))
        new_pdfs = [f for f in (current_files - existing_files) if f.endswith(".pdf")]
        crdownloads = [f for f in current_files if f.endswith(".crdownload")]

        if new_pdfs and not crdownloads:
            latest_pdf = max(
                [os.path.join(download_dir, f) for f in new_pdfs],
                key=os.path.getctime,
            )
            return latest_pdf

        time.sleep(1)

    return None


# ------------------ Selenium Setup ------------------

chrome_options = Options()
chrome_options.add_argument("--start-maximized")
prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True,
}
chrome_options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(
    service=Service("C:\\WebDriver\\chromedriver.exe"), options=chrome_options
)
wait = WebDriverWait(driver, 15)

# ------------------ Lexpera PDF Downloader ------------------

driver.get(
    "https://www.lexpera.com.tr/mevzuat/arama?p=lZJBi8IwEIX%2fisxZihX24q1WyhZDlSZ2DyISzFACYQuTeFDxvzvKlqpUdE8D733vJUNyAqlgAgJr650OtvmFIchMwGR9goKdpa6RpYoFiGFzHnaytMfOGnVe6lBTZsmHnuDdSdtUB6wbOiTGEHrf0nmuyuiOi0TyE82%2f59e5VeVgUQ5eIsvVVORpovJFwSg39pOFKN%2b2XZmnut4tKiTPUwYd9j5ul8i08%2fhBYNwGFO0%2f4b%2f%2bycej8YsryYZCZtGZ1q%2b0s4ZTFGb8MI%2fgzBLu%2fr7HDTbod4xszhc%3d"
)
time.sleep(3)

search_box = wait.until(
    EC.presence_of_element_located(
        (By.XPATH, "//input[contains(@placeholder, 'Bugün ne arıyorsunuz')]")
    )
)
search_box.send_keys(f'"{keyword}"')

search_button = wait.until(EC.element_to_be_clickable((By.ID, "buInlineSearch")))
search_button.click()


def load_all_results():
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


load_all_results()
time.sleep(2)

# ------------------ Main Download Loop ------------------

document_counter = 120

while True:
    links = driver.find_elements(By.XPATH, "//a[contains(@class, 'document-link')]")
    total_links = len(links)
    print(f"\n📄 {total_links} belge bulundu. İşlenen: {document_counter}")

    for i in range(document_counter, total_links):
        try:
            print(f"\n▶ {i+1}. belgeye tıklanıyor...")

            links = driver.find_elements(
                By.XPATH, "//a[contains(@class, 'document-link')]"
            )
            driver.execute_script("arguments[0].click();", links[i])
            time.sleep(2)

            download_tab = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(@aria-label, 'İndirme seçenekleri')]")
                )
            )
            download_tab.click()
            time.sleep(1)

            pdf_link = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'PDF')]"))
            )
            pdf_url = pdf_link.get_attribute("href")
            print(f"⏳ PDF indiriliyor: {pdf_url}")

            pdf_link.click()
            time.sleep(5)

            downloaded_pdf = wait_for_new_pdf()
            if downloaded_pdf:
                file_hash = calculate_md5(downloaded_pdf)

                if file_hash in existing_hashes:
                    print(
                        f"⚠️ Aynı içerik daha önce indirilmiş, siliniyor: {downloaded_pdf}"
                    )
                    os.remove(downloaded_pdf)
                else:
                    print(f"✅ Yeni PDF indirildi: {downloaded_pdf}")
                    existing_hashes.add(file_hash)
            else:
                print("❌ PDF klasöre inmedi.")

            driver.back()
            time.sleep(3)
            document_counter += 1

        except Exception as e:
            print(f"❌ Hata ({i+1}. belge): {e}")
            driver.back()
            time.sleep(2)
            document_counter += 1

    try:
        show_more = driver.find_element(By.CLASS_NAME, "ShowEntireDocumentButton")
        if show_more.is_displayed():
            print("🔁 Daha fazla sonuç göster tıklanıyor...")
            driver.execute_script("arguments[0].click();", show_more)
            time.sleep(3)
            load_all_results()
        else:
            print("✅ Gösterilecek başka sonuç yok.")
            break
    except Exception:
        print("🚫 'Daha fazla sonuç göster' bulunamadı, script durduruluyor.")
        break

driver.quit()
print(f"\n🎉 Tüm PDF'ler indirildi. Klasör: {download_dir}")
