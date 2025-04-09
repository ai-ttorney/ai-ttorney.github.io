import glob
import json
import os
import time

import fitz  # PyMuPDF
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ------------------ Config ------------------

download_dir = r"D:\AITTORNEY"
output_jsonl_path = r"D:\AITTORNEY\ai-ttorney_kanunlar_3.jsonl"
os.makedirs(download_dir, exist_ok=True)

# ------------------ OpenAI Setup ------------------


class LawArticle(BaseModel):
    question: str = Field(description="Generate a relevant legal question.")
    explanation: str = Field(description="Provide a detailed explanation.")


OPENAI_API_KEY = ""  # api key here

parser = JsonOutputParser(pydantic_object=LawArticle)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.2, max_tokens=1000)

prompt = PromptTemplate(
    template=(
        "Generate a legal question and explanation based on the following Turkish financial law article. "
        "Provide precise and structured answers with legal context.\n{format_instructions}\n{context}"
    ),
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

# ------------------ PDF + GPT Functions ------------------


def wait_for_new_pdf(timeout=60):
    for _ in range(timeout):
        files = os.listdir(download_dir)
        crdownloads = [f for f in files if f.endswith(".crdownload")]
        pdfs = [f for f in files if f.endswith(".pdf")]

        if not crdownloads and pdfs:
            latest = max(
                [os.path.join(download_dir, f) for f in pdfs], key=os.path.getctime
            )
            return latest
        time.sleep(1)
    return None


def extract_madde_sections(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    lines = text.split("\n")
    madde_sections = {}
    current_madde = None
    current_text = []

    for line in lines:
        line = line.strip()
        if line.startswith("MADDE"):
            if current_madde:
                madde_sections[current_madde] = "\n".join(current_text)
            current_madde = line
            current_text = []
        elif current_madde:
            current_text.append(line)

    if current_madde:
        madde_sections[current_madde] = "\n".join(current_text)

    return madde_sections, text  # return both parsed and full text


def append_to_jsonl(question, explanation):
    entry = {
        "messages": [
            {
                "role": "system",
                "content": "AI-ttorney is a financial law advice chatbot and will only answer questions related to Turkey's financial law...",
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": explanation},
        ]
    }
    with open(output_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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

# ------------------ Lexpera: Search & Scroll ------------------

driver.get(
    "https://www.lexpera.com.tr/mevzuat/arama?p=lZJBi8IwEIX%2fisxZihX24q1WyhZDlSZ2DyISzFACYQuTeFDxvzvKlqpUdE8D733vJUNyAqlgAgJr650OtvmFIchMwGR9goKdpa6RpYoFiGFzHnaytMfOGnVe6lBTZsmHnuDdSdtUB6wbOiTGEHrf0nmuyuiOi0TyE82%2f59e5VeVgUQ5eIsvVVORpovJFwSg39pOFKN%2b2XZmnut4tKiTPUwYd9j5ul8i08%2fhBYNwGFO0%2f4b%2f%2bycej8YsryYZCZtGZ1q%2b0s4ZTFGb8MI%2fgzBLu%2fr7HDTbod4xszhc%3d"
)
time.sleep(3)

search_box = wait.until(
    EC.presence_of_element_located(
        (By.XPATH, "//input[contains(@placeholder, 'Bugün ne arıyorsunuz')]")
    )
)
search_box.send_keys('"finansal"')

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

# ------------------ Main Download + Q&A Loop ------------------

skip_count = 107
document_counter = skip_count

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
                print(f"✅ PDF indirildi: {downloaded_pdf}")
                madde_sections, full_text = extract_madde_sections(downloaded_pdf)

                if madde_sections:
                    for madde, content in madde_sections.items():
                        try:
                            response = chain.invoke({"context": content})
                            append_to_jsonl(
                                response["question"], response["explanation"]
                            )
                        except Exception as e:
                            print(f"❌ MADDE atlandı: {e}")
                else:
                    print("⚠️  MADDE bulunamadı, tüm belge gönderiliyor.")
                    try:
                        response = chain.invoke({"context": full_text})
                        append_to_jsonl(response["question"], response["explanation"])
                    except Exception as e:
                        print(f"❌ Belge tamamı işlenemedi: {e}")
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

    # "Daha fazla sonuç göster" pagination
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
    except Exception as e:
        print("🚫 'Daha fazla sonuç göster' bulunamadı, script durduruluyor.")
        break

driver.quit()
print(f"\n🎉 Tüm PDF'ler işlendi. JSONL dosyası: {output_jsonl_path}")
