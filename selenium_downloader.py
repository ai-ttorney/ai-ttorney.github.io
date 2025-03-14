import os
import time
import json
import glob
import fitz  # PyMuPDF
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

#-----------------------------------SELENIUM KARAR İNDİRİCİ-----------------------------------

download_dir = os.path.join(os.getcwd(), "lexpera_kararlar")
os.makedirs(download_dir, exist_ok=True)

chrome_options = Options()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True
})
chrome_options.add_argument("--start-maximized")

chrome_driver_path = "C:\\WebDriver\\chromedriver.exe"
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

driver.get("https://www.lexpera.com.tr/ictihat/bolge-adliye-mahkemesi-istinaf-kararlari")
time.sleep(5)

try:
    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//input[contains(@placeholder, 'Bugün ne arıyorsunuz')]"))
    )
except Exception as e:
    print(f"Hata: Arama kutusu bulunamadı! {e}")
    driver.quit()
    exit(1)

search_box.send_keys('"mali hukuk"')
time.sleep(1)

try:
    search_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//a[@id='buInlineSearch']"))
    )
    search_button.click()
except Exception as e:
    print(f"Hata: Ara butonu tıklanamadı! {e}")
    driver.quit()
    exit(1)

time.sleep(5)

karar_listesi = driver.find_elements(By.XPATH, "//a[contains(@href, '/ictihat/bolge-adliye-mahkemesi')]")
print(f"{len(karar_listesi)} karar bulundu.")

def wait_for_download(directory, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        files = glob.glob(os.path.join(directory, "*.pdf"))
        if len(files) >= 1:  
            return True
        time.sleep(1)
    return False

for index in range(len(karar_listesi)):
    try:
        karar_listesi = driver.find_elements(By.XPATH, "//a[contains(@href, '/ictihat/bolge-adliye-mahkemesi')]")
        karar_listesi[index].click()
        time.sleep(2)

        try:
            indir_buton = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='İndirme seçenekleri']"))
            )
            indir_buton.click()
            time.sleep(2)

            pdf_buton = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@aria-label, 'PDF')]"))
            )
            pdf_buton.click()
        except Exception as e:
            print(f"Hata: İndir/PDF butonu tıklanamadı! {e}")

        if wait_for_download(download_dir):
            print(f"{index+1}. karar başarıyla indirildi.")
        else:
            print(f"{index+1}. kararın indirilmesi başarısız!")

        driver.back()
        time.sleep(3)

    except Exception as e:
        print(f"Hata: {e}")

time.sleep(5)
downloaded_files = glob.glob(os.path.join(download_dir, "*.pdf"))

if not downloaded_files:
    print("Hata: Hiç PDF inmemiş! Tarayıcıyı kapatmıyorum, hatayı manuel kontrol et.")
else:
    print(f"{len(downloaded_files)} PDF başarıyla indirildi.")
    driver.quit()

print("Selenium işlemi tamamlandı, PDF'ler başarıyla indirildi.")


#-----------------------------------LANGCHAIN PDF JSON DÖNÜŞTÜRÜCÜ-----------------------------------

class Document(BaseModel):
    son: str = Field(description="What is the result of this case? Give the penalty information.")
    seb: str = Field(description="What is the reason for this case? Explain the reason for this cases step by step and be precise.")
    summary: str = Field(description="Post summary")

OPENAI_API_KEY = "OPENAI-KEY"

parser = JsonOutputParser(pydantic_object=Document)

llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.2, max_tokens=2000)

prompt = PromptTemplate(
    template=(
        "Extract detailed and precise information according to the specified format. "
        "Provide comprehensive insights, including relevant legal contexts and practical implications. "
        "Avoid generic responses. \n{format_instructions}\n{context}"
    ),
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

pdf_files = glob.glob(os.path.join(download_dir, "*.pdf"))
print(f"{len(pdf_files)} adet PDF bulundu.")

def load_non_editable_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

output_jsonl_file = "output.jsonl"

for pdf_file in pdf_files:
    try:
        pdf_text = load_non_editable_pdf(pdf_file)
        response = chain.invoke({"context": pdf_text})

        jsonl_output = {
            "messages": [
                {"role": "system", "content": "AI-ttorney is a financial law advice chatbot and will only answer questions related to Turkey's financial law. For all other questions, it must respond: 'AI-ttorney can't answer the questions outside of Turkey's financial law.'"},
                {"role": "user", "content": response['seb']},
                {"role": "assistant", "content": response['son']}
            ]
        }

        with open(output_jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(jsonl_output, ensure_ascii=False) + "\n")

        print(f"{pdf_file} başarıyla JSON'a dönüştürüldü.")
    except Exception as e:
        print(f"Hata: {e} - Dosya: {pdf_file}")

print(f"Tüm PDF'ler {output_jsonl_file} dosyasına başarıyla dönüştürüldü.")
