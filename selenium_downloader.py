from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import os

download_dir = os.path.join(os.getcwd(), "uyap_kararlar") 

chrome_options = Options()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,  
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True  
})
chrome_driver_path = "C:\\WebDriver\\chromedriver.exe"

service = Service(chrome_driver_path) 
driver = webdriver.Chrome(service=service, options=chrome_options)

driver.maximize_window()


driver.get("https://karararama.yargitay.gov.tr/")

time.sleep(5)


search_box = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
search_box.send_keys('"mali hukuk"') 
time.sleep(1)

search_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Ara')]")
search_button.click()
time.sleep(5)  


karar_listesi = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")


print(f"{len(karar_listesi)} karar bulundu.")


for index, karar in enumerate(karar_listesi):
    try:
        karar.click()  
        time.sleep(2)  

        pdf_buton = driver.find_element(By.XPATH, "//a[@onclick='kararSavePdf();']")
        pdf_buton.click()
        
        print(f"{index+1}. karar indirildi...")
        time.sleep(5)  
    except Exception as e:
        print(f"Hata: {e}")

driver.quit()
