import os
import time
import random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm

# Constants
BASE_URL = "https://www.angelone.in"
CATEGORY_URLS = [
    f"{BASE_URL}/support/add-and-withdraw-funds",
    f"{BASE_URL}/support/your-account",
    f"{BASE_URL}/support/portfolio-and-corporate-actions",
    f"{BASE_URL}/support/reports-and-statements",
    f"{BASE_URL}/support/your-orders",
    f"{BASE_URL}/support/charges-and-cashbacks",
    f"{BASE_URL}/support/angel-one-recommendations",
    f"{BASE_URL}/support/ipo-and-ofs",
    f"{BASE_URL}/support/loans"
]
SAVE_DIR = "data/web_content"
os.makedirs(SAVE_DIR, exist_ok=True)

def clean_filename(title):
    return "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in title).strip().replace(" ", "_")

def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def extract_articles_from_section(section):
    articles = []
    for block in section.find_all(["h1", "h2", "h3", "p", "li", "strong"]):
        text = block.get_text(strip=True)
        if text:
            articles.append(text)
    return "\n".join(articles)

def scrape_category_page(driver, category_url):
    print(f"🔍 Crawling category: {category_url}")
    driver.get(category_url)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "sidebar-section"))
        )
    except:
        print("⚠️ Timed out waiting for sidebar-section on", category_url)
        return []

    soup = BeautifulSoup(driver.page_source, "html.parser")
    sections = soup.find_all("div", class_="sidebar-section")
    all_articles = []

    for idx, section in enumerate(sections):
        content = extract_articles_from_section(section)
        if content.strip():
            all_articles.append((f"{category_url.split('/')[-1]}_{idx}", content))
    
    return all_articles

def run_scraper():
    driver = get_driver()
    all_scraped = []

    for category_url in CATEGORY_URLS:
        articles = scrape_category_page(driver, category_url)
        all_scraped.extend(articles)
        time.sleep(random.uniform(1.5, 3))

    print(f"\n✅ Total articles extracted: {len(all_scraped)}\n")

    for filename_suffix, content in tqdm(all_scraped):
        filename = clean_filename(filename_suffix[:100]) + ".txt"
        with open(os.path.join(SAVE_DIR, filename), "w", encoding="utf-8") as f:
            f.write(content)

    driver.quit()

if __name__ == "__main__":
    run_scraper()
