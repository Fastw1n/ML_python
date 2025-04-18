from lxml import html
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import datetime
import locale
import time
from fake_useragent import UserAgent
import arff
from sklearn.preprocessing import OneHotEncoder



locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')

def preprocess_data(df):
    # Заменяем пропуски в "Growth" на 0
    df['Growth, %'] = df['Growth, %'].fillna(0).astype(float)

    # Заполняем пропуски в "Released"
    last_valid_released = None
    for i in range(len(df)):
        if pd.isna(df.loc[i, 'Released']):
            df.loc[i, 'Released'] = last_valid_released
        else:
            last_valid_released = df.loc[i, 'Released']

    # Обработка "ID", оставляем только числа
    df['ID'] = df['ID'].str.extract(r'(\d+)').astype(int)

    # Заменяем пропуски в "Subtheme" на "unknown"
    df['Subtheme'] = df['Subtheme'].fillna('unknown')

    return df
 
def scrape_all_minifigs(filename):
    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(options=options)

    url = "https://www.brickeconomy.com/minifigs/theme/star-wars"
    driver.get(url)

    all_links = [] 

    while True:
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH, '//td[@class="ctlminifigs-left"]/div[1]/h4/a'))
            )

            links = driver.find_elements(By.XPATH, '//td[@class="ctlminifigs-left"]/div[1]/h4/a')
            for link in links:
                href = link.get_attribute('href')
                if href not in all_links: 
                    all_links.append(href)

            print(f"Собрано {len(all_links)} ссылок...")

            next_button = driver.find_element(By.XPATH, '//a[contains(@class, "page-link") and text()="Next"]')
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(3)
            if len(all_links) == 1399:
                break
        except Exception as e:
            print(f"Ошибка: {e}")
            break

    driver.quit()

    with open(filename, "w") as file:
        file.write("\n".join(all_links))

    print(f"Собрано {len(all_links)} ссылок. Ссылки сохранены в 'minifigs_links_all.txt'.")

def parse_minifig_link(driver, link):
    time.sleep(1)
    result = dict()
    flat_id = link.split('/')[-2]
    result = {'ID': [flat_id]}

    def getValid(name, data, default=''):
        result[name] = data
        print(f"{name} {data}")
        
    driver.get(link)
    

    character_element = driver.find_element(By.XPATH, "//div[@class='side-box-body']/div[2]/div[@class = 'col-xs-7']")
    character = character_element.text
    getValid('Name', str(character).lower())

    try:
        
        subtheme_element = driver.find_element(By.XPATH, "//div[@class='side-box-body']//div[contains(@class, 'row rowlist')][.//div[contains(text(), 'Subtheme')]]//div[@class='col-xs-7']/a")
        subtheme = subtheme_element.text
        getValid('Subtheme', str(subtheme).lower())
    except Exception as e:
        print(f"Ошибка при подключении к {link}: {"нет элемента: Subtheme"}")    

    try:
        year_element = driver.find_element(By.XPATH, "//div[@class='side-box-body']//div[contains(@class, 'row rowlist')][.//div[contains(text(), 'Year')]]//div[@class='col-xs-7']")
        year = year_element.text
        getValid('Year', year)
    except Exception as e:
        print(f"Ошибка при подключении к {link}: {"нет элемента: Year"}") 

    try:
        rel_element = driver.find_element(By.XPATH, "//div[@class='side-box-body']//div[contains(@class, 'row rowlist')][.//div[contains(text(), 'Released')]]//div[@class='col-xs-7']")
        rel = rel_element.text
        getValid('Released', str(rel).lower())    
    except Exception as e:
        print(f"Ошибка при подключении к {link}: {"нет элемента: Released"}") 

    try:
        set_element = driver.find_element(By.XPATH, "//div[@class='side-box-body']//div[contains(@class, 'row rowlist')][.//div[contains(text(), 'Sets')]]//div[@class='col-xs-7']/span[1]")
        set = set_element.text
        getValid('Sets', set)      
    except Exception as e:
        print(f"Ошибка при подключении к {link}: {"нет элемента: Sets"}") 

    try:
        value_element = driver.find_element(By.XPATH, "//div[@id='ContentPlaceHolder1_UpdatePanelMinifigPricing']/div[1]/div[2]/div[1]/div[2]")
        value = value_element.text
        getValid('Value, €', str(value[:-2]).replace(',', '.'))
    except Exception as e:
        print(f"Ошибка при подключении к {link}: {"нет элемента: Value"}") 

    try:
        growth_element = driver.find_element(By.XPATH, "//div[@id='ContentPlaceHolder1_UpdatePanelMinifigPricing']//div[@class='col-xs-7']//span[contains(@class, 'label-value-up')]")
        growth = growth_element.text
        if growth is None:
            getValid('Growth, %', growth)   
        else:
            getValid('Growth, %', str(growth)[1:-1]) 

    except Exception as e:
        print(f"Ошибка при подключении к {link}: {"нет элемента: Growth"}")
    
    return result
    
def main():
    minifigs_links_path = 'minifigs_links_all.txt'
    try:
        with open(minifigs_links_path, 'r') as flat_links_file:
            links = flat_links_file.readlines()
    except FileNotFoundError:
        links = []

    if not links:
        scrape_all_minifigs(minifigs_links_path)
        with open(minifigs_links_path, 'r') as flat_links_file:
            links = flat_links_file.readlines()

    pd.set_option('max_colwidth', 100)
    pd.set_option('display.width', 100)
    df = pd.DataFrame(columns=['ID', 'Sets', 'Value, €', 'Growth, %','Year' ,'Released' ,'Subtheme', 'Name'])

    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=options)
    stealth(
        driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )

    try:
        for link in links:
            d = parse_minifig_link(driver, link.strip())
            if d is not None:
                df = pd.concat([df, pd.DataFrame(d)], ignore_index=True)
    finally:
        
        driver.quit()

    
    column_widths = {col: max(len(str(x)) for x in df[col]) + 2 for col in df.columns}

    def format_row_dynamic(row):
        return '\t'.join(str(row[col]).ljust(column_widths[col]) for col in df.columns)


    with open('table.tsv', 'w', encoding='utf-8') as f:
        f.write('\t'.join(col.ljust(column_widths[col]) for col in df.columns) + '\n')
        for _, row in df.iterrows():
            f.write(format_row_dynamic(row) + '\n')

    def save_to_arff(df, filename):
    # Определяем типы признаков
        attribute_types = {
            'ID': 'STRING',
            'Name': 'STRING',
            'Subtheme': 'STRING',
            'Year': 'STRING',
            'Released': 'STRING',
            'Sets': 'NUMERIC',
            'Value, €': 'NUMERIC',
            'Growth, %': 'NUMERIC'
        }

        # Формируем атрибуты
        attributes = [(col, attribute_types[col]) for col in df.columns]

        # Подготавливаем данные
        data = df.fillna('').values.tolist()

        # Создаем объект ARFF
        arff_data = {
            'description': 'Star Wars Minifigures dataset',
            'relation': 'minifigs',
            'attributes': attributes,
            'data': data
        }

        # Сохраняем файл
        with open(filename, 'w', encoding='utf-8') as f:
            arff.dump(arff_data, f)

    print(f"Данные успешно сохранены в ARFF файл: {'minifigs_data.arff'}")

    save_to_arff(df, 'minifigs_data.arff')
    df_preprocessed = preprocess_data(df)
    save_to_arff(df_preprocessed, 'minifigs_data__preprocessed.arff')


if __name__ == "__main__":
    main()