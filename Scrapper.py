from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
import csv
import os
import logging

# Configuración del logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    filename='cabaprop_scraper.log')

def get_element_text(element, selector, label=""):
    """Obtiene el texto de un elemento utilizando un selector CSS."""
    try:
        return element.find_element(By.CSS_SELECTOR, selector).text.strip()
    except NoSuchElementException:
        logging.warning(f"No se encontró el elemento {label}.")
        return "No especificado"

def extract_global_elements(driver, selector, label):
    """Extrae todos los elementos globales visibles en la página utilizando un selector CSS."""
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
        )
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        return [element.text.strip() for element in elements]
    except Exception as e:
        logging.error(f"No se pudieron extraer los {label}: {e}")
        return []

SELECTORS = {
    'property_card': 'div.feat_property',
    'title': 'div.details h4',
    'property_type': 'div.details > a > div > p:first-of-type',
    'address': 'div.details > a > div > p',
    'zone': 'div.details p strong',
    'rooms': 'div.details ul li:nth-child(1) p',
    'bedrooms': 'div.details ul li:nth-child(2) p',
    'bathrooms': 'div.details ul li:nth-child(3) p',
    'total_area': 'div.details ul li:nth-child(4)',
    'built_area': 'div.details ul li:nth-child(5)',
    'price': 'span.lc-price-normal',
    'expenses': 'span.lc-price-small',
    'publication_date': 'span.fw400'
}

def extract_property_data(card):
    """Extrae los datos de una tarjeta de propiedad."""
    return {
        'Título': get_element_text(card, SELECTORS['title'], "Título"),
        'Tipo de Propiedad': get_element_text(card, SELECTORS['property_type'], "Tipo de Propiedad"),
        'Dirección': get_element_text(card, SELECTORS['address'], "Dirección"),
        'Zona': get_element_text(card, SELECTORS['zone'], "Zona"),
        'Número de Ambientes': get_element_text(card, SELECTORS['rooms'], "Número de Ambientes"),
        'Dormitorios': get_element_text(card, SELECTORS['bedrooms'], "Dormitorios"),
        'Número de Baños': get_element_text(card, SELECTORS['bathrooms'], "Número de Baños"),
        'Superficie Total': get_element_text(card, SELECTORS['total_area'], "Superficie Total"),
        'Superficie Construida': get_element_text(card, SELECTORS['built_area'], "Superficie Construida"),
        'Precio': "No especificado",
        'Expensas': "No especificado",
        'Fecha de Publicación': "No especificado"
    }

def scrape_cabaprop():
    """Inicia el scraper para extraer información de propiedades en todas las páginas de Cabaprop."""
    chromedriver_path = "C:\\Users\\artki\\sistema-inteligente-compra-venta-de-inmuebles-1\\chromedriver.exe"

    if not os.path.exists(chromedriver_path):
        logging.error(f"ChromeDriver not found at {chromedriver_path}")
        raise FileNotFoundError(f"ChromeDriver not found at {chromedriver_path}")

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    service = Service(chromedriver_path)
    driver = None
    csv_file = 'cabaprop_properties.csv'
    all_properties_data = []

    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(30)

        # Iterar sobre todas las páginas
        for page in range(1, 1867):
            url = f"https://cabaprop.com.ar/propiedades/comprar-casa-departamento?pagina={page}"
            logging.info(f"Navigating to {url}")
            driver.get(url)

            try:
                # Esperar hasta que las tarjetas estén presentes
                WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, SELECTORS['property_card']))
                )

                # Extraer las tarjetas
                property_cards = driver.find_elements(By.CSS_SELECTOR, SELECTORS['property_card'])

                # Extraer datos globales
                global_prices = extract_global_elements(driver, SELECTORS['price'], "precios")
                global_expenses = extract_global_elements(driver, SELECTORS['expenses'], "expensas")
                global_publication_dates = extract_global_elements(driver, SELECTORS['publication_date'], "fechas de publicación")

                for index, card in enumerate(property_cards):
                    property_data = extract_property_data(card)

                    # Asignar datos globales si están disponibles
                    if index < len(global_prices):
                        property_data['Precio'] = global_prices[index]
                    if index < len(global_expenses):
                        property_data['Expensas'] = global_expenses[index]
                    if index < len(global_publication_dates):
                        property_data['Fecha de Publicación'] = global_publication_dates[index]

                    all_properties_data.append(property_data)
                    logging.debug(f"Property data: {property_data}")

            except TimeoutException:
                logging.warning(f"No se pudo cargar la página {page}.")
                continue

        # Guardar los datos en un archivo CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as output_file:
            if all_properties_data:
                keys = all_properties_data[0].keys()
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(all_properties_data)
                logging.info(f"Saved {len(all_properties_data)} properties to {csv_file}")
            else:
                logging.warning("No properties to save.")

    except WebDriverException as e:
        logging.error(f"WebDriver error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    scrape_cabaprop()
