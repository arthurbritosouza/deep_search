from duckduckgo_search import DDGS
from state import classState
from markdownify import markdownify as md
import httpx
import time
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()

def searchTavily(queryList):
    print("🌐 Iniciando a pesquisa na web com Tavily...")
    sources = []
    for query in queryList:
        print(f"🔍 Buscando por: '{query}'")
        tavily = TavilySearch(max_results=1)
        result = tavily.invoke(query)
        print("Resultados encontrados:", result)
        for item in result.get("results", []):
            if item.get("url") and item["url"] not in sources:
                sources.append(item["url"])
    print("✅ Fontes encontradas:")
    for d in sources:
        print(d)
    return sources

def contentSource(source):
    print("📥 Iniciando a coleta de conteúdo das fontes...")
    from bs4 import BeautifulSoup
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    content = []
    for url in source:
        try:
            with httpx.Client(timeout=10.0, headers=headers) as client:
                response = client.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                content.append(soup.get_text())
                print(f"✅ Conteúdo da fonte '{url}' coletado com sucesso.")
                time.sleep(5)  # Aguarde 5 segundos entre requisições
        except httpx.RequestError as e:
            print(f"❌ Erro ao acessar {url}: {e}")
        except Exception as e:
            print(f"❌ Erro ao processar o conteúdo da fonte '{url}': {e}")
    print("✅ Conteúdo de todas as fontes coletado com sucesso!\n")
    return content
# contentSource(['https://www.fcnoticias.com.br/os-avancos-mais-recentes-em-inteligencia-artificial-e-suas-aplicacoes/'])


