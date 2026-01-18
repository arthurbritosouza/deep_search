# 🔍 Deep Search - Sistema de Busca Inteligente com IA

Um sistema avançado de pesquisa e análise de informações que utiliza **LangGraph**, **Google Gemini** e **Tavily Search** para responder perguntas complexas através de buscas web inteligentes e processamento de linguagem natural.

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura do Sistema](#-arquitetura-do-sistema)
- [Fluxo de Execução](#-fluxo-de-execução)
- [Componentes Principais](#-componentes-principais)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Pré-requisitos](#-pré-requisitos)
- [Como Executar](#-como-executar)
- [Variáveis de Ambiente](#-variáveis-de-ambiente)
- [Exemplos de Uso](#-exemplos-de-uso)

---

## 🎯 Visão Geral

O **Deep Search** é um agente inteligente que:

1. **Recebe uma pergunta** do usuário
2. **Gera queries otimizadas** para busca web usando IA
3. **Busca informações** na web através do Tavily Search
4. **Extrai e analisa** o conteúdo das fontes encontradas
5. **Sintetiza** as informações em um contexto coerente
6. **Avalia** se o contexto é suficiente para responder
7. **Gera uma resposta** detalhada em formato Markdown ou **repete o ciclo** se necessário

## 🏗️ Arquitetura do Sistema

O sistema é construído usando **LangGraph**, que permite criar fluxos de trabalho complexos com agentes de IA.

### 🧩 Como os Componentes Interagem (Arquitetura)

Este diagrama mostra como o código (Python) interage com as APIs externas (Google e Tavily) para processar sua pergunta.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SEU COMPUTADOR (Docker)                       │
│                                                                         │
│  ┌──────────────┐      ┌───────────────┐      ┌──────────────────────┐  │
│  │   main.py    │ ───▶ │  workflow.py  │ ───▶ │       Agentes        │  │
│  │ (Entrada do  │      │ (Gerente do   │      │ (Cérebro do Sistema) │  │
│  │   Usuário)   │      │   Processo)   │      │                      │  │
│  └──────────────┘      └───────┬───────┘      └──────────┬───────────┘  │
│                                │                         │              │
└────────────────────────────────┼─────────────────────────┼──────────────┘
                                 │                         │
                                 ▼                         ▼
                      ┌──────────────────────┐   ┌──────────────────────┐
                      │    🌐 TAVILY API     │   │    🤖 GOOGLE API     │
                      │ (Busca na Internet)  │   │     (Inteligência)   │
                      └──────────────────────┘   └──────────────────────┘
```

---

## 🔄 Fluxo de Execução (O Cérebro do Agente)

Este diagrama explica como o sistema "pensa" e decide o que fazer passo a passo.

```
                   INÍCIO
                     │
                     ▼
      ┌─────────────────────────────┐
      │  1️⃣ BUSCAR NA WEB          │
      │  • IA cria perguntas        │
      │  • Busca no Tavily          │
      │  • Lê o conteúdo dos sites  │
      └──────────────┬──────────────┘
                     │
                     ▼
      ┌─────────────────────────────┐
      │  2️⃣ PROCESSAR CONTEÚDO     │
      │  • IA resume tudo o que leu │
      │  • Organiza as informações  │
      └──────────────┬──────────────┘
                     │
                     ▼
      ┌─────────────────────────────┐
      │  3️⃣ ANALISAR E DECIDIR     │
      │  • Tenho resposta boa?      │
      └──────┬───────────────┬──────┘
             │               │
      ┌──────▼──────┐  ┌─────▼──────┐
      │     SIM     │  │     NÃO    │
      │ (Suficiente)│  │ (Falta info)│
      └──────┬──────┘  └─────┬──────┘
             │               │
             ▼               │
    ┌─────────────────┐      │
    │ 📝 GERAR        │      │ (Volta para buscar
    │    RESPOSTA     │      │  mais coisas)
    └────────┬────────┘      │
             │               │
             ▼               │
            FIM              │
                             ▼
                    (Repete Etapa 1)
```

### 📄 **state.py** - Estado do Grafo

Define a estrutura de dados compartilhada entre todos os nós:

```python
class classState(TypedDict):
    questionUser: str                          # Pergunta original
    searchList: Annotated[list, operator.add] # Queries geradas
    sourceSearchTavily: Annotated[list, ...]  # URLs encontradas
    contentSearchTavily: Annotated[list, ...] # Conteúdo extraído
    summaryContent: str                        # Resumo consolidado
    context: str                               # Contexto estruturado
    should_repeat: bool                        # Flag de repetição
    responseGenerator: str                     # Resposta final
```

### 🤖 **agentes.py** - Funções de IA

Contém todas as funções que interagem com o Google Gemini:

| Função | Descrição |
|--------|-----------|
| `buildSearch()` | Gera 3 queries de busca otimizadas |
| `summaryContentSearch()` | Resume o conteúdo de múltiplos artigos |
| `createContext()` | Estrutura os resumos em um contexto coerente |
| `contextAnalysis()` | Avalia se o contexto é suficiente |
| `responseGenerator()` | Gera a resposta final formatada |

**Modelo utilizado:** `gemini-2.5-flash` (configurável)

### 🔍 **search.py** - Busca e Extração

Funções de busca e scraping:

| Função | Descrição |
|--------|-----------|
| `searchTavily()` | Busca usando Tavily Search API |
| `contentSource()` | Extrai conteúdo HTML com BeautifulSoup |

### 🌐 **workflow.py** - Orquestração

Define o grafo LangGraph e a lógica de roteamento:

- **Nós:** `search_web`, `context`, `analysis_and_response`
- **Roteamento condicional:** Decide entre finalizar ou repetir busca
- **Ponto de entrada:** `search_web`

### 🚀 **main.py** - Interface Principal

Ponto de entrada da aplicação:
- Recebe input do usuário
- Inicializa o estado
- Invoca o grafo
- Exibe a resposta

---

## 📁 Estrutura do Projeto

```
deep_search/
├── 📄 main.py                 # Ponto de entrada da aplicação
├── 🔄 workflow.py             # Definição do grafo LangGraph
├── 🤖 agentes.py              # Funções de IA (Gemini)
├── 🔍 search.py               # Funções de busca (Tavily) e scraping
├── 📊 state.py                # Definição do estado compartilhado
├── 🐳 Dockerfile              # Configuração do container Docker
├── 🐳 docker-compose.yml      # Orquestração Docker
├── 📦 pyproject.toml          # Dependências do projeto (uv)
├── 🔒 uv.lock                 # Lock file de dependências
├── 🚫 .dockerignore           # Arquivos ignorados pelo Docker
├── 🚫 .gitignore              # Arquivos ignorados pelo Git
├── 🔑 .env                    # Variáveis de ambiente (não versionado)
└── 📖 README.md               # Este arquivo
```

---

## 🛠️ Tecnologias Utilizadas

### Core
- **Python 3.12+** - Linguagem base
- **LangGraph 0.5.1** - Framework de agentes e workflows
- **LangChain 0.3.26** - Integração com LLMs

### IA e Busca
- **Google Gemini (gemini-2.5-flash)** - Modelo de linguagem
- **Tavily Search API** - Motor de busca otimizado para IA

### Web Scraping
- **httpx** - Cliente HTTP assíncrono
- **BeautifulSoup4** - Parser HTML
- **markdownify** - Conversão HTML → Markdown

### Infraestrutura
- **Docker** - Containerização
- **uv** - Gerenciador de pacotes Python ultrarrápido

---

## ✅ Pré-requisitos

Para executar este projeto, você precisa de:

### 1. **Docker & Docker Compose**
   - [Instalar Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Verificar instalação:
     ```bash
     docker --version
     docker compose version
     ```

### 2. **Chaves de API**
   - **Google Gemini API Key**
     - Acesse: https://makersuite.google.com/app/apikey
     - Crie uma chave gratuita
   
   - **Tavily API Key**
     - Acesse: https://tavily.com/
     - Registre-se e obtenha uma chave gratuita

---

## 🚀 Como Executar

### Passo 1: Clone o Repositório

```bash
git clone <seu-repositorio>
cd deep_search
```

### Passo 2: Configure as Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
# No Windows (PowerShell)
New-Item .env -ItemType File

# No Linux/Mac
touch .env
```

Edite o arquivo `.env` e adicione suas chaves:

```env
GOOGLE_API_KEY=sua_chave_do_google_gemini_aqui
TAVILY_API_KEY=sua_chave_do_tavily_aqui
```

⚠️ **IMPORTANTE:** Nunca compartilhe ou versione suas chaves de API!

### Passo 3: Construa a Imagem Docker

```bash
docker compose build
```

Este comando irá:
- Baixar a imagem base Python 3.12
- Instalar o `uv` package manager
- Instalar todas as dependências do `pyproject.toml`
- Configurar o ambiente virtual

**Tempo estimado:** 2-5 minutos (primeira execução)

### Passo 4: Execute a Aplicação

```bash
docker compose run --rm app
```

**Opções do comando:**
- `run` - Executa o container de forma interativa
- `--rm` - Remove o container automaticamente após a execução
- `app` - Nome do serviço definido no `docker-compose.yml`

### Passo 5: Faça sua Pergunta

Quando solicitado, digite sua pergunta:

```
Faça uma pergunta: Como funciona a arquitetura de containers Docker?
```

O sistema irá:
1. 🔍 Gerar queries de busca
2. 🌐 Buscar informações na web
3. 📥 Extrair conteúdo das fontes
4. 📝 Resumir e estruturar informações
5. 🧠 Analisar suficiência do contexto
6. ✅ Gerar resposta detalhada em Markdown

### Passo 6: Visualize a Resposta

A resposta será exibida no terminal formatada em Markdown.

---

## 🔧 Comandos Úteis

### Executar em modo detached (background)
```bash
docker compose up -d
```

### Ver logs em tempo real
```bash
docker compose logs -f app
```

### Parar todos os containers
```bash
docker compose down
```

### Reconstruir sem cache
```bash
docker compose build --no-cache
```

### Acessar o shell do container
```bash
docker compose run --rm app bash
```

### Instalar novas dependências
1. Adicione no `pyproject.toml`:
   ```toml
   dependencies = [
       "nova-biblioteca>=1.0.0",
   ]
   ```

2. Reconstrua a imagem:
   ```bash
   docker compose build
   ```

---

## 🔐 Variáveis de Ambiente

| Variável | Descrição | Obrigatória | Exemplo |
|----------|-----------|-------------|---------|
| `GOOGLE_API_KEY` | Chave da API do Google Gemini | ✅ Sim | `AIzaSy...` |
| `TAVILY_API_KEY` | Chave da API do Tavily Search | ✅ Sim | `tvly-...` |

### Como Obter as Chaves

#### Google Gemini API Key
1. Acesse [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Faça login com sua conta Google
3. Clique em "Create API Key"
4. Copie a chave gerada

#### Tavily API Key
1. Acesse [Tavily.com](https://tavily.com/)
2. Clique em "Get Started"
3. Registre-se com seu email
4. Acesse o dashboard e copie sua API key

---

## 📖 Exemplos de Uso

### Exemplo 1: Pesquisa Tecnológica

```
Faça uma pergunta: Quais são os últimos avanços em computação quântica?
```

**Saída esperada:** Resposta detalhada com:
- Últimas descobertas e breakthroughs
- Empresas e instituições envolvidas
- Aplicações práticas
- Desafios atuais

### Exemplo 2: Análise de Tendências

```
Faça uma pergunta: Como a inteligência artificial está impactando o mercado de trabalho em 2024?
```

**Saída esperada:** Análise completa sobre:
- Setores mais afetados
- Novas profissões emergentes
- Habilidades demandadas
- Perspectivas futuras

### Exemplo 3: Comparação de Tecnologias

```
Faça uma pergunta: Qual a diferença entre containers Docker e máquinas virtuais?
```

**Saída esperada:** Comparação estruturada com:
- Conceitos fundamentais
- Vantagens e desvantagens
- Casos de uso recomendados
- Performance e recursos

---

## 🐛 Solução de Problemas

### Erro: `ModuleNotFoundError`

**Causa:** Dependências não instaladas corretamente

**Solução:**
```bash
docker compose build --no-cache
```

### Erro: `API Key inválida`

**Causa:** Chaves de API incorretas ou ausentes no `.env`

**Solução:**
1. Verifique se o arquivo `.env` existe na raiz
2. Confirme se as chaves estão corretas
3. Reinicie o container

### Erro: `EOFError: EOF when reading a line`

**Causa:** Container executado em modo não-interativo

**Solução:**
Use `docker compose run --rm app` ao invés de `docker compose up`

### Timeout ao buscar fontes

**Causa:** Timeout de rede ao acessar URLs

**Solução:**
- Verifique sua conexão com a internet
- O sistema tentará continuar com as fontes que conseguiu acessar

---

## 🔄 Ciclo de Vida do Sistema (Simplificado)

(Removido para simplificar, consulte o Fluxo de Execução acima)

---

## 📈 Melhorias Futuras

- [ ] Interface web com Streamlit ou Gradio
- [ ] Cache de respostas para perguntas similares
- [ ] Suporte a múltiplos idiomas
- [ ] Histórico de conversas
- [ ] Exportação de respostas em PDF
- [ ] Integração com mais motores de busca
- [ ] Sistema de feedback sobre qualidade das respostas
- [ ] Métricas de performance e uso

---

## 📝 Licença

Este projeto é fornecido "como está", sem garantias de qualquer tipo.

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir novas funcionalidades
- Melhorar a documentação
- Submeter pull requests

---

## 📧 Contato

Para dúvidas, sugestões ou problemas, abra uma issue no repositório.

---

**Desenvolvido com ❤️ usando LangGraph, Google Gemini e Docker**
