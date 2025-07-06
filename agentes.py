from datetime import datetime
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from state import classState

load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat",  
    temperature=0.1,
    max_tokens=4000
)

def buildSearch(questionUser):
    print("🚀 Iniciando a construção das perguntas de pesquisa...")
    
    systemPrompt = """\
    # Papel:
    Você é um pesquisador experiente com a habilidade de criar perguntas de pesquisa claras e eficazes para buscas na web.

    # Objetivo
    Seu objetivo é gerar UMA pergunta de pesquisa com base na pergunta do usuário.

    # Contexto
    - A data atual é: {data_atual}

    # Atenção
    - A pergunta deve ser formulada em linguagem natural, como um humano digitaria no Google.
    - A pergunta deve ser uma string simples, sem aspas ou formatação especial.

    # Pergunta do usuário
    {questionUser}
    
    # Tarefa:
    Com base na pergunta do usuário, gere de 3 perguntas de pesquisa otimizadas para encontrar fontes de alta qualidade na web.
    """
    class searchBase(BaseModel):
        """
        Define a estrutura de dados para as perguntas de pesquisa geradas.
        """
        searchQuestions: list[str] = Field(
            description="Uma lista Python contendo 3 perguntas de pesquisa."
        )

    systemPrompt = systemPrompt.format(data_atual=datetime.now().strftime("%d/%m/%Y"), questionUser=questionUser)
    llm_structure = llm.with_structured_output(searchBase)
    result = llm_structure.invoke(systemPrompt) 
    print("✅ Perguntas de pesquisa construídas com sucesso!\n")
    return result.searchQuestions

def summaryContentSearch(contentList):
    print("📝 Iniciando a geração de resumos dos conteúdos...")
    systemPrompt = """\
    # Papel:
    Você é um assistente de pesquisa que deve resumir o conteúdo de artigos.

    # Tarefa:
    Gere um resumo conciso e informativo para cada artigo na lista de conteúdo.
    
    # Conteúdos:
    {content_list}
    """
    class summaryBase(BaseModel):
        summaries: str = Field(
            description="Retornar um resumo completo de todos os conteúdos que recebeu em seu no prompt, reusmo completo com todas as informações mais importantes de todos os artigos."
        )
    systemPrompt = systemPrompt.format(content_list=contentList)
    llm_structure = llm.with_structured_output(summaryBase)
    try:
        result = llm_structure.invoke(systemPrompt)
        if result is None:
            print("❌ ERRO: O LLM não retornou um resultado válido.")
            return "Não foi possível gerar o resumo."
        print("✅ Resumos gerados com sucesso!\n")
        return result.summaries
    except Exception as e:
        print(f"❌ Erro ao gerar resumo: {e}")
        return "Erro ao gerar resumo."

def createContext(summaryContents):
    print("🧠 Iniciando a criação do contexto a partir dos resumos...")
    systemPrompt = """\
    # Papel:
    Você é um analisador e criador de contextos para llms, você tem o papel de analisar resumos e criar um contexto completo e informativo a partir deles.

    # Tarefa:
    Gere um contexto completo e informativo a partir dos resumos fornecidos.
    
    # Resumos:
    {summary_contents}
    """
    class contextBase(BaseModel):
        context: str = Field(
            description="Retornar um contexto completo e informativo a partir dos resumos fornecidos."
        )
    systemPrompt = systemPrompt.format(summary_contents=summaryContents)
    llm_structure = llm.with_structured_output(contextBase)
    result = llm_structure.invoke(systemPrompt)
    print("✅ Contexto gerado com sucesso!\n")
    return result.context

def contextAnalysis(questionUser, context):
    print("🧠 Iniciando a análise do contexto...")
    systemPrompt = """\
    # Papel:
    Você é um analisador de contexto, você tem o papel de analisar o contexto fornecido e gerar uma análise completa e informativa.

    # Tarefa:
    Analisar todo o contexto fornecido, gerar uma análise, e verificar se com esse contexto é possível responder a pergunta do usuário.
    
    # Contexto:
    {context}

    # Pergunta do usuário:
    {questionUser}
    """
    class analysisBase(BaseModel):
        analysis: str = Field(
            description="Retorne '0' se o contexto for suficiente para responder à pergunta do usuário, e '1' se não for. A resposta deve ser apenas '0' ou '1'."
        )
    systemPrompt = systemPrompt.format(context=context,questionUser=questionUser)
    llm_structure = llm.with_structured_output(analysisBase)
    result = llm_structure.invoke(systemPrompt)
    print(f"✅ Análise do contexto concluída. Resultado: {result.analysis}\n")
    return result.analysis

def responseGenerator(questionUser, context):
    print("📝 Iniciando a geração da resposta final...")
    systemPrompt = """\
    # Papel:
    Você é um especialista em gerar respostas detalhadas e bem estruturadas. Sua principal função é analisar um contexto fornecido e, com base nele, responder à pergunta do usuário de forma completa e informativa, utilizando o formato Markdown para organizar a resposta.

    # Tarefa:
    Com base na pergunta do usuário e no contexto fornecido, gere uma resposta completa e informativa em formato Markdown. A resposta deve abordar todos os principais tópicos presentes no contexto, organizando-os de maneira lógica e clara.

    # Instruções de Formato (Markdown):
    - Utilize cabeçalhos (`#`, `##`, `###`) para estruturar os diferentes tópicos da resposta.
    - Empregue listas (`-` ou `*`) para enumerar pontos importantes, características ou exemplos.
    - Use negrito (`**texto**`) ou itálico (`*texto*`) para destacar termos-chave e conceitos relevantes.
    - Se o contexto contiver dados que possam ser apresentados em formato de tabela, organize-os dessa maneira para maior clareza.

    # Processo:
    1.  **Análise do Contexto:** Identifique os principais temas, argumentos e informações presentes no texto de contexto.
    2.  **Estruturação da Resposta:** Organize os tópicos identificados em uma sequência lógica que responda à pergunta do usuário da maneira mais eficaz.
    3.  **Elaboração do Conteúdo:** Desenvolva cada tópico, explicando os pontos importantes e citando as informações relevantes do contexto.
    4.  **Formatação em Markdown:** Aplique a formatação Markdown para garantir que a resposta seja clara, legível e bem organizada.

    # Pergunta do usuário:
    {questionUser}

    # Contexto:
    {context}
    """
    class responseBase(BaseModel):
        response: str = Field(
            description="Retorne uma resposta completa e informativa no formato md a partir da pergunta do usuário e do contexto fornecido."
        )
    systemPrompt = systemPrompt.format(context=context, questionUser=questionUser)
    llm_structure = llm.with_structured_output(responseBase)
    result = llm_structure.invoke(systemPrompt)
    return result.response


    