from typing import Literal
from langgraph.graph import END, StateGraph
from state import classState
from agentes import *
from search import *

def searchWebNode(state: classState) -> classState:
    """Nó que fez uma pesquisa na web com base na pergunta do usuário."""
    queries = buildSearch(state['questionUser'])
    source_list = searchTavily(queries)
    content_source = contentSource(source_list)

    return {
        'searchList': queries,
        'sourceSearchTavily': source_list,
        'contentSearchTavily': content_source
    }
def contextNode(state: classState) -> classState:
    """Nó que constrói o contexto a partir do conteúdo pesquisado."""

    summary_content = summaryContentSearch(state['contentSearchTavily'])
    context  = createContext(summary_content)

    return {
        'summaryContent': summary_content,
        'context': context
    }
def AnalysAndResponseNode(state: classState) -> classState:
    """Nó que analisa o contexto e gera uma resposta."""

    context_analysis = contextAnalysis(state['questionUser'], state['context'])
    if context_analysis == "0" or context_analysis == 0:
        response_generator = responseGenerator(state['questionUser'], state['context'])
        should_repeat = False
    else:
        response_generator = ""
        should_repeat = True
    return {
        'responseGenerator': response_generator,
        'should_repeat': should_repeat,
    }

def direction(state: classState) -> Literal["search_web", "end"]:
    return "search_web" if state["should_repeat"] else "end"
    

graph_builder = StateGraph(classState)
graph_builder.add_node("search_web", searchWebNode)
graph_builder.add_node("context", contextNode)
graph_builder.add_node("analysis_and_response", AnalysAndResponseNode)

graph_builder.set_entry_point("search_web")
graph_builder.add_edge("search_web", "context")
graph_builder.add_edge("context", "analysis_and_response")

graph_builder.add_conditional_edges(
    "analysis_and_response",
    direction,
    {"search_web": "search_web", "end": END}
)

graph = graph_builder.compile()
