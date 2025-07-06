question = input("Faça uma pergunta: ")
initial_state = {
    "questionUser": question,
    "searchList": [],
    "sourceSearchTavily": [],
    "contentSearchTavily": [],
    "summaryContent": "",
    "context": "",
    "should_repeat": False,
    "responseGenerator": ""
}

from workflow import graph

from dotenv import load_dotenv
load_dotenv()

response = graph.invoke(initial_state)
print(response["responseGenerator"])
