import json
import os
import datetime

# #region agent log
def log_debug(message, data=None):
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": message,
            "data": data,
            "sessionId": "debug-session"
        }
        with open("debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Logging failed: {e}")
# #endregion

print("Iniciando main.py...")
log_debug("Starting main.py execution")

question = input("Faça uma pergunta: ")
log_debug("Question received", {"question": question})

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

try:
    log_debug("Importing workflow...")
    from workflow import graph
    log_debug("Workflow imported successfully")
except ImportError as e:
    log_debug("ImportError importing workflow", {"error": str(e)})
    raise
except Exception as e:
    log_debug("Exception importing workflow", {"error": str(e)})
    raise

from dotenv import load_dotenv
load_dotenv()

try:
    log_debug("Invoking graph...")
    response = graph.invoke(initial_state)
    log_debug("Graph invocation complete", {"response_keys": list(response.keys()) if response else None})
    print(response["responseGenerator"])
except Exception as e:
    log_debug("Error during graph invocation", {"error": str(e)})
    print(f"Error: {e}")
