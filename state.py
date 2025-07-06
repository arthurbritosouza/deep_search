from typing import List, Annotated
from typing_extensions import TypedDict
import operator

class classState(TypedDict):
    questionUser: str
    searchList: Annotated[list, operator.add]
    sourceSearchTavily: Annotated[list, operator.add]
    contentSearchTavily: Annotated[list, operator.add]
    summaryContent: str
    context: str
    should_repeat: bool
    responseGenerator: str
