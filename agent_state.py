import operator
from typing import Annotated, Optional

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    task: dict
    best_score: Optional[float]
    best_exp: Optional[str]
    best_code: Optional[str]
    last_error: Optional[str]
    experiment_num: int
    fails_in_row: int
    nudges: int
    llm_failures: int
    history: Annotated[list, operator.add]
