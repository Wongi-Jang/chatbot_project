from typing import TypedDict, Annotated
import operator
from pydantic import BaseModel, Field


class ScoreSummary(BaseModel):
    score: str = Field(
        description="주어진 question,query,answer을 종합했을때 승인할지 말지. 통과면 \"Pass\" 아니면 \"Fail\""
    )
    feedback: str = Field(
        description="vector DB를 재탐색할 query를 다시 작성하도록 기존 query가 무엇이 문제인지 피드백을 2~3문장으로 정리하세요."
    )


class RelevantSummary(BaseModel):
    isrel: str = Field(description="관련성이 있으면 relevant, 없으면 irrelevant")
    feedback: str = Field(description="irrelevant인 경우 관련성 부족 이유를 1~2문장으로 작성")


class SupportSummary(BaseModel):
    issup: str = Field(
        description="answer가 docs에 기반하면 fully supported/partially supported/no support 중 하나로 판단"
    )
    feedback: str = Field(description="no support인 경우 왜 근거가 부족한지 1~2문장으로 작성")


class GraphState(TypedDict):
    question: Annotated[str, "user question"]
    query: Annotated[dict, "retrieval queries by db type"]
    docs_raw: Annotated[list, "retrieved documents raw"]
    docs: Annotated[str, "retrieved docs"]
    answer: Annotated[str, "agent answer"]
    do_retrieve: Annotated[bool, "should perform retrieval"]
    retrieve_types: Annotated[list[str], "target db types for retrieval"]
    db_meta: Annotated[dict, "vector db metadata summary"]
    isrel: Annotated[str, "is doc rel to x? [relevant,irrelevant]"]
    issup: Annotated[str, "is y supported by d? [fully supported,partially supported, no support]"]
    isuse: Annotated[str, "is y useful to x? [5,4,3,2,1]"]
    irrelevant_count: Annotated[int, "irrelevant routing count"]
    relevant_loop: Annotated[int, "relevant check loop count"]
    supporting_loop: Annotated[int, "supporting check loop count"]
    scoring_loop: Annotated[int, "scoring check loop count"]
    history: Annotated[list[dict[str]], operator.add]
    feedback: Annotated[str, "feedback"]
    ispass: Annotated[str, "d"]
    no_docs: Annotated[bool, "no relevant docs available for current question"]
