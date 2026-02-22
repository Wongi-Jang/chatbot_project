from typing import TypedDict, Annotated, Any
import operator
from pydantic import BaseModel, Field


class ScoreSummary(BaseModel):
    score: str = Field(
        description='주어진 question,query,answer을 종합했을때 승인할지 말지. 통과면 "Pass" 아니면 "Fail"'
    )
    feedback: str = Field(
        description="vector DB를 재탐색할 query를 다시 작성하도록 기존 query가 무엇이 문제인지 피드백을 2~3문장으로 정리하세요."
    )


class RelevantSummary(BaseModel):
    # isrel: str = Field(description="관련성이 있으면 relevant, 없으면 irrelevant")
    # feedback: str = Field(description="irrelevant인 경우 관련성 부족 이유를 1~2문장으로 작성")
    isrel: list[str] = Field(
        description="단일 query에 대한 문서 관련성 라벨 목록. 형식: ['relevant'|'irrelevant', ...]"
    )
    relevant_feedback: str = Field(
        description="해당 query 문서가 모두 irrelevant인 경우에만 작성. 아니면 빈 문자열"
    )


class SupportSummary(BaseModel):
    issup: str = Field(
        description="answer가 docs에 기반하면 fully supported/partially supported/no support 중 하나로 판단"
    )
    feedback: str = Field(
        description="no support인 경우 왜 근거가 부족한지 1~2문장으로 작성"
    )


class FranchiseQuery(BaseModel):
    query: str = Field(description="프랜차이즈 DB 검색용 개별 쿼리 문자열")
    brand: str = Field(
        description="이 쿼리가 검색할 특정 브랜드명. 여러 브랜드를 비교할 때는 각 쿼리마다 해당 브랜드 하나만 지정하세요. (없으면 빈 문자열)",
        default="",
    )


class QuerySummary(BaseModel):
    law_query: list[str] = Field(
        description="법령 DB 검색용 쿼리 목록. 단일 법률 질문이라도 유의어/동의어를 활용해 같은 의미의 다른 쿼리 여러 개를 생성하세요."
    )
    franchise_query: list[FranchiseQuery] = Field(
        description="프랜차이즈 DB 검색용 쿼리 및 대상 브랜드 목록. 단일 질문이라도 유의어/동의어를 활용해 같은 의미의 다른 쿼리 여러 개를 생성하고, 비교 질문이라면 브랜드별로도 분리해야 합니다."
    )
    target_brands: list[str] = Field(
        description="질문에서 언급된 타겟 프랜차이즈 브랜드 이름 목록 (없으면 빈 리스트)",
        default_factory=list,
    )


class GraphState(TypedDict):
    question: Annotated[str, "user question"]
    # query: Annotated[dict, "retrieval queries by db type"]
    query: Annotated[dict[str, list[Any]], "retrieval queries by db type"]
    docs_raw: Annotated[list, "retrieved documents raw grouped by query"]
    # docs: Annotated[str, "retrieved docs grouped by query"]
    docs: Annotated[dict[str, str], "retrieved docs context grouped by query"]
    answer: Annotated[str, "agent answer"]
    do_retrieve: Annotated[bool, "should perform retrieval"]
    retrieve_types: Annotated[list[str], "target db types for retrieval"]
    db_meta: Annotated[dict, "vector db metadata summary"]
    # isrel: Annotated[str, "is doc rel to x? [relevant,irrelevant]"]
    isrel: Annotated[dict, "per-query per-doc relevance labels"]
    query_state: Annotated[dict, "per-query bool: has at least one relevant doc"]
    issup: Annotated[
        str, "is y supported by d? [fully supported,partially supported, no support]"
    ]
    # isuse: Annotated[str, "is y useful to x? [5,4,3,2,1]"]
    irrelevant_count: Annotated[int, "irrelevant routing count"]
    relevant_loop: Annotated[int, "relevant check loop count"]
    supporting_loop: Annotated[int, "supporting check loop count"]
    scoring_loop: Annotated[int, "scoring check loop count"]
    history: Annotated[list[dict[str]], operator.add]
    # feedback: Annotated[str, "feedback"]
    relevant_feedback: Annotated[
        dict[str, str], "query-level feedback from relevant node"
    ]
    supporting_feedback: Annotated[str, "text feedback from supporting/scoring nodes"]
    ispass: Annotated[str, "d"]
    no_docs: Annotated[bool, "no relevant docs available for current question"]
    target_brands: Annotated[list[str], "target brand names for franchise db filter"]
