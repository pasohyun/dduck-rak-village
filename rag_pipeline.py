from typing import Dict, Any, List
from retriever import Retriever
from rerank import ReRanker
from llm import generate
from config import settings

POLICY_SCORING_GUIDE = """
[적합도 점수화 규칙]
- 대상 일치(업종/지역/업력/매출규모/특정 조건) 각 항목 0~2점, 최대 10점.
- 급성 이슈(매출 급락, 임대료 부담, 대출 한도 부족 등)에 대한 직접 효용: 0~3점.
- 신청 난이도(서류/심사/경쟁률 추정)에 따른 리스크: 0~2점 (난이도 높으면 감점).
- 총합 0~15점, 12점 이상은 "강추" 표시.
"""

SYS_PROMPT = f"""
너는 한국 소상공인 정책 추천 전문가다.
아래 컨텍스트(정책 발췌)만을 근거로, 사용자 프로필에 가장 적합한 정책을 3~5개 추천하라.
반드시 한국어로 답하고, 각 항목별로 점수 근거를 정량/정성 혼합으로 제시하라.
중요: 컨텍스트에 없는 정보는 추측하지 말고 '근거 부족'이라고 표기하라.
{POLICY_SCORING_GUIDE}
출력 형식:
1) 요약 한 줄
2) 추천 리스트 (정책명, 적합도 점수/15, 핵심 이유 2~3개, 신청링크, 출처 doc_id)
3) 준비물 체크리스트(공통/정책별)
4) 다음 액션 3가지
"""

USER_PROFILE_TEMPLATE = """
[사업자 프로필]
- 업종: {biz_type}
- 지역: {region}
- 업력: {years}년
- 직원수: {employee_cnt}명
- 최근 이슈: {issues}
- 매출 규모/변동: {sales}
- 필요 지원 유형(복수): {needs}
"""


def build_context(hits: List[Dict[str, Any]]):
    # 상위 문서들을 모아 citation friendly 컨텍스트 생성
    ctx_lines = []
    for h in hits:
        m = h["meta"]
        header = f"[doc_id={m.get('doc_id')} | {m.get('title')} | {m.get('region')} | {m.get('issuer')}]"
        ctx_lines.append(header)
        ctx_lines.append(h["doc"].strip())
        ctx_lines.append("\n")
    return "\n".join(ctx_lines)


def recommend_policies(profile: Dict[str, Any], k: int | None = None, filters: Dict[str, Any] | None = None):
    """profile keys: biz_type, region, years, employee_cnt, issues, sales, needs"""
    k = k or settings.top_k
    retriever = Retriever()

    # Query 생성 (instruction tuned for E5)
    q = (
        f"소상공인 정책 추천. 업종={profile.get('biz_type')}, 지역={profile.get('region')}, "
        f"업력={profile.get('years')}년, 직원수={profile.get('employee_cnt')}명, "
        f"이슈={profile.get('issues')}, 매출={profile.get('sales')}, 필요={profile.get('needs')}"
    )

    hits = retriever.search(q, filters=filters, top_k=k*2)  # 여유있게 뽑고

    # optional rerank
    rr = ReRanker()
    hits = rr.rerank(q, hits, top_k=k) if settings.use_rerank else hits[:k]

    context = build_context(hits)

    user_profile_txt = USER_PROFILE_TEMPLATE.format(**profile)

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_profile_txt + "\n\n[검색 컨텍스트]\n" + context}
    ]

    answer = generate(messages)
    return {
        "query": q,
        "hits": hits,
        "answer": answer,
    }
