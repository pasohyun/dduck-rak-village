import argparse
from rag_pipeline import recommend_policies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--biz_type", required=True, help="예: 카페, 치킨집, 미용실")
    parser.add_argument("--region", required=True, help="예: 서울, 부산, 전국")
    parser.add_argument("--years", type=int, default=1)
    parser.add_argument("--employee_cnt", type=int, default=1)
    parser.add_argument("--issues", default="매출 감소")
    parser.add_argument("--sales", default="연매출 2억원, 최근 3개월 15% 감소")
    parser.add_argument("--needs", default="운전자금, 임대료, 마케팅")
    args = parser.parse_args()

    profile = vars(args)
    res = recommend_policies(profile)

    print("\n===== ANSWER =====\n")
    print(res["answer"])  # LLM 출력

    print("\n===== TOP HITS (for debugging) =====\n")
    for i, h in enumerate(res["hits"], 1):
        m = h["meta"]
        print(f"[{i}] {m.get('doc_id')} | {m.get('title')} | {m.get('region')} | {m.get('issuer')} | score={h.get('score')}")
