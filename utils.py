# utils.py
import streamlit as st
import joblib
import pandas as pd
import json

# =Setting: 모델과 변수 파일명
MODEL_PATH = "lgbm_franchise_model.pkl"
FEATURES_PATH = "model_features.json"

@st.cache_resource
def load_model_and_features():
    """
    앱 로딩 시 모델과 변수 리스트를 메모리에 올립니다.
    @st.cache_resource 데코레이터로 캐싱되어 한 번만 실행됩니다.
    (data_analysis.py의 '모델 로딩' 파트)
    """
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None, None
        
    try:
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            features = json.load(f)
    except FileNotFoundError:
        return None, None

    return model, features

def get_prediction(model, features, user_input_dict):
    """
    사용자 입력을 받아 모델 예측 확률을 반환합니다.
    (data_compare.py의 핵심 로직)
    """
    # 1. 사용자 입력을 DataFrame으로 변환
    input_df = pd.DataFrame([user_input_dict])
    
    # 2. 모델이 학습한 순서대로 컬럼을 재정렬 (매우 중요!)
    # fill_value=0은 혹시 모를 누락값을 0으로 채웁니다.
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # 3. 예측 수행 (0: 안전, 1: 위험)
    # [:, 1]은 '위험'할 확률(1)을 의미합니다.
    try:
        probability = model.predict_proba(input_df)[0, 1]
        return probability
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
        return None

def get_policy_link(risk_probability):
    """
    위험도에 따라 정책 링크나 정보를 반환합니다.
    (policy_analysis.py의 단순화된 버전)
    """
    if risk_probability > 0.7:
        st.subheader("⚠️ 고위험군 추천 정책")
        st.markdown("[소상공인시장진흥공단 '희망리턴패키지' (사업정리/재기 지원)](https://www.sbiz.or.kr/hop/hopa/hpba0100.do)")
    elif risk_probability > 0.4:
        st.subheader("🟡 주의군 추천 정책")
        st.markdown("[소상공인시장진흥공단 '경영개선지원' (컨설팅, 교육)](https://www.sbiz.or.kr/sup/supa/supa0100.do)")
    else:
        st.subheader("🟢 안정군 추천 정책")
        st.markdown("[소상공인시장진흥공단 '성장지원' (스마트기술 도입 등)](https://www.sbiz.or.kr/sup/supa/supa0300.do)")
        
    st.markdown("---")
    st.markdown("더 많은 정책 정보는 [소상공인마당](https://www.sbiz.or.kr/) 또는 [기업마당](https://www.bizinfo.go.kr/)에서 확인하세요.")