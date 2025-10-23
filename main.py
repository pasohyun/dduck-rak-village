# main.py
import streamlit as st
import utils # 헬퍼 파일 임포트

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="소상공인 리스크 분석 대시보드",
    page_icon="🏪",
    layout="wide"
)

# --- 모델 로드 시도 ---
# 앱이 시작될 때 utils.py의 함수를 호출해 모델을 미리 로드합니다.
model, features = utils.load_model_and_features()

if model is None or features is None:
    st.error(f"""
    모델 파일({utils.MODEL_PATH}) 또는 변수 파일({utils.FEATURES_PATH})을 찾을 수 없습니다.
    
    **솔루션:**
    1. `train_model.py` 스크립트를 먼저 실행하여 모델 파일(.pkl)과 변수 파일(.json)을 생성해주세요.
    2. `프랜차이즈.csv` 파일이 `train_model.py`와 같은 경로에 있는지 확인하세요.
    """)
else:
    # --- 메인 페이지 ---
    st.title("🏪 소상공인 리스크 분석 및 정책 제안")
    
    st.markdown("""
    본 대시보드는 LightGBM 머신러닝 모델을 기반으로
    소상공인(프랜차이즈)의 잠재적인 미래 리스크(휴/폐업)를 예측합니다.
    
    **왼쪽 사이드바에서 다음 단계를 진행하세요.**
    
    ### ➡️ 1. 정보 입력하기
    분석에 필요한 가게의 핵심 정보를 입력합니다.
    
    ### ➡️ 2. 분석 결과 보기
    입력된 정보를 바탕으로 모델의 예측 결과와 관련 정책을 확인합니다.
    
    ---
    *본 분석 결과는 통계적 예측이며, 실제 상황과 다를 수 있습니다.*
    """)