# pages/2_result_page.py
import streamlit as st
import utils  # 헬퍼 파일 임포트
import time

st.title("2. 리스크 분석 결과 및 정책 제안")

# --- 1. 입력 데이터 확인 ---
if 'user_input' not in st.session_state:
    st.error("😭 먼저 '1. 정보 입력하기' 페이지에서 데이터를 입력하고 저장해주세요.")
    st.stop()

# --- 2. 모델 및 변수 로드 ---
model, features = utils.load_model_and_features()
if model is None or features is None:
    st.error("모델을 불러오는 데 실패했습니다. main.py 페이지의 오류 메시지를 확인하세요.")
    st.stop()

# --- 3. 예측 수행 (data_compare.py) ---
st.subheader("📊 리스크 분석 결과")

# 프로그레스 바 (시각적 효과)
progress_bar = st.progress(0, text="모델이 예측을 수행 중입니다...")
for perc_complete in range(100):
    time.sleep(0.01)
    progress_bar.progress(perc_complete + 1, text="모델이 예측을 수행 중입니다...")
time.sleep(0.5)
progress_bar.empty()

# utils.py의 예측 함수 호출
user_input = st.session_state['user_input']
risk_prob = utils.get_prediction(model, features, user_input)

if risk_prob is not None:
    risk_percent = risk_prob * 100
    
    # 위험도에 따라 색상 결정
    if risk_percent > 70:
        color = "error"
        level = "고위험군"
        help_text = "매우 높은 수준의 잠재적 위험(휴/폐업)이 감지되었습니다."
    elif risk_percent > 40:
        color = "warning"
        level = "주의군"
        help_text = "평균 이상의 잠재적 위험이 감지되었습니다. 지속적인 모니터링이 필요합니다."
    else:
        color = "success"
        level = "안정군"
        help_text = "현재 잠재적 위험이 낮은 안정적인 상태로 판단됩니다."

    # st.metric으로 결과 표시
    st.metric(
        label=f"미래 리스크 예측 확률 (분류: {level})",
        value=f"{risk_percent:.2f} %",
        delta=f"{risk_percent - 50:.2f} (중앙값 50% 대비)", # 50%를 임의의 중앙값으로 가정
        delta_color="inverse", # delta 값이 높을수록 나쁜 것이므로 'inverse'
        help=help_text
    )

    st.markdown("---")

    # --- 4. 변수 중요도 표시 ---
    st.subheader("📈 모델 예측 변수 중요도")
    try:
        st.image('feature_importance.png', caption='모델이 예측 시 중요하게 생각하는 변수들')
    except Exception as e:
        st.warning(f"변수 중요도 그래프를 불러오는 데 실패했습니다: {e}")
        st.info("먼저 'train_model.py'를 실행하여 'feature_importance.png' 파일을 생성해야 합니다.")

    st.markdown("---")

    # --- 5. 정책 제안 (policy_analysis.py) ---
    st.subheader("📇 맞춤형 정책 제안")
    utils.get_policy_link(risk_prob) # utils.py의 정책 링크 함수 호출

    # --- 6. 입력값 다시 보여주기 ---
    st.markdown("---")
    with st.expander("내가 입력한 데이터 확인하기"):
        st.json(user_input)

else:
    st.error("예측을 생성하는 데 문제가 발생했습니다.")