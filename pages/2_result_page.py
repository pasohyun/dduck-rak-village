# pages/2_result_page.py (수정된 코드)
import streamlit as st
import pandas as pd
import utils  # 헬퍼 파일 임포트
import time

st.title("2. 리스크 분석 결과 및 정책 제안")

# --- 1. 업로드된 데이터 확인 ---
if 'user_data' not in st.session_state or 'biz_type' not in st.session_state:
    st.error("😭 먼저 '메인 페이지'에서 분석할 CSV 파일을 업로드해주세요.")
    st.stop()

# --- 2. 모델 및 변수 로드 ---
# biz_type에 따라 다른 모델을 로드할 수 있으나, 현재는 '프랜차이즈'만 로드
biz_type = st.session_state['biz_type']
user_df = st.session_state['user_data']

model, features = utils.load_model_and_features()

if model is None or features is None:
    st.error(f"모델 파일({utils.MODEL_PATH}) 또는 변수 파일({utils.FEATURES_PATH})을 불러오는 데 실패했습니다.")
    st.info("`train_model.py`를 실행하여 모델 파일이 정상적으로 생성되었는지 확인하세요.")
    st.stop()

st.subheader(f"📊 {biz_type} 상점 리스크 분석 결과")
st.markdown("업로드하신 데이터의 **가장 최근 월(TA_YM)을 기준**으로 분석을 수행합니다.")

# --- 3. 업로드된 데이터에서 최신 정보 추출 ---
try:
    # 1. TA_YM을 날짜 형식으로 변환 (오류 시 강제 변환)
    user_df['TA_YM'] = pd.to_datetime(user_df['TA_YM'], errors='coerce')
    
    # 2. 날짜 컬럼에 유효한 값이 있는지 확인
    if user_df['TA_YM'].isnull().all():
        st.error("업로드한 파일에 유효한 'TA_YM' 날짜 데이터가 없습니다.")
        st.stop()
        
    # 3. 가장 최신 월(TA_YM)의 데이터(행)를 찾습니다.
    latest_data_row = user_df.sort_values(by='TA_YM', ascending=False).iloc[0]
    
    # 4. 해당 행에서 모델에 필요한 5개의 변수 값을 딕셔너리로 추출
    input_dict = latest_data_row[features].to_dict()

except KeyError as e:
    st.error(f"업로드한 CSV 파일에 모델 분석에 필수적인 변수가 누락되었습니다: {e}")
    st.info(f"모델은 다음 변수들이 반드시 필요합니다: {features}")
    st.stop()
except Exception as e:
    st.error(f"업로드한 데이터를 처리하는 중 오류가 발생했습니다: {e}")
    st.info("CSV 파일 형식이 올바른지, 'TA_YM' 컬럼이 포함되어 있는지 확인하세요.")
    st.stop()

# --- 4. 예측 수행 ---
progress_bar = st.progress(0, text="모델이 예측을 수행 중입니다...")
for perc_complete in range(100):
    time.sleep(0.01)
    progress_bar.progress(perc_complete + 1, text="모델이 예측을 수행 중입니다...")
time.sleep(0.5)
progress_bar.empty()

# utils.py의 예측 함수 호출
risk_prob = utils.get_prediction(model, features, input_dict)

if risk_prob is not None:
    risk_percent = risk_prob * 100
    
    # 위험도에 따라 색상 결정 (이전과 동일)
    if risk_percent > 70:
        color = "error"; level = "고위험군"
        help_text = "매우 높은 수준의 잠재적 위험(휴/폐업)이 감지되었습니다."
    elif risk_percent > 40:
        color = "warning"; level = "주의군"
        help_text = "평균 이상의 잠재적 위험이 감지되었습니다. 지속적인 모니터링이 필요합니다."
    else:
        color = "success"; level = "안정군"
        help_text = "현재 잠재적 위험이 낮은 안정적인 상태로 판단됩니다."

    # st.metric으로 결과 표시
    st.metric(
        label=f"미래 리스크 예측 확률 (분류: {level})",
        value=f"{risk_percent:.2f} %",
        delta=f"{risk_percent - 50:.2f} (중앙값 50% 대비)",
        delta_color="inverse",
        help=help_text
    )

    st.markdown("---")

    # --- 5. 정책 제안 ---
    st.subheader("📇 맞춤형 정책 제안")
    utils.get_policy_link(risk_prob)

    # --- 6. 분석 근거 데이터 ---
    st.markdown("---")
    with st.expander("분석에 사용된 최신 데이터 확인하기 (입력값)"):
        st.json(input_dict)
        
    with st.expander("업로드한 전체 데이터 확인하기"):
        st.dataframe(user_df)

else:
    st.error("예측을 생성하는 데 문제가 발생했습니다.")