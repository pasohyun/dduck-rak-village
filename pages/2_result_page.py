# pages/2_result_page.py (수정된 코드)
import streamlit as st
import pandas as pd
import utils  # 헬퍼 파일 임포트
import time

st.title("2. 리스크 분석 결과")

# --- 1. 세션에서 데이터 가져오기 ---
if 'store_data' not in st.session_state or 'biz_type' not in st.session_state:
    st.error("😭 먼저 '메인 페이지'에서 상점 정보를 조회해주세요.")
    st.stop()

store_data = st.session_state['store_data'] # 조회된 전체 이력
biz_type = st.session_state['biz_type']     # '프랜차이즈' or '개인영업'

# --- 2. 모델 및 변수 로드 ---
models, features_lists = utils.load_models_and_features()

if models is None or features_lists is None:
    st.error("모델 로딩에 실패했습니다. 'final_model.py'를 실행했는지 확인하세요.")
    st.stop()

# 현재 업종에 맞는 모델과 변수 리스트 선택
try:
    current_model = models[biz_type]
    current_features = features_lists[biz_type]
except KeyError:
    st.error(f"'{biz_type}'에 해당하는 모델을 찾을 수 없습니다.")
    st.stop()

# --- 3. 최신 데이터(행) 추출 ---
try:
    latest_data_row = store_data.sort_values(by='TA_YM', ascending=False).iloc[0]
except (IndexError, KeyError):
    st.error("조회된 데이터에서 최신 정보를 추출할 수 없습니다.")
    st.stop()

st.header(f"'{latest_data_row['MCT_NM']}' 상점 분석 결과")
st.subheader(f"(기준: {latest_data_row['TA_YM'].strftime('%Y년 %m월')})")

# --- 4. 예측 수행 ---
with st.spinner('AI가 리스크를 분석 중입니다...'):
    time.sleep(1) # 시각적 효과
    risk_prob = utils.get_prediction(current_model, current_features, latest_data_row)

if risk_prob is None:
    st.error("리스크 확률을 계산하는 데 실패했습니다.")
    st.stop()

# --- 5. 위험도 3단계 표시 ---
risk_level, color = utils.get_risk_level(risk_prob)
risk_percent = risk_prob * 100

st.metric(
    label=f"미래 리스크 예측 확률 (분류: {risk_level})",
    value=f"{risk_percent:.2f} %",
    delta=f"{risk_percent - 50:.2f} (중앙값 50% 대비)",
    delta_color="inverse"
)


st.markdown("---")

# --- 6. TOP 3 중요 변수 (SHAP) ---
st.subheader("📈 리스크 예측 주요 요인 (TOP 3)")

with st.spinner("AI가 예측 근거를 분석 중입니다... (SHAP)"):
    top_3_features = utils.get_top_3_features(current_model, current_features, latest_data_row)

if top_3_features is not None and not top_3_features.empty:
    for i, (feature_name, shap_value) in enumerate(top_3_features.items()):
        
        # SHAP 값에 따라 긍정/부정 판단
        if shap_value > 0:
            impact_text = f"**위험도를 높이는**"
            st.error(f"**{i+1}순위 (위험 요인): `{feature_name}`**")
            st.markdown(f"    - 이 요인이 {impact_text} 방향으로 작용했습니다. (SHAP: {shap_value:+.3f})")
        else:
            impact_text = f"**위험도를 낮추는**"
            st.success(f"**{i+1}순위 (안전 요인): `{feature_name}`**")
            st.markdown(f"    - 이 요인이 {impact_text} 방향으로 작용했습니다. (SHAP: {shap_value:+.3f})")
else:
    st.warning("주요 요인을 분석하는 데 실패했습니다.")

st.markdown("---")

# --- 7. 시각화 (매출 & 인구) ---
st.subheader("📊 최근 6개월 상세 분석")

# 1. 최근 6개월 데이터 필터링
last_6m_data = store_data.sort_values(by='TA_YM', ascending=False).head(6).sort_values(by='TA_YM')
last_6m_data = last_6m_data.set_index('TA_YM') # 차트를 위해 TA_YM을 인덱스로

# (가정) 컬럼명 정의 - data_preprocessing.py와 일치해야 함
sales_col = 'RC_M1_SAA' # 매출금액
age_gender_cols = [
    'M12_MAL_1020_RAT', 'M12_MAL_3040_RAT', 'M12_MAL_5060_RAT',
    'M12_FME_1020_RAT', 'M12_FME_3040_RAT', 'M12_FME_5060_RAT'
]

# 2. 최근 6개월 매출 분석
if sales_col in last_6m_data.columns:
    st.markdown("#### ① 최근 6개월 매출 추이")
    st.line_chart(last_6m_data[sales_col], use_container_width=True)
else:
    st.warning(f"'{sales_col}' 컬럼이 없어 매출 분석을 표시할 수 없습니다.")

# 3. 최근 인구 분포 (가장 최신 월 기준)
if all(col in latest_data_row for col in age_gender_cols):
    st.markdown("#### ② 최신 고객 인구 통계 (성별/연령)")
    pop_data = latest_data_row[age_gender_cols]
    pop_df = pd.DataFrame({
        '비중 (%)': pop_data.values
    }, index=[
        '남성(10-20대)', '남성(30-40대)', '남성(50-60대)',
        '여성(10-20대)', '여성(30-40대)', '여성(50-60대)'
    ])
    st.bar_chart(pop_df, use_container_width=True)
else:
    st.warning(f"고객 통계 컬럼이 없어 인구 분석을 표시할 수 없습니다.")

st.markdown("---")

# --- 8. 정책 제안 ---
st.subheader("📇 맞춤형 정책 제안")
# (utils.py에 해당 함수가 필요합니다. 이전 버전의 utils.py에서 가져옴)
if 'get_policy_link' in dir(utils):
    utils.get_policy_link(risk_prob)
else:
    st.info("정책 제안 모듈이 로드되지 않았습니다.")