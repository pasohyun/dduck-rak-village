# pages/1_input_page.py
import streamlit as st

st.title("1. 귀하의 상점 정보 입력하기")

# --- 1. 사업자 형태 선택 ---
# 요청하신 선택 칸
biz_type = st.selectbox(
    "운영하시는 사업장 형태를 선택해주세요.",
    ("프랜차이즈", "개인영업"),
    help="현재는 '프랜차이즈' 모델만 분석 가능합니다."
)

if biz_type == "개인영업":
    st.warning("현재는 '프랜차이즈' 가맹점 분석 기능만 제공됩니다. '프랜차이즈'를 선택해주세요.")
    st.stop()

# --- 2. 변수 입력 폼 ---
# train_model.py에서 사용한 5개 변수를 입력받습니다.
st.markdown("---")
st.subheader("최근 1개월 기준 핵심 지표 입력")
st.markdown("""
SHAP 분석을 통해 검증된 **가장 영향력 있는 5가지 변수**를 입력해주세요.
각 변수의 의미는 SHAP 분석 결과와 동일합니다.
""")

with st.form(key="input_form"):
    # 1. RC_M1_UE_CUS_CN
    rc_m1_ue_cus_cn = st.number_input(
        label="① 유니크 고객 수 구간 (상위 %)",
        help="전체 가맹점 중 고객 수 순위입니다. (예: 10% 이하는 10, 25% 이하는 25)",
        min_value=0.0, max_value=100.0, value=50.0, step=1.0
    )
    
    # 2. M1_SME_RY_SAA_RAT
    m1_sme_ry_saa_rat = st.number_input(
        label="② 동일 업종 매출액 평균 대비 비율 (%)",
        help="동일 업종 평균 매출이 100일 때, 내 가게의 매출 수준입니다. (예: 평균보다 20% 높으면 120)",
        min_value=0.0, value=100.0, step=1.0
    )
    
    # 3. MCT_UE_CLN_NEW_RAT
    mct_ue_cln_new_rat = st.number_input(
        label="③ 신규 고객 비중 (%)",
        help="전체 고객 중 신규 방문 고객의 비율입니다. (예: 30%)",
        min_value=0.0, max_value=100.0, value=30.0, step=1.0
    )
    
    # 4. RC_M1_AV_NP_AT_CHG3M
    rc_m1_av_np_at_chg3m = st.number_input(
        label="④ 객단가 3개월 변화율 (%)",
        help="3개월 전 대비 최근 1개월 고객 1인당 평균 결제 금액의 변화율입니다. (예: 10% 감소 시 -10)",
        value=0.0, step=1.0
    )
    
    # 5. M1_SME_RY_CNT_RAT
    m1_sme_ry_cnt_rat = st.number_input(
        label="⑤ 동일 업종 결제건수 평균 대비 비율 (%)",
        help="동일 업종 평균 결제건수가 100일 때, 내 가게의 결제건수 수준입니다. (예: 평균보다 10% 낮으면 90)",
        min_value=0.0, value=100.0, step=1.0
    )
    
    # 제출 버튼
    submitted = st.form_submit_button("저장하고 분석 준비하기")

if submitted:
    # 제출된 정보를 st.session_state에 딕셔너리 형태로 저장
    st.session_state['user_input'] = {
        'RC_M1_UE_CUS_CN': rc_m1_ue_cus_cn,
        'M1_SME_RY_SAA_RAT': m1_sme_ry_saa_rat,
        'MCT_UE_CLN_NEW_RAT': mct_ue_cln_new_rat,
        'RC_M1_AV_NP_AT_CHG3M': rc_m1_av_np_at_chg3m,
        'M1_SME_RY_CNT_RAT': m1_sme_ry_cnt_rat
    }
    st.success("정보가 성공적으로 저장되었습니다.")
    st.markdown("---")
    st.markdown("### 👈 이제 왼쪽 사이드바에서 **'2. 분석 결과 보기'** 페이지로 이동하세요.")