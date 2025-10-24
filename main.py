# main.py (수정된 코드)
import streamlit as st
import pandas as pd
import warnings
import utils # utils.py 임포트 (필수)

warnings.filterwarnings('ignore')

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="소상공인 리스크 분석 대시보드",
    page_icon="🏪",
    layout="wide"
)

st.title("🏪 소상공인 리스크 분석 대시보드")
st.markdown("---")

# --- 1. 개인정보 수집 이용 동의 ---
st.subheader("개인정보 수집 및 이용 동의") 
consent_text = """
**1. 수집하는 개인정보 항목**
본 서비스는 귀하가 입력하는 성함, 가맹점 구분번호를 수집합니다.
... (이하 약관 내용 동일) ...
"""
st.text_area("이용 약관", value=consent_text, height=250, disabled=True)

if 'consent_given' not in st.session_state:
    st.session_state['consent_given'] = False
agree = st.checkbox("위의 개인정보 수집 및 이용 약관에 모두 동의합니다.")
st.session_state['consent_given'] = agree
st.markdown("---")

# --- 2. 동의한 경우에만 앱의 핵심 기능 표시 ---
if st.session_state['consent_given']:

    # --- 3. 마스터 데이터 로드 ---
    @st.cache_data
    def load_master_data(biz_type):
        filename = "프랜차이즈.csv" if biz_type == "프랜차이즈" else "개인영업.csv"
        try:
            df = pd.read_csv(filename, encoding='cp949')
        except FileNotFoundError:
            return None # 파일이 없으면 None 반환
        except Exception as e:
            st.error(f"{filename} 로드 중 오류: {e}")
            return None
            
        df['ENCODED_MCT'] = df['ENCODED_MCT'].astype(str)
        df['MCT_NM'] = df['MCT_NM'].astype(str)
        df['TA_YM'] = pd.to_datetime(df['TA_YM'], errors='coerce')
        return df

    st.subheader("1. 상점 정보 조회")
    
    biz_type = st.selectbox(
        "① 사업장 형태를 선택하세요.",
        ("프랜차이즈", "개인영업"),
        index=0
    )
    
    # 선택에 따라 마스터 데이터 로드
    master_df = load_master_data(biz_type)
    
    # st.session_state에 biz_type 저장 (결과 페이지에서 사용)
    st.session_state['biz_type'] = biz_type

    if master_df is None:
        st.error(f"'{biz_type}.csv' 마스터 데이터 파일을 찾을 수 없습니다.")
        st.info(f"'data_preprocessing.py'를 실행하고, 그 다음 'final_model.py'를 실행했는지 확인하세요.")
        st.stop()

    # --- 4. 사용자 정보 입력 (폼) ---
    with st.form(key="lookup_form"):
        user_name = st.text_input("② 성함을 입력하세요.")
        merchant_id = st.text_input("③ 가맹점 구분번호 (ENCODED_MCT)")
        lookup_button = st.form_submit_button("내 상점 정보 조회하기")

    # --- 5. 조회 로직 실행 ---
    if lookup_button:
        if not user_name or not merchant_id:
            st.error("성함과 가맹점 구분번호를 정확히 입력해주세요.")
        else:
            id_to_find = merchant_id.strip()
            found_data = master_df[master_df['ENCODED_MCT'] == id_to_find]
            
            if found_data.empty:
                st.error(f"'{biz_type}.csv'에서 가맹점 구분번호 '{id_to_find}'를 찾을 수 없습니다.")
                st.warning("사업장 형태 또는 가맹점 구분번호를 다시 확인해주세요.")
                if 'store_data' in st.session_state:
                    del st.session_state['store_data']
            else:
                actual_name = found_data.sort_values(by='TA_YM', ascending=False)['MCT_NM'].iloc[0]
                st.success(f"'{actual_name}' ({id_to_find}) 상점 정보를 성공적으로 조회했습니다.")
                st.info(f"안녕하세요, {user_name}님. 아래에서 상점의 전체 이력을 확인하세요.")
                
                found_data_sorted = found_data.sort_values(by="TA_YM", ascending=False)
                st.subheader(f"'{actual_name}' 상점의 전체 이력 데이터 (최신순)")
                st.dataframe(found_data_sorted)
                
                # 조회된 데이터를 세션에 저장 (결과 페이지에서 사용)
                st.session_state['store_data'] = found_data_sorted
                
                st.markdown("---")
                st.markdown("### 👈 이제 왼쪽 사이드바에서 **'2. 분석 결과 보기'** 페이지로 이동하세요.")
    
    elif 'store_data' in st.session_state:
        st.info("이미 상점 정보 조회가 완료되었습니다. '2. 분석 결과 보기' 페이지로 이동하세요.")

# --- 6. 동의하지 않은 경우 ---
else:
    st.warning("🚨 서비스 이용을 위해 개인정보 수집 및 이용 약관에 동의해주세요.")
    if 'store_data' in st.session_state:
        del st.session_state['store_data']
    st.stop()