# main.py (수정된 버전: 가맹점 번호로만 조회)

import streamlit as st
import pandas as pd
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

# --- 페이지 기본 설정 ---
try:
    icon = Image.open("새싹.png") # 👈 아이콘으로 사용할 png 파일 경로
except FileNotFoundError:
    icon = "🏪" # 파일을 못 찾을 경우 기본 이모티콘 사용

# --- 화면을 2개의 열로 분할 ---
col1, col2 = st.columns([1, 6]) # [이미지 너비 비율, 제목 너비 비율]

with col1:
    # 첫 번째 열에 이미지 표시
    st.image("새싹.png", width=80)

with col2:
    # 두 번째 열에 제목 표시 (이모티콘 제거)
    st.title("소상공인 리스크 분석 대시보드")

st.markdown("---")

# --- 1. 개인정보 수집 이용 동의 ---
st.subheader("개인정보 수집 및 이용 동의") # 글씨 크기 줄임

consent_text = """
**1. 수집하는 개인정보 항목**
본 서비스는 귀하가 입력하는 성함, 가맹점 구분번호를 수집합니다.

**2. 수집 및 이용 목적**
수집된 정보는 신한카드 데이터베이스(DB) 내의 상점 정보를 조회하고, 해당 상점의 과거 이력 데이터를 기반으로 미래 리스크(휴/폐업)를 예측하는 목적으로만 이용됩니다.

**3. 보유 및 이용 기간**
입력된 정보 및 조회된 데이터는 사용자의 브라우저 세션이 종료되면(웹페이지를 닫거나 새로고침 시) 즉시 파기되며, 서버에 별도로 저장되지 않습니다.

**4. 동의를 거부할 권리**
귀하는 위와 같은 개인정보 수집 및 이용에 동의하지 않을 수 있습니다. 단, 동의를 거부할 경우 본 리스크 분석 서비스를 이용하실 수 없습니다.
"""
st.text_area("이용 약관", value=consent_text, height=250, disabled=True)

# 동의 상태를 세션에 저장
if 'consent_given' not in st.session_state:
    st.session_state['consent_given'] = False

agree = st.checkbox("위의 개인정보 수집 및 이용 약관에 모두 동의합니다.")

if agree:
    st.session_state['consent_given'] = True
else:
    st.session_state['consent_given'] = False

st.markdown("---")


# --- 2. 동의한 경우에만 앱의 핵심 기능 표시 ---
if st.session_state['consent_given']:

    # --- 3. 마스터 데이터 로드 ---
    # @st.cache_data를 사용해 CSV 파일을 한 번만 읽어옵니다.
    @st.cache_data
    def load_master_data(biz_type):
        filename = "프랜차이즈.csv" if biz_type == "프랜차이즈" else "개인영업.csv"
        try:
            # cp949로 먼저 시도 (원본 데이터 인코딩)
            df = pd.read_csv(filename, encoding='cp949')
        except UnicodeDecodeError:
            df = pd.read_csv(filename, encoding='utf-8') # 실패 시 utf-8
        except FileNotFoundError:
            return None
        
        # 조회 키가 되는 컬럼들을 문자열로 통일 (매우 중요)
        df['ENCODED_MCT'] = df['ENCODED_MCT'].astype(str)
        df['MCT_NM'] = df['MCT_NM'].astype(str) # 가맹점명은 나중에 확인용으로 사용
        df['TA_YM'] = pd.to_datetime(df['TA_YM'], errors='coerce')
        return df

    st.subheader("1. 상점 정보 조회")
    
    # --- 4. 사용자 정보 입력 (폼) ---
    biz_type = st.selectbox(
        "① 사업장 형태를 선택하세요.",
        ("프랜차이즈", "개인영업"),
        index=0, # '프랜차이즈'를 기본값으로
        help="조회할 데이터베이스를 선택합니다."
    )
    
    # 선택한 biz_type에 따라 마스터 데이터 로드
    master_df = load_master_data(biz_type)

    if master_df is None:
        st.error(f"'{biz_type}.csv' 마스터 데이터 파일을 찾을 수 없습니다.")
        st.info(f"프로젝트 폴더에 '{biz_type}.csv' 파일이 있는지 확인해주세요.")
        st.stop()

    # 입력 폼
    with st.form(key="lookup_form"):
        user_name = st.text_input("② 성함을 입력하세요.")
        merchant_id = st.text_input("③ 가맹점 구분번호 (ENCODED_MCT)")
        
        lookup_button = st.form_submit_button("내 상점 정보 조회하기")

    # --- 5. 조회 로직 실행 ---
    if lookup_button:
        if not user_name or not merchant_id:
            st.error("성함과 가맹점 구분번호를 정확히 입력해주세요.")
        else:
            # 입력값 공백 제거
            id_to_find = merchant_id.strip()
            
            if master_df is not None:
                # 1. 가맹점 번호(ID)로 데이터 필터링
                found_data = master_df[master_df['ENCODED_MCT'] == id_to_find]
                
                if found_data.empty:
                    # 실패: ID가 없는 경우
                    st.error(f"가맹점 구분번호 '{id_to_find}'를 데이터에서 찾을 수 없습니다.")
                    st.warning("가맹점 구분번호를 다시 확인해주세요.")
                    if 'store_data' in st.session_state: # 기존 성공 기록 삭제
                        del st.session_state['store_data']
                else:
                    # 2. 조회 성공!
                    # ID가 존재하므로, 해당 ID의 최신 가맹점명을 DB에서 가져옴
                    actual_name = found_data.sort_values(by='TA_YM', ascending=False)['MCT_NM'].iloc[0]
                    
                    st.success(f"'{actual_name}' ({id_to_find}) 상점 정보를 성공적으로 조회했습니다.")
                    st.info(f"안녕하세요, {user_name}님. 아래에서 상점의 전체 이력을 확인하세요.")
                    
                    # 날짜순으로 정렬하여 전체 이력 표시
                    found_data_sorted = found_data.sort_values(by="TA_YM", ascending=False)
                    st.subheader(f"'{actual_name}' 상점의 전체 이력 데이터 (최신순)")
                    st.dataframe(found_data_sorted)
                    
                    # 조회된 데이터를 세션에 저장 (결과 페이지에서 사용)
                    st.session_state['store_data'] = found_data_sorted
                    st.session_state['franchise_id'] = id_to_find
                    st.session_state['biz_type'] = biz_type
                    
                    # 다음 단계 안내
                    st.markdown("---")
                    st.markdown("### 👈 이제 왼쪽 사이드바에서 **'result page'** 페이지로 이동하세요.")
            
    # 조회가 이미 성공한 상태에서 메인 페이지에 다시 접속한 경우
    elif 'store_data' in st.session_state:
        st.info("이미 상점 정보 조회가 완료되었습니다. 'result page' 페이지로 이동하세요.")
        with st.expander("조회된 데이터 다시 보기"):
            st.dataframe(st.session_state['store_data'])


# --- 6. 동의하지 않은 경우 ---
else:
    st.warning("🚨 서비스 이용을 위해 개인정보 수집 및 이용 약관에 동의해주세요.")
    
    # 동의 해제 시, 세션에 저장된 모든 데이터 삭제
    if 'store_data' in st.session_state:
        del st.session_state['store_data']
    
    st.stop() # 동의하지 않으면 앱의 나머지 부분 실행 중단