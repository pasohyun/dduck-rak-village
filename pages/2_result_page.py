# pages/2. 분석 결과 보기.py (최종 완성본 + 모든 시각화)

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import warnings
import ollama
import xgboost as xgb
import seaborn as sns

from langchain_community.document_loaders import DataFrameLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

warnings.filterwarnings('ignore')

# ==============================================================================
# ⭐️ 세션 상태 확인
# ==============================================================================
if 'store_data' not in st.session_state:
    st.warning("👈 먼저 '1. 상점 정보 조회' 페이지에서 상점 정보를 조회해주세요.")
    st.info("조회가 성공적으로 완료되면, 이 페이지에서 리포트를 볼 수 있습니다.")
    st.stop()

# ==============================================================================
# 🔨 핵심 기능 함수
# ==============================================================================
# 마스터 데이터를 로드하는 함수 (결과 페이지에서도 재사용)
@st.cache_data
def load_master_data(biz_type):
    filename = "프랜차이즈.csv" if biz_type == "프랜차이즈" else "개인영업.csv"
    try:
        df = pd.read_csv(filename, encoding='cp949')
    except UnicodeDecodeError:
        df = pd.read_csv(filename, encoding='utf-8')
    except FileNotFoundError:
        return None
    df['TA_YM'] = pd.to_datetime(df['TA_YM'], errors='coerce')
    if 'HPSN_MCT_ZCD_NM' in df.columns:
         df['HPSN_MCT_ZCD_NM'] = df['HPSN_MCT_ZCD_NM'].astype(str)
    if 'RC_M1_SAA' in df.columns:
         df['RC_M1_SAA'] = pd.to_numeric(df['RC_M1_SAA'], errors='coerce')
    return df

@st.cache_resource
def build_rag_system():
    try:
        policy_df = pd.read_csv('expanded_policy_data.csv')
        policy_df['document_text'] = policy_df.apply(lambda row: f"정책명: {row['title']}\n기관: {row['issuer']}\n지역: {row['region']}\n대상: {row['eligibility_text']}\n혜택: {row['benefit_text']}", axis=1)
        loader = DataFrameLoader(policy_df, page_content_column="document_text")
        documents = loader.load()
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(documents, embedding_function)
        st.success("LLM 정책 추천 시스템이 성공적으로 준비되었습니다.")
        return vector_db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"정책 추천 시스템 준비 중 오류 발생: {e}")
        return None

def predict_risk_with_ensemble(model_package, store_data_row):
    # --- 1. 데이터 전처리 ---
    X = store_data_row.drop(columns=['ENCODED_MCT', 'TA_YM', 'Unnamed: 0'], errors='ignore')

    categorical_features = ['Delivery_Group_Code', 'LIFE_AREA', "HPSN_MCT_ZCD_NM"]
    for cat_col in categorical_features:
        if cat_col in X.columns:
            X[cat_col] = X[cat_col].fillna('Unknown').astype("category")

    numeric_cols = X.select_dtypes(include=np.number).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)

    # --- 2. pkl 파일에서 모든 구성 요소 추출 ---
    xgb_model = model_package["xgb"]
    cat_model = model_package["cat"]
    w_xgb, w_cat = model_package["weights"]
    feature_order = model_package["feature_order"]

    # --- 3. XGBoost 예측 ---
    xgb_feats = feature_order["xgb"]
    X_test_xgb = X[xgb_feats]
    dtest = xgb.DMatrix(X_test_xgb, enable_categorical=True)
    xgb_prob = xgb_model.predict(dtest)

    # --- 4. CatBoost 예측 ---
    cat_feats = feature_order["cat"]
    X_test_cat = X[cat_feats]
    cat_prob = cat_model.predict_proba(X_test_cat)[:, 1]

    # --- 5. 가중 평균 앙상블 ---
    ens_prob = (w_xgb * xgb_prob) + (w_cat * cat_prob)
    
    return ens_prob[0]

def generate_visualizations(store_history_df, master_df):
    # --- ⭐️ 아이콘(b.png)과 제목을 함께 표시 ---
    col_title, col_text = st.columns([1, 10])
    with col_title:
        st.image("b.png", width=50)
    with col_text:
        st.subheader("경영 상태 분석")
    
    # --- ⭐️ 그래프 스타일 및 폰트 설정 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        st.warning("한글 폰트(Malgun Gothic)를 찾을 수 없어 기본 폰트로 표시됩니다.")
        
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 월별 매출 추이 비교")
        if 'HPSN_MCT_ZCD_NM' not in store_history_df.columns or store_history_df['HPSN_MCT_ZCD_NM'].isnull().all():
            st.warning("'업종' 정보가 없어 동종업계 비교가 불가능합니다.")
        else:
            my_industry = store_history_df['HPSN_MCT_ZCD_NM'].dropna().iloc[0]
            industry_df = master_df[master_df['HPSN_MCT_ZCD_NM'] == my_industry]
            industry_avg_sales = industry_df.groupby('TA_YM')['RC_M1_SAA'].mean().reset_index()
            my_sales = store_history_df[['TA_YM', 'RC_M1_SAA']].dropna().sort_values(by='TA_YM')

            fig1, ax1 = plt.subplots(figsize=(8, 4))
            
            ax1.plot(my_sales['TA_YM'], my_sales['RC_M1_SAA'], marker='o', linestyle='-', color='#1f77b4', label='우리 가맹점')
            ax1.plot(industry_avg_sales['TA_YM'], industry_avg_sales['RC_M1_SAA'], marker='s', linestyle='--', color='#ff7f0e', label='동종업계 평균')
            
            ax1.set_title(f"매출 비교 (업종: {my_industry})", fontsize=15, weight='bold')
            ax1.set_ylabel('월 매출액 (단위: 원)', fontsize=12)
            ax1.tick_params(axis='x', rotation=30)
            ax1.legend(fontsize=11)
            
            st.pyplot(fig1)

    with col2:
        st.markdown("#### 주요 고객 분포 분석")
        latest_data = store_history_df.iloc[0]
        
        ratios = {
            '20대 여성': latest_data.get('M12_FME_1020_RAT', 0), '20대 남성': latest_data.get('M12_MAL_1020_RAT', 0),
            '30-40대 여성': latest_data.get('M12_FME_3040_RAT', 0), '30-40대 남성': latest_data.get('M12_MAL_3040_RAT', 0),
            '50-60대 여성': latest_data.get('M12_FME_5060_RAT', 0), '50-60대 남성': latest_data.get('M12_MAL_5060_RAT', 0),
        }

        demo_df = pd.DataFrame(list(ratios.items()), columns=['Category', 'Ratio'])
        other_ratio = 100 - demo_df['Ratio'].sum()
        if other_ratio > 0:
            demo_df.loc[len(demo_df)] = ['기타', other_ratio]
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        
        custom_palette = []
        for category in demo_df['Category']:
            if '여성' in category:
                custom_palette.append('lightcoral')
            elif '남성' in category:
                custom_palette.append('cornflowerblue')
            else:
                custom_palette.append('lightgrey')

        barplot = sns.barplot(data=demo_df, x='Ratio', y='Category', ax=ax2, palette=custom_palette, orient='h')
        
        ax2.set_title("주요 고객 연령 및 성별 분포", fontsize=15, weight='bold')
        ax2.set_xlabel("매출 비율 (%)", fontsize=12)
        ax2.set_ylabel("")
        
        for p in barplot.patches:
            width = p.get_width()
            ax2.text(width + 1, p.get_y() + p.get_height() / 2, f'{width:.1f}%', ha='left', va='center')
        
        st.pyplot(fig2)

def generate_llm_recommendation(retriever, franchise_id, risk_score):
    # --- ⭐️ 아이콘(c.png)과 제목을 함께 표시 ---
    col_title, col_text = st.columns([1, 10])
    with col_title:
        st.image("c.png", width=50)
    with col_text:
        st.subheader("AI 컨설턴트의 맞춤 정책 추천")
        
    if retriever is None: return

    if risk_score > 0.7:
        diagnosis = f"위험도 점수가 {risk_score*100:.1f}점으로 매우 높습니다. 단기 유동성 위기 및 매출 급락 가능성이 매우 큰 상황으로 판단됩니다."
    elif risk_score > 0.4:
        diagnosis = f"위험도 점수가 {risk_score*100:.1f}점으로 '주의' 단계입니다. 현금 흐름 관리가 필요하며, 경영 효율화를 통한 안정성 확보가 권장됩니다."
    else:
        diagnosis = f"위험도 점수가 {risk_score*100:.1f}점으로 '안전' 단계입니다. 현재 상태를 유지하며, 성장을 위한 비금융 지원(컨설팅, 교육 등)을 고려해볼 시점입니다."

    user_query = f"분석 대상: 서울 성동구 소상공인 (가맹점 ID: {franchise_id})\n진단 요약: {diagnosis}"
    retrieved_docs = retriever.invoke(user_query)
    context = "\\n\\n---\\n\\n".join([doc.page_content for doc in retrieved_docs])
    final_prompt = f"### 배경 정보\n당신은 '서울 성동구 소상공인'을 돕기 위해 만들어진 지역 경제 정책 전문 AI 컨설턴트입니다. 당신의 모든 답변은 성동구 소상공인의 입장에서 시작해야 합니다. 아래 '검색된 정책 정보'만을 근거로 답변해야 하며, 모르는 내용은 절대 지어내면 안 됩니다.\n\n### 검색된 정책 정보\n{context}\n\n### 사용자 요청 및 임무\n{user_query}\n\n위 '사용자 요청'에 명시된 가맹점의 문제를 해결하고 성장을 도울 수 있는 정책 2가지를 추천해주세요. 반드시 성동구 정책을 최우선으로 고려하고, '성동구 소상공인으로서' 왜 이 정책이 유리한지 명확히 설명해야 합니다. 답변은 '추천 이유', '주요 혜택', '자격 요건'을 포함하는 전문적인 보고서 형식으로 작성해주세요."
    
    with st.spinner('LLM이 맞춤형 정책 리포트를 생성하고 있습니다... (실제 호출 중)'):
        try:
            response = ollama.chat(model='gemma:2b', messages=[{'role': 'user', 'content': final_prompt}])
            llm_response_text = response['message']['content']
            st.markdown(llm_response_text)
        except Exception as e:
            st.error(f"LLM 호출 중 오류가 발생했습니다: {e}")
            st.warning("Ollama 데스크탑 앱이 실행 중인지, 또는 'ollama serve' 명령어로 서버가 실행 중인지 확인해주세요.")

# ==============================================================================
# 📜 메인 로직
# ==============================================================================
policy_retriever = build_rag_system()
store_history = st.session_state['store_data']
franchise_id = st.session_state['franchise_id']
biz_type = st.session_state['biz_type']
master_df = load_master_data(biz_type)

st.header(f"'{franchise_id}' 종합 경영 진단 리포트")
st.markdown("---")

# --- 1️⃣. 위험도 점수 먼저 표시 ---
col_title_risk, col_text_risk = st.columns([1, 10])
with col_title_risk:
    st.image("a.png", width=50)
with col_text_risk:
    st.subheader("종합 위험도 점수")

with st.spinner('AI가 실제 데이터를 기반으로 위험도를 분석하고 있습니다...'):
    try:
        model_package = joblib.load('fran_final_ensemble.pkl')
        latest_data = store_history.iloc[[0]]
        risk_score = predict_risk_with_ensemble(model_package, latest_data)
    except Exception as e:
        st.error(f"위험도 분석 중 오류가 발생했습니다: {e}")
        st.stop()

risk_score_percent = risk_score * 100

if risk_score > 0.7:
    risk_level_text = "AI 진단 요약: 모델 분석 결과, 상점의 현재 상태는 '위험' 단계에 해당합니다."
    st.error(risk_level_text)
elif risk_score > 0.4:
    risk_level_text = "AI 진단 요약: 모델 분석 결과, 상점의 현재 상태는 '주의' 단계에 해당합니다."
    st.warning(risk_level_text)
else:
    risk_level_text = "AI 진단 요약: 모델 분석 결과, 상점의 현재 상태는 '안전' 단계에 해당합니다."
    st.success(risk_level_text)

st.metric(label="미래 휴/폐업 리스크 점수", value=f"{risk_score_percent:.1f} 점")
st.progress(int(risk_score_percent))
st.markdown("---")

# --- 2️⃣. 시각화 자료 다음으로 표시 ---
if master_df is not None:
    generate_visualizations(store_history, master_df)
else:
    st.error("동종업계 비교를 위한 마스터 데이터를 로드할 수 없습니다.")
st.markdown("---")

# --- 3️⃣. AI 정책 추천 마지막으로 표시 ---
generate_llm_recommendation(policy_retriever, franchise_id, risk_score)