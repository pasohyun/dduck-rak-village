# utils.py (수정된 코드)
import streamlit as st
import joblib
import xgboost as xgb
import pandas as pd
import json
import shap
import os

MODEL_DIR = "models"

@st.cache_resource
def load_models_and_features():
    """
    앱 로딩 시 두 개의 모델과 변수 리스트를 모두 메모리에 올립니다.
    """
    models = {}
    features = {}
    
    # 1. 프랜차이즈 모델 로드 (PKL)
    try:
        model_path = os.path.join(MODEL_DIR, "fran_final_ensemble.pkl")
        models['프랜차이즈'] = joblib.load(model_path)
        
        feature_path = os.path.join(MODEL_DIR, "fran_features.json")
        with open(feature_path, "r", encoding="utf-8") as f:
            features['프랜차이즈'] = json.load(f)
    except FileNotFoundError:
        st.error(f"프랜차이즈 모델 파일({model_path}) 또는 변수 파일({feature_path})을 찾을 수 없습니다.")
        return None, None

    # 2. 개인영업 모델 로드 (XGBoost JSON)
    try:
        model_path = os.path.join(MODEL_DIR, "개인영업_xgboost_model.json")
        models['개인영업'] = xgb.XGBClassifier()
        models['개인영업'].load_model(model_path)
        
        feature_path = os.path.join(MODEL_DIR, "개인영업_features.json")
        with open(feature_path, "r", encoding="utf-8") as f:
            features['개인영업'] = json.load(f)
    except FileNotFoundError:
        st.error(f"개인영업 모델 파일({model_path}) 또는 변수 파일({feature_path})을 찾을 수 없습니다.")
        return None, None
    except Exception as e:
        st.error(f"개인영업 모델 로드 중 오류: {e}")
        return None, None
        
    return models, features

def get_prediction(model, features_list, latest_data_row):
    """
    최신 데이터 행(Series)을 받아 모델 예측 확률을 반환합니다.
    """
    try:
        # 1. Series를 DataFrame으로 변환
        input_df = pd.DataFrame([latest_data_row])
        
        # 2. 모델이 학습한 순서대로 컬럼을 재정렬
        input_df_reordered = input_df.reindex(columns=features_list, fill_value=0)
        
        # 3. 예측 (0: 안전, 1: 위험)
        probability = model.predict_proba(input_df_reordered)[0, 1]
        return probability
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
        return None

def get_risk_level(probability):
    """
    확률값(0~1)을 3단계 위험도로 변환합니다.
    """
    if probability > 0.7:
        return "위험", "error"
    elif probability > 0.4:
        return "주의", "warning"
    else:
        return "안전", "success"

@st.cache_data
def get_top_3_features(_model, _features_list, _latest_data_row):
    """
    SHAP을 사용해 해당 예측에 가장 큰 영향을 미친 TOP 3 변수를 찾습니다.
    (Streamlit 캐시를 위해 함수 인자에 _를 붙임)
    """
    try:
        # 1. 예측과 동일한 형태로 입력 데이터 준비
        input_df = pd.DataFrame([_latest_data_row])
        input_df_reordered = input_df.reindex(columns=_features_list, fill_value=0)
        
        # 2. SHAP Explainer 생성
        explainer = shap.TreeExplainer(_model)
        
        # 3. SHAP 값 계산 (단일 행에 대해)
        shap_values = explainer.shap_values(input_df_reordered)
        
        # 4. shap_values[1] (위험 클래스에 대한 기여도) 사용
        #    shap.TreeExplainer는 종종 [class_0_sv, class_1_sv] 리스트를 반환
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1][0] # [0]은 첫 번째 행(단일 행)
        else:
            shap_values_class1 = shap_values[0] # [0]은 첫 번째 행
            
        # 5. SHAP 값과 변수명 매핑
        feature_impacts = pd.Series(shap_values_class1, index=_features_list)
        
        # 6. 절대값 기준 TOP 3 선정
        top_3_abs = feature_impacts.abs().nlargest(3).index
        top_3_series = feature_impacts[top_3_abs]
        
        return top_3_series
        
    except Exception as e:
        print(f"SHAP 분석 오류: {e}")
        return None