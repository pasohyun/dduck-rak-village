# final_model.py
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib
import json
import os

# 모델과 변수 리스트를 저장할 폴더 생성
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("모델 학습 및 저장을 시작합니다...")

# --- 1. 프랜차이즈 모델 (LGBM 앙상블) ---
print("[1/2] 프랜차이즈 모델 학습...")
try:
    df_fran = pd.read_csv("프랜차이즈.csv", encoding="cp949")
    
    # (여기에 프랜차이즈 모델 학습 로직을 구현합니다)
    # ...
    # ... (df_fran 전처리, X_fran, y_fran 분리) ...
    
    # 예시:
    fran_features = ['RC_M1_UE_CUS_CN', 'M1_SME_RY_SAA_RAT', 'MCT_UE_CLN_NEW_RAT', 'RC_M1_AV_NP_AT_CHG3M', 'M1_SME_RY_CNT_RAT'] # 실제 변수 리스트
    X_fran = df_fran[fran_features]
    y_fran = df_fran['y_next'] # 'y_next'가 타겟이라고 가정
    
    # 모델 학습 (LGBM 예시)
    fran_model = lgb.LGBMClassifier(random_state=42)
    fran_model.fit(X_fran, y_fran)
    
    # 1-1. 모델 저장 (PKL)
    joblib.dump(fran_model, os.path.join(MODEL_DIR, "fran_final_ensemble.pkl"))
    
    # 1-2. 변수 리스트 저장 (JSON)
    with open(os.path.join(MODEL_DIR, "fran_features.json"), "w", encoding="utf-8") as f:
        json.dump(fran_features, f)
        
    print("프랜차이즈 모델 저장 완료.")

except FileNotFoundError:
    print("[오류] '프랜차이즈.csv' 파일을 찾을 수 없습니다. 'data_preprocessing.py'를 먼저 실행하세요.")
except Exception as e:
    print(f"[오류] 프랜차이즈 모델 학습 중 오류 발생: {e}")


# --- 2. 개인영업 모델 (XGBoost) ---
print("\n[2/2] 개인영업 모델 학습...")
try:
    df_indiv = pd.read_csv("개인영업.csv", encoding="cp949")
    
    # (여기에 개인영업 모델 학습 로직을 구현합니다)
    # ...
    # ... (df_indiv 전처리, X_indiv, y_indiv 분리) ...
    
    # 예시:
    indiv_features = ['RC_M1_UE_CUS_CN', 'M1_SME_RY_SAA_RAT', 'MCT_UE_CLN_NEW_RAT', 'RC_M1_AV_NP_AT_CHG3M', 'AGEGEN_Hn'] # 실제 변수 리스트
    X_indiv = df_indiv[indiv_features]
    y_indiv = df_indiv['y_next'] # 'y_next'가 타겟이라고 가정
    
    # 모델 학습 (XGBoost 예시)
    indiv_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    indiv_model.fit(X_indiv, y_indiv)
    
    # 2-1. 모델 저장 (JSON)
    indiv_model.save_model(os.path.join(MODEL_DIR, "개인영업_xgboost_model.json"))
    
    # 2-2. 변수 리스트 저장 (JSON)
    with open(os.path.join(MODEL_DIR, "개인영업_features.json"), "w", encoding="utf-8") as f:
        json.dump(indiv_features, f)
        
    print("개인영업 모델 저장 완료.")

except FileNotFoundError:
    print("[오류] '개인영업.csv' 파일을 찾을 수 없습니다. 'data_preprocessing.py'를 먼저 실행하세요.")
except Exception as e:
    print(f"[오류] 개인영업 모델 학습 중 오류 발생: {e}")

print("\n모든 모델 학습 및 저장이 완료되었습니다.")