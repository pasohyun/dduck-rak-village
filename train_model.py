# train_model.py
# (data_analysis.py의 모델 학습/저장 파트)
# 이 스크립트는 앱 실행 전 한 번만 실행하여 모델 파일을 생성합니다.

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

print("모델 학습을 시작합니다...")

# =========================
# 1) 데이터 로드 및 최소 전처리
# =========================
try:
    df = pd.read_csv("프랜차이즈.csv", encoding="utf-8")
except Exception as e:
    print(f"파일 읽기 오류: {e}")
    print("프랜차이즈.csv 파일이 train_model.py와 동일한 폴더에 있는지 확인하세요.")
    exit()

# 날짜 파싱
df["TA_YM"] = pd.to_datetime(df["TA_YM"], errors="coerce")
df = df.sort_values(["TA_YM", "ENCODED_MCT"]).reset_index(drop=True)

# 결측치 처리 (원본 로직 단순화)
df = df.fillna(0)

# 라벨 생성
df["is_risk_final"] = ((df["is_crash_final"].astype(int) == 1) | (
    df["is_closed"].astype(int) == 1)).astype(int)
df["y_next"] = df.groupby("ENCODED_MCT")["is_risk_final"].shift(-1)
target_col = "y_next"

df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)

# =========================
# 2) 핵심 변수 5개 선정 (SHAP 기반)
# =========================
# 사용자가 직접 입력할 수 있는 현실적인 개수의 핵심 변수만 선정합니다.
# (SHAP 분석에서 중요했던 변수들로 가정)
key_features = [
    'RC_M1_UE_CUS_CN',      # 유니크 고객 수 구간
    'M1_SME_RY_SAA_RAT',    # 동일 업종 매출액 평균 대비 비율
    'MCT_UE_CLN_NEW_RAT',   # 신규 고객 비중
    'RC_M1_AV_NP_AT_CHG3M',  # 객단가 3개월 변화율
    'M1_SME_RY_CNT_RAT'     # 동일 업종 결제건수 평균 대비 비율
]

# 해당 컬럼이 존재하는지 확인
for col in key_features + [target_col]:
    if col not in df.columns:
        print(f"[오류] 필수 컬럼 '{col}'이 CSV 파일에 존재하지 않습니다.")
        exit()

X = df[key_features]
y = df[target_col]

print(f"사용된 변수: {key_features}")

# =========================
# 3) 모델 학습 (단순화된 버전)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

spw = (y_train.value_counts()[0] / max(1, y_train.value_counts()[1]))

model = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    scale_pos_weight=spw,
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          eval_metric="auc",
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
          )

print(
    f"테스트 AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

# =========================
# 4) 변수 중요도 시각화 및 저장
# =========================
try:
    # 한글 폰트 설정 (Windows: Malgun Gothic, macOS: AppleGothic)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"한글 폰트 설정 오류: {e}")
    print("그래프의 한글이 깨질 수 있습니다. 'Malgun Gothic' 폰트 설치를 확인하세요.")


feature_importances = model.feature_importances_
feature_names = X.columns

# 중요도를 데이터프레임으로 만들어 정렬
fi_df = pd.DataFrame(
    {'feature': feature_names, 'importance': feature_importances})
fi_df = fi_df.sort_values('importance', ascending=False)

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=fi_df)
plt.title('모델 변수 중요도')
plt.xlabel('중요도')
plt.ylabel('변수')
plt.tight_layout()

# 그래프 파일로 저장
plt.savefig('feature_importance.png')
print("변수 중요도 그래프가 'feature_importance.png'로 저장되었습니다.")


# =========================
# 5) 모델 및 변수 리스트 저장
# =========================
# 모델 저장
joblib.dump(model, "lgbm_franchise_model.pkl")
print(f"모델이 'lgbm_franchise_model.pkl'로 저장되었습니다.")

# 변수 리스트 저장 (매우 중요)
with open("model_features.json", "w", encoding="utf-8") as f:
    json.dump(key_features, f)
print(f"모델 변수 목록이 'model_features.json'로 저장되었습니다.")