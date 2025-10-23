# data_preprocessing.py (수정 완료된 버전)

import pandas as pd
import numpy as np
import json
import warnings

# 불필요한 경고 메시지 무시
warnings.filterwarnings("ignore")

print("[Step 1/4] 데이터 로드를 시작합니다...")

# --- 1. 데이터 로드 (data/ 폴더 경로 적용) ---
try:
    df1 = pd.read_csv('big_data_set1_f.csv', encoding='cp949')
    df2 = pd.read_csv('big_data_set2_f.csv', encoding='cp949')
    df3 = pd.read_csv('big_data_set3_f.csv', encoding='cp949')
except FileNotFoundError:
    print("[오류] 'data/' 폴더에서 원본 CSV 파일을 찾을 수 없습니다.")
    print("big_data_set1_f.csv, big_data_set2_f.csv, big_data_set3_f.csv 파일을 data/ 폴더에 넣어주세요.")
    exit()

# --- 2. 데이터 병합 (Master Table 생성) ---
df_monthly = pd.merge(df2, df3, on=['ENCODED_MCT', 'TA_YM'], how='left')
master_df = pd.merge(df1, df_monthly, on='ENCODED_MCT', how='right')

print("[Step 2/4] 기본 전처리를 수행합니다...")

# --- 3. 특별 값(-999999.9)을 NaN으로 변환 ---
SENTINEL = -999999.9
master_df.replace(SENTINEL, np.nan, inplace=True)

# --- 4. 데이터 타입 변환 (날짜) ---
master_df['ARE_D'] = pd.to_datetime(master_df['ARE_D'], format='%Y%m%d', errors='coerce')
master_df['MCT_ME_D'] = pd.to_datetime(master_df['MCT_ME_D'], format='%Y%m%d', errors='coerce')
master_df['TA_YM'] = pd.to_datetime(master_df['TA_YM'], format='%Y%m') # YYYYMM 형식 변환
master_df["ENCODED_MCT"] = master_df["ENCODED_MCT"].astype(str)

# --- 5. 'is_closed' 생성: 단순 폐업 여부 ---
master_df['is_closed'] = master_df['MCT_ME_D'].notna().astype(int)

# --- 6. '강남구' 주소 데이터 필터링 ---
master_df_filtered = master_df[~master_df['MCT_BSE_AR'].str.contains('강남구', na=False)].copy()

# --- 7. 이상치(Outlier) 처리 ---
target_merchant_id = '502658D9C9'
outlier_value = 111.2
merchant_df = master_df_filtered[master_df_filtered['ENCODED_MCT'] == target_merchant_id]
mean_value = merchant_df[merchant_df['DLV_SAA_RAT'] != outlier_value]['DLV_SAA_RAT'].mean()

master_df_filtered.loc[
    (master_df_filtered['ENCODED_MCT'] == target_merchant_id) &
    (master_df_filtered['DLV_SAA_RAT'] == outlier_value),
    'DLV_SAA_RAT'
] = mean_value

df = master_df_filtered

# --- 8. 불필요한 원본 컬럼 삭제 ---
drop_cols = [
    'MCT_SIGUNGU_NM', 'HPSN_MCT_BZN_CD_NM', 'ARE_D',
    'APV_CE_RAT', 'M12_SME_BZN_SAA_PCE_RT', 'M12_SME_BZN_ME_MCT_RAT',
    'M12_SME_RY_ME_MCT_RAT'
]
df = df.drop(columns=drop_cols, errors='ignore')


print("[Step 3/4] 파생 변수(Features)를 생성합니다...")

# ===== 9) 다양성 지표(Entropy) 생성 =====
male_cols = ["M12_MAL_1020_RAT","M12_MAL_30_RAT","M12_MAL_40_RAT","M12_MAL_50_RAT","M12_MAL_60_RAT"]
female_cols = ["M12_FME_1020_RAT","M12_FME_30_RAT","M12_FME_40_RAT","M12_FME_50_RAT","M12_FME_60_RAT"]
age_gender_cols = [c for c in (male_cols + female_cols) if c in df.columns]
cust_type_cols = [c for c in ["MCT_UE_CLN_REU_RAT","MCT_UE_CLN_NEW_RAT"] if c in df.columns]
lifestyle_cols = [c for c in ["RC_M1_SHC_RSD_UE_CLN_RAT","RC_M1_SHC_WP_UE_CLN_RAT","RC_M1_SHC_FLP_UE_CLN_RAT"] if c in df.columns]

# 숫자형 변환
for c in age_gender_cols + cust_type_cols + lifestyle_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

def to_probabilities(frame: pd.DataFrame, cols: list) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=frame.index)
    sub = frame[cols].copy()
    needs_pct_to_prob = (sub.max(skipna=True) > 1.5).any()
    if needs_pct_to_prob:
        sub = sub / 100.0
    sub = sub.clip(lower=0, upper=1)
    row_sum = sub.sum(axis=1)
    divisor = row_sum.replace({0: np.nan})
    sub = sub.div(divisor, axis=0)
    return sub

P_age_gender = to_probabilities(df, age_gender_cols)
P_cust_type  = to_probabilities(df, cust_type_cols)
P_lifestyle  = to_probabilities(df, lifestyle_cols)

def shannon_entropy(P: pd.DataFrame, axis=1) -> pd.Series:
    if P.empty: return pd.Series(np.nan, index=df.index)
    val = -(P * np.log(P)).replace({-0.0: 0, 0.0: 0}).sum(axis=axis)
    return val

def normalized_entropy(H: pd.Series, k: int) -> pd.Series:
    if k <= 1: return pd.Series(np.nan, index=H.index)
    return H / np.log(k)

def gini_simpson(P: pd.DataFrame, axis=1) -> pd.Series:
    if P.empty: return pd.Series(np.nan, index=df.index)
    return 1.0 - (P.pow(2).sum(axis=axis))

def balance_gap(P: pd.DataFrame, axis=1) -> pd.Series:
    if P.empty: return pd.Series(np.nan, index=df.index)
    return (P.max(axis=axis) - P.min(axis=axis))

feat = pd.DataFrame({"ENCODED_MCT": df["ENCODED_MCT"], "TA_YM": df["TA_YM"]})
H_age = shannon_entropy(P_age_gender); Hn_age = normalized_entropy(H_age, k=P_age_gender.shape[1] if not P_age_gender.empty else 0)
G_age = gini_simpson(P_age_gender); BG_age = balance_gap(P_age_gender)
feat["AGEGEN_H"] = H_age; feat["AGEGEN_Hn"] = Hn_age; feat["AGEGEN_GS"] = G_age; feat["AGEGEN_BG"] = BG_age

H_ct = shannon_entropy(P_cust_type); Hn_ct = normalized_entropy(H_ct, k=P_cust_type.shape[1] if not P_cust_type.empty else 0)
G_ct = gini_simpson(P_cust_type); BG_ct = balance_gap(P_cust_type)
feat["TYPE_H"] = H_ct; feat["TYPE_Hn"] = Hn_ct; feat["TYPE_GS"] = G_ct; feat["TYPE_BG"] = BG_ct

H_ls = shannon_entropy(P_lifestyle); Hn_ls = normalized_entropy(H_ls, k=P_lifestyle.shape[1] if not P_lifestyle.empty else 0)
G_ls = gini_simpson(P_lifestyle); BG_ls = balance_gap(P_lifestyle)
feat["LIFE_H"] = H_ls; feat["LIFE_Hn"] = Hn_ls; feat["LIFE_GS"] = G_ls; feat["LIFE_BG"] = BG_ls

merged = df.merge(feat, on=["ENCODED_MCT","TA_YM"], how="left")
df = merged

# ===== 10) 배달 의존도 그룹 생성 =====
df['DLV_SAA_RAT'].fillna(0, inplace=True)
mct_avg_delivery_ratio = df.groupby('ENCODED_MCT')['DLV_SAA_RAT'].mean()
delivery_off_stores = mct_avg_delivery_ratio[mct_avg_delivery_ratio == 0].index
delivery_on_stores_ratio = mct_avg_delivery_ratio.drop(index=delivery_off_stores)

q1 = delivery_on_stores_ratio.quantile(0.25)
q3 = delivery_on_stores_ratio.quantile(0.75)

def classify_dependency(avg_ratio):
    if avg_ratio == 0: return '배달 미진행'
    elif avg_ratio >= q3: return '의존도(상)'
    elif avg_ratio < q1: return '의존도(하)'
    else: return '의존도(중)'

classification = mct_avg_delivery_ratio.map(classify_dependency)
mapping = {'배달 미진행': 0, '의존도(하)': 1, '의존도(중)': 2, '의존도(상)': 3}
classification_numeric = classification.map(mapping).astype('Int64')
classification_numeric.name = 'Delivery_Group_Code'

df_out = df.merge(classification_numeric, on='ENCODED_MCT', how='left')

# ===== 11) 주소(ADDR_DONG) 매핑 =====
try:
    with open('address_mapping.json', 'r', encoding='utf-8') as f:
        seongdong_mapping_auto = json.load(f)
except FileNotFoundError:
    print("[오류] 'data/address_mapping.json' 파일을 찾을 수 없습니다.")
    exit()

df = df_out

if "MCT_BSE_AR" in df.columns:
    def split_street_building(s):
        parts = str(s).strip().split()
        rest = " ".join(parts[2:])
        idx = rest.rfind(" ")
        if idx == -1:
            street_name = rest
            building_number = np.nan
        else:
            street_name = rest[:idx].strip()
            building_number = rest[idx:].strip()
        return pd.Series([street_name, building_number])
    
    df[["ADDR_STREET", "ADDR_BUILDING"]] = df["MCT_BSE_AR"].apply(split_street_building)
    df = df.drop(columns=["MCT_BSE_AR"])

    def map_seongdong(addr):
        for dong, streets in seongdong_mapping_auto.items():
            if addr in streets:
                return dong
        return None

    df['ADDR_DONG'] = df['ADDR_STREET'].apply(map_seongdong)
    
    # ADDR_DONG 결측치/None/빈 문자열 제거 (여기서 최종 정리)
    df = df[df['ADDR_DONG'].notna()]
    df = df[df['ADDR_DONG'].astype(str).str.strip().ne('None')]
    df = df[df['ADDR_DONG'].astype(str).str.strip() != '']
else:
    print("[경고] 'MCT_BSE_AR' 컬럼이 없어 ADDR_DONG을 생성할 수 없습니다.")


# ===== 12) 구간 변수 숫자형 변환 =====
default_interval_cols = [
    "MCT_OPE_MS_CN", "RC_M1_SAA", "RC_M1_TO_UE_CT",
    "RC_M1_UE_CUS_CN", "RC_M1_AV_NP_AT"
]
interval_columns = [c for c in default_interval_cols if c in df.columns]

for col in interval_columns:
    split_values = df[col].astype(str).str.split('_').str[0]
    df[col] = pd.to_numeric(split_values, errors='coerce')

# ===== 13) 성별&연령 축소 =====
df['M12_MAL_3040_RAT'] = df['M12_MAL_30_RAT'] + df['M12_MAL_40_RAT']
df['M12_MAL_5060_RAT'] = df['M12_MAL_50_RAT'] + df['M12_MAL_60_RAT']
df['M12_FME_3040_RAT'] = df['M12_FME_30_RAT'] + df['M12_FME_40_RAT']
df['M12_FME_5060_RAT'] = df['M12_FME_50_RAT'] + df['M12_FME_60_RAT']

# (참고) 원본 컬럼 삭제는 모델링 스크립트(train_model.py)에서 exclude_cols로 처리하는 것이 더 안전합니다.
# 여기서 미리 삭제하지 않아도 됩니다.

# ===== 14) 시계열 변화량 생성 =====
score_vars = [
    'RC_M1_UE_CUS_CN', 'RC_M1_AV_NP_AT', 'RC_M1_TO_UE_CT', 'RC_M1_SAA', 'MCT_OPE_MS_CN',
]
ratio_vars_data2 = [
    'DLV_SAA_RAT', 'M1_SME_RY_SAA_RAT', 'M1_SME_RY_CNT_RAT', 'M12_SME_RY_SAA_PCE_RT',
]
customer_vars = [
    'M12_MAL_1020_RAT', 'M12_MAL_3040_RAT', 'M12_MAL_5060_RAT',
    'M12_FME_1020_RAT', 'M12_FME_3040_RAT', 'M12_FME_5060_RAT',
    'MCT_UE_CLN_REU_RAT', 'MCT_UE_CLN_NEW_RAT',
    'RC_M1_SHC_RSD_UE_CLN_RAT', 'RC_M1_SHC_WP_UE_CLN_RAT', 'RC_M1_SHC_FLP_UE_CLN_RAT'
]
all_vars = score_vars + ratio_vars_data2 + customer_vars

def create_time_series_features(df, var_name):
    # (중복 코드 제거됨)
    grouped = df.groupby('ENCODED_MCT')[var_name]
    
    # 3개월 이동평균
    df[f'{var_name}_MA3'] = grouped.transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # 3개월 변화율 (%)
    df[f'{var_name}_CHG3M'] = grouped.transform(
        lambda x: ((x - x.shift(3)) / x.shift(3) * 100)
    )
    return df

print("시계열 변수를 생성 중입니다... (시간이 다소 걸릴 수 있습니다)")
for var in all_vars:
    if var in df.columns:
        df = create_time_series_features(df, var)

print("[Step 4/4] 최종 파일을 저장합니다...")

# ===== 15) 프랜차이즈 / 개인영업 분리 저장 =====
# MCT_BRD_NUM (프랜차이즈 번호)의 유무로 분리
df_not_null = df[df['MCT_BRD_NUM'].notnull()]
df_null = df[df['MCT_BRD_NUM'].isnull()]
df_null = df_null.drop(columns=['MCT_BRD_NUM'], errors='ignore')

# 최종 결과물은 메인 폴더에 저장
output_path_not_null = "프랜차이즈.csv"
output_path_null = "개인영업.csv"

df_not_null.to_csv(output_path_not_null, index=False)
df_null.to_csv(output_path_null, index=False)

print("="*50)
print(f"전처리 완료: '{output_path_not_null}' ({len(df_not_null)} 행)")
print(f"전처리 완료: '{output_path_null}' ({len(df_null)} 행)")
print("="*50)