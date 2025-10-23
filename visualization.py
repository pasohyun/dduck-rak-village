# visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _prep_12m(df: pd.DataFrame, month_col="TA_YM", value_col="SALES"):
    d = df[[month_col, value_col]].copy()
    d[month_col] = pd.to_datetime(d[month_col])
    d = d.sort_values(month_col)
    if d.empty:
        raise ValueError("12개월 준비: 입력 데이터가 비어있습니다.")
    end = d[month_col].max().to_period("M")
    months = pd.period_range(end=end, periods=12, freq="M").to_timestamp()
    d = d.set_index(month_col).reindex(months)
    d.index.name = month_col
    d = d.reset_index()
    d["label"] = d[month_col].dt.strftime("%Y-%m")
    return d

def plot_sales_trend(df_user: pd.DataFrame, df_industry: pd.DataFrame,
                     title="📈 월별 매출 추세 비교 (내 매장 vs 업종 평균)",
                     month_col="TA_YM", user_col="SALES", industry_col="AVG_SALES"):
    u = _prep_12m(df_user, month_col, user_col)
    i = _prep_12m(df_industry, month_col, industry_col)

    merged = pd.merge(u[[month_col, "label", user_col]],
                      i[[month_col, industry_col]],
                      on=month_col, how="outer").sort_values(month_col)

    # 최근 달 업종 대비 편차(%)
    last = merged.dropna(subset=[user_col, industry_col]).tail(1)
    diff_pct = None
    if not last.empty and float(last[industry_col].iloc[0]) != 0:
        diff_pct = (float(last[user_col].iloc[0]) - float(last[industry_col].iloc[0])) / float(last[industry_col].iloc[0]) * 100

    # 최근 3개월 기울기(만원/월)
    def slope(s):
        y = s.values.astype(float)
        x = np.arange(len(y))
        if len(y) < 2 or np.any(np.isnan(y)): return np.nan
        xm, ym = x.mean(), y.mean()
        denom = np.sum((x - xm)**2)
        if denom == 0: return np.nan
        return np.sum((x - xm)*(y - ym))/denom

    s_user = slope(merged[user_col].tail(3))
    s_ind  = slope(merged[industry_col].tail(3))

    fig = plt.figure(figsize=(8,4))
    ax = fig.gca()
    ax.plot(merged["label"], merged[user_col], marker="o", label="내 매장")
    ax.plot(merged["label"], merged[industry_col], linestyle="--", marker="s", label="업종 평균")
    ax.set_title(title)
    ax.set_xlabel("월")
    ax.set_ylabel("월 매출 (만원)")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend()

    info = []
    if diff_pct is not None and np.isfinite(diff_pct): info.append(f"최근달 편차: {diff_pct:+.1f}%")
    if np.isfinite(s_user): info.append(f"3개월 추세(내): {s_user:+.1f}")
    if np.isfinite(s_ind):  info.append(f"3개월 추세(업): {s_ind:+.1f}")
    if info:
        ax.text(0.99, 0.02, "\n".join(info), transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig

def analyze_risk(store_df: pd.DataFrame, industry_avg: float | None, sales_col="SALES"):
    risks = []
    # 1) 업종 평균 대비 낮은 매출
    if industry_avg is not None and not store_df.empty:
        last_sales = float(store_df.sort_values("TA_YM").tail(1)[sales_col].iloc[0])
        if last_sales < industry_avg * 0.8:
            risks.append(f"동일 업종 평균 대비 낮은 매출 (최근달 {last_sales:,.0f}만원, 업종평균 {industry_avg:,.0f}만원)")

    # 2) 최근 3개월 하락 추세
    s = store_df.sort_values("TA_YM")[sales_col].tail(3).values
    if len(s) >= 2 and np.all(np.isfinite(s)) and (s[-1] < s[0]):
        risks.append("최근 3개월 매출 하락 추세")

    # 3) 매출 변동성 과다(표준편차/평균 > 0.25)
    vals = store_df[sales_col].dropna().values
    if len(vals) >= 3 and np.mean(vals) > 0:
        vol = np.std(vals)/np.mean(vals)
        if vol > 0.25:
            risks.append("매출 변동성이 큽니다(안정성 낮음)")

    return risks[:3] if risks else ["유의미한 위험 신호가 뚜렷하지 않습니다."]
