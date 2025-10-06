# app.py
# -*- coding: utf-8 -*-
# 🏪 점포 마케팅 전략 챗봇 (Streamlit + CatBoost + (선택)Gemini)
# - CSV 4종을 읽어 안전 병합 → 다음달 타깃 생성 → CatBoost 학습(캐시) → 가맹점 리포트
# - Gemini API Key가 있으면 LLM 상담까지 제공

import os, re, json, math
import numpy as np
import pandas as pd
import streamlit as st

# CatBoost
try:
    from catboost import CatBoostRegressor, Pool
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "--quiet"])
    from catboost import CatBoostRegressor, Pool

# Gemini(선택)
_HAS_GEMINI = True
try:
    import google.generativeai as genai
except Exception:
    _HAS_GEMINI = False

# -----------------------------
# 경로/기본 설정 (✅ 네 로컬 경로로 교체)
# -----------------------------
SEED = 42
PATHS = {
    "main": r"./dataset/BasicData_Prep.csv",
    "set1": r"./dataset/big_data_set1_f.csv",
    "set2": r"./dataset/big_data_set2_f.csv",
    "set3": r"./dataset/big_data_set3_f.csv"
}
ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
ID_COL = "ENCODED_MCT"
YM_COL = "TA_YM"
TARGET_MODE = "sales"  # "sales" | "visits" | "sales_growth"

# 사람이 읽기 쉬운 변수명 매핑
PRETTY = {
    "RC_M1_SAA": "월 매출액",
    "RC_M1_TO_UE_CT": "방문 횟수",
    "RC_M1_UE_CUS_CN": "순 방문 고객 수",
    "RC_M1_AV_NP_AT": "객단가",
    "DLV_SAA_RAT": "배달 매출 비중",
    "APV_CE_RAT": "카드 승인율",
    "M1_SME_RY_SAA_RAT": "월 매출 구성비",
    "M1_SME_RY_CNT_RAT": "월 방문 비중",
    "M12_SME_RY_SAA_PCE_RT": "최근 12개월 매출 구성비",
    "M12_SME_RY_CNT_RAT": "최근 12개월 방문 비중",
    "M12_SME_BZN_SAA_PCE_RT": "최근 12개월 상권 매출 구성비",
    "M12_SME_RY_ME_MCT_RAT": "최근 12개월 우리 점 매출 점유",
    "M12_SME_BZN_ME_MCT_RAT": "최근 12개월 상권 매출 점유",
    "M12_MAL_30_RAT": "30대 남성 고객 비율",
    "M12_MAL_50_RAT": "50대 남성 고객 비율",
    "M12_MAL_60_RAT": "60대 이상 남성 고객 비율",
    "MCT_OPE_MS_CN": "운영 개월 수",
    "month": "월(시즌)", "qtr": "분기", "year": "연도"
}
def pretty(col: str) -> str:
    base = re.sub(r"_(s2|s3)$", "", col)
    if base.startswith("TARGET_NEXT_"):
        b = base.replace("TARGET_NEXT_", "")
        return f"다음 달 {PRETTY.get(b, b)}"
    if base.startswith("TARGET_GROWTH_"):
        b = base.replace("TARGET_GROWTH_", "")
        return f"다음 달 {PRETTY.get(b, b)} 증감률"
    return PRETTY.get(base, base)

def action_for(name: str, high: bool) -> str:
    nm = name
    if "방문 횟수" in nm: return "재방문(스탬프·멤버십)으로 매출 전환 강화" if high else "지도/SNS 노출 강화로 신규 유입 확대"
    if "순 방문 고객" in nm: return "단골 전환(친구초대·2회차 쿠폰) 강화" if high else "동네 타깃 리뷰/체험단으로 인지도 확보"
    if "매출액" in nm or "매출 구성비" in nm: return "히트 카테고리 집중, 세트/프리미엄 확장" if high else "시간대/메뉴 재구성 + 밸류/세트로 전환율↑"
    if "객단가" in nm: return "프리미엄/사이드 업셀 유지" if high else "런치밸류/테이크아웃 구성으로 침투율↑"
    if "배달" in nm: return "리뷰/단골 관리 + 광고 ROI 점검" if high else "입점/메뉴 최적화·소액 광고 테스트"
    if any(k in nm for k in ["남성 30대", "여성 20", "여성 30"]): return "핵심 타깃 유지형 프로모션" if high else "타깃 맞춤형 SNS·오퍼 보완"
    if "운영 개월" in nm: return "리프레시·멤버십 재정비" if high else "오픈이벤트/리뷰 모으기 집중"
    if "월(시즌)" in nm: return "성수기: 병목 완화·객단가 극대화" if high else "비수기: 시즌 한정·세트 할인·근거리 광고"
    return "강점 유지 + 약점 보완 A/B 실험(2주 단위)"

# -----------------------------
# 데이터 로드/병합/타깃
# -----------------------------
def load_csv_best(path: str) -> pd.DataFrame:
    last = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            st.write(f"✅ Loaded {os.path.basename(path)} {df.shape} (enc={enc})")
            return df
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to load {path}. Last error: {last}")

def assert_unique(df: pd.DataFrame, keys, name: str):
    if any(k not in df.columns for k in keys):
        raise KeyError(f"[{name}] missing key columns {keys}")
    d = df.duplicated(keys).sum()
    if d != 0:
        top = df.groupby(keys).size().sort_values(ascending=False).head(5)
        raise ValueError(f"[{name}] keys {keys} NOT unique (dup={d}/{len(df)})\n{top}")

def safe_merge(paths: dict) -> pd.DataFrame:
    main = load_csv_best(paths["main"])
    set1 = load_csv_best(paths["set1"])
    set2 = load_csv_best(paths["set2"])
    set3 = load_csv_best(paths["set3"])

    def clean(df):
        bad = [c for c in df.columns if c.lower().startswith("unnamed")] + \
              [c for c in df.columns if re.fullmatch(r"index|level_\\d+", c, re.I)]
        if bad: df = df.drop(columns=bad)
        return df

    main, set1, set2, set3 = map(clean, [main, set1, set2, set3])

    for d in [main, set1, set2, set3]:
        if ID_COL in d.columns: d[ID_COL] = d[ID_COL].astype(str)
        if YM_COL in d.columns: d[YM_COL] = pd.to_numeric(d[YM_COL], errors="coerce").astype("Int64")

    assert_unique(main, [ID_COL, YM_COL], "main")
    assert_unique(set2, [ID_COL, YM_COL], "set2")
    assert_unique(set3, [ID_COL, YM_COL], "set3")
    assert_unique(set1, [ID_COL], "set1")

    m = main.merge(set2, on=[ID_COL, YM_COL], how="left", suffixes=("","_s2"))
    m = m.merge(set3, on=[ID_COL, YM_COL], how="left", suffixes=("","_s3"))
    m = m.merge(set1, on=[ID_COL], how="left", suffixes=("","_s1"))
    st.success(f"✅ SAFE MERGE: {main.shape} -> {m.shape}")
    return m

def make_target(df: pd.DataFrame, mode="sales") -> tuple[pd.DataFrame, str]:
    df = df.copy()
    CAND_SALES = ["RC_M1_SAA", "M1_SME_RY_SAA_RAT", "M12_SME_RY_SAA_PCE_RT"]
    CAND_VISIT = ["RC_M1_TO_UE_CT", "M1_SME_RY_CNT_RAT"]

    def first_exists(cols):
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                return c
        return None

    if mode == "sales":
        base = first_exists(CAND_SALES) or first_exists(CAND_VISIT)
        if base is None: raise RuntimeError("매출 관련 컬럼을 찾지 못했습니다.")
        tgt = f"TARGET_NEXT_{base}"
        df = df.sort_values([ID_COL, YM_COL])
        df[tgt] = df.groupby(ID_COL)[base].shift(-1)

    elif mode == "visits":
        base = first_exists(CAND_VISIT) or first_exists(CAND_SALES)
        if base is None: raise RuntimeError("방문 관련 컬럼을 찾지 못했습니다.")
        tgt = f"TARGET_NEXT_{base}"
        df = df.sort_values([ID_COL, YM_COL])
        df[tgt] = df.groupby(ID_COL)[base].shift(-1)

    elif mode == "sales_growth":
        base = first_exists(CAND_SALES) or first_exists(CAND_VISIT)
        if base is None: raise RuntimeError("매출 관련 컬럼을 찾지 못했습니다.")
        df = df.sort_values([ID_COL, YM_COL])
        lvl = f"TARGET_NEXT_{base}"
        df[lvl] = df.groupby(ID_COL)[base].shift(-1)
        tgt = f"TARGET_GROWTH_{base}"
        eps = 1e-6
        df[tgt] = (df[lvl] - df[base]) / (df[base].abs() + eps)
    else:
        raise ValueError("mode must be one of ['sales','visits','sales_growth']")

    before = len(df)
    df = df[~df[tgt].isna()].copy()
    after = len(df)
    st.info(f"🧱 TARGET 생성(mode={mode}): '{tgt}', 제거된 마지막달 행: {before-after}")
    return df, tgt

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=[np.number]).columns:
        col = df[c]
        if pd.api.types.is_float_dtype(col): df[c] = pd.to_numeric(col, downcast="float")
        else: df[c] = pd.to_numeric(col, downcast="integer")
    return df

# -----------------------------
# 학습(캐시) — ✅ cat_features 문자열 강제 변환 포함 (핵심 수정)
# -----------------------------
@st.cache_resource(show_spinner=True)
def build_and_train_model(paths: dict, target_mode="sales"):
    # 1) 병합
    df = safe_merge(paths)
    df = downcast_numeric(df)
    # 2) 타깃
    df, target_col = make_target(df, mode=target_mode)
    # 3) 파생
    if YM_COL in df.columns:
        df["year"] = (df[YM_COL] // 100).astype("int32")
        df["month"] = (df[YM_COL] % 100).astype("int8")
        df["qtr"] = ((df["month"] - 1) // 3 + 1).astype("int8")

    # 4) 입력 분리
    drop_cols = [c for c in [ID_COL, YM_COL] if c in df.columns]
    for c in df.columns:
        if df[c].isna().mean() > 0.98: drop_cols.append(c)
    drop_cols = sorted(set(drop_cols))
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col].astype(float)

    # ✅ 범주형: object/string/category dtype만
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object","string","category")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 숫자 결측: 중앙값
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float).fillna(X[c].median())

    # 범주 결측: "Unknown", 그리고 ✅ 문자열로 강제 (핵심)
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("Unknown").astype(str)

    # 5) 시계열 분할
    months = sorted(df[YM_COL].unique())
    n = len(months)
    n_test = max(1, int(n*0.2)); n_val = max(1, int(n*0.1))
    test_m = set(months[-n_test:]); val_m = set(months[-(n_test+n_val):-n_test])
    idx_te = df[YM_COL].isin(test_m); idx_va = df[YM_COL].isin(val_m); idx_tr = ~(idx_te | idx_va)

    def sel(a, mask): return a.loc[mask] if isinstance(a, pd.DataFrame) else a[mask]
    X_tr, y_tr = sel(X, idx_tr), y[idx_tr]
    X_va, y_va = sel(X, idx_va), y[idx_va]
    X_te, y_te = sel(X, idx_te), y[idx_te]

    # ✅ cat_features 인덱스 계산
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]

    # ✅ 안전장치: CatBoost에 넣기 전, cat_features를 문자열로 강제 (핵심)
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype(str)
        X_va[c] = X_va[c].astype(str)
        X_te[c] = X_te[c].astype(str)

    # 6) CatBoost 학습
    model = CatBoostRegressor(
        iterations=1200, learning_rate=0.03, depth=8,
        l2_leaf_reg=3.0, loss_function="MAE",
        random_seed=SEED, early_stopping_rounds=100, verbose=False
    )
    trp = Pool(X_tr, label=y_tr.values, cat_features=cat_idx)
    vap = Pool(X_va, label=y_va.values, cat_features=cat_idx)
    model.fit(trp, eval_set=vap, verbose=False)

    # 7) 성능
    te_pool = Pool(X_te, label=y_te.values, cat_features=cat_idx)
    y_pred_te = model.predict(te_pool)
    te_mae = float(np.mean(np.abs(y_pred_te - y_te.values)))
    st.success(f"📏 Test MAE: {te_mae:.4f}  (타깃={pretty(target_col)})")

    # 8) 메타 저장 (추론용)
    num_medians = {c: float(X_tr[c].median()) for c in num_cols}
    valid_num = []
    for c in num_cols:
        miss = df[c].isna().mean(); stdv = float(df[c].std() or 0.0)
        if miss < 0.20 and stdv > 0: valid_num.append(c)

    meta = {
        "target_col": target_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_medians": num_medians
    }

    # 전체 df 저장(최신행 조회용)
    return model, meta, df

# -----------------------------
# 추론/리포트
# -----------------------------
def topk_z(df: pd.DataFrame, meta: dict, store_code: str, k_each=3):
    cols = meta.get("num_cols", [])
    if not cols: return [], []
    z = (df[cols] - df[cols].mean()) / (df[cols].std() + 1e-9)
    row = z[df[ID_COL].astype(str) == str(store_code)]
    if row.empty: return [], []
    s = row.iloc[0].dropna().sort_values(ascending=False)
    highs = [(pretty(c), float(v)) for c, v in s.head(max(1, k_each)).items()]
    lows  = [(pretty(c), float(v)) for c, v in s.tail(max(1, k_each)).items()]
    return highs, lows

def prep_latest_row(df: pd.DataFrame, meta: dict, store_code: str):
    d = df[df[ID_COL].astype(str) == str(store_code)]
    if d.empty: return None, None, None, None
    last = int(d[YM_COL].max()); d = d[d[YM_COL] == last].copy()
    drop_cols = [meta["target_col"], ID_COL, YM_COL]
    X = d.drop(columns=[c for c in drop_cols if c in d.columns], errors="ignore")
    # 학습 시 사용한 피처 순서 맞추기
    use_cols = meta["num_cols"] + meta["cat_cols"]
    X = X.reindex(columns=use_cols, fill_value=np.nan)

    # 숫자/범주 결측 처리 (✅ cat은 문자열로)
    for c in meta["num_cols"]:
        med = meta["num_medians"].get(c, 0.0)
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float).fillna(med)
    for c in meta["cat_cols"]:
        X[c] = X[c].astype("string").fillna("Unknown").astype(str)

    y_true = d[meta["target_col"]].iloc[0] if meta["target_col"] in d.columns else np.nan
    return X, float(y_true) if not pd.isna(y_true) else None, last, use_cols

def build_report(model, meta, df, store_code: str):
    X, y_true, last, use_cols = prep_latest_row(df, meta, store_code)
    if X is None: return None
    cat_idx = [use_cols.index(c) for c in meta["cat_cols"]]
    # ✅ 예측 시에도 cat을 문자열로 보장
    for c in meta["cat_cols"]:
        X[c] = X[c].astype(str)
    pool = Pool(X, cat_features=cat_idx)
    pred = float(model.predict(pool)[0])

    highs, lows = topk_z(df, meta, store_code, k_each=3)
    pred_txt = f"{pred:.2f}" if pred == pred else "산출 불가"
    act_txt  = f"{y_true:.2f}" if y_true is not None else "N/A"
    perf = None
    if (pred == pred and y_true is not None): perf = "양호" if y_true >= pred else "저조"
    summary = (
        f"최근 분석 결과, 해당 가게의 {pretty(meta['target_col'])} 예측값은 {pred_txt}, "
        f"실제값은 {act_txt}입니다." + (f" 전반적으로 {perf}한 수준으로 판단됩니다." if perf else "")
    )
    strengths = "강점 요인: " + ", ".join([f"{n}(z≈{z:.2f})" for n, z in highs]) if highs else "강점 요인은 뚜렷하지 않습니다."
    weaknesses = "개선 필요: " + ", ".join([f"{n}(z≈{z:.2f})" for n, z in lows]) if lows else "주요 개선 항목은 없습니다."
    actions = []
    for n, _ in highs: actions.append(f"[↑ {n}] {action_for(n, True)}")
    for n, _ in lows:  actions.append(f"[↓ {n}] {action_for(n, False)}")
    actions_text = "\n".join([f"- {a}" for a in actions[:6]]) if actions else "- 유입→전환(후크메뉴)→재방문(혜택) 2주 A/B 실험"

    report_md = f"""
**📊 요약**  
{summary}

**💪 강점**  
{strengths}

**⚠️ 개선 필요**  
{weaknesses}

**🧭 추천 전략**  
{actions_text}
""".strip()

    return {
        "merchant": store_code,
        "month": int(last),
        "prediction": pred,
        "actual": y_true,
        "report_md": report_md
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="점포 마케팅 전략 챗봇", page_icon="🏪", layout="wide")
st.title("🏪 점포 마케팅 전략 챗봇")
st.markdown("가맹점 코드(ENCODED_MCT)를 입력하면 **다음 달 성과 예측 + 자동 리포트**를 제공합니다. (옵션: Gemini로 추가 전략 상담)")

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("1) 데이터 준비 및 모델 학습(최초 1회 캐시)")
    target_opt = st.selectbox("타깃 선택", ["sales(다음달 매출)", "visits(다음달 방문)", "sales_growth(다음달 매출 증감률)"], index=0)
    TARGET_MODE = target_opt.split("(")[0]

    if st.button("데이터 로드 & 모델 학습/갱신"):
        with st.spinner("학습 중... (캐시됨)"):
            model, meta, merged_df = build_and_train_model(PATHS, target_mode=TARGET_MODE)
            st.session_state.model = model
            st.session_state.meta = meta
            st.session_state.df = merged_df
        st.success("모델 준비 완료!")

    # 캐시에서 자동 준비
    if "model" not in st.session_state:
        with st.spinner("모델 상태 확인 중..."):
            model, meta, merged_df = build_and_train_model(PATHS, target_mode=TARGET_MODE)
            st.session_state.model = model
            st.session_state.meta = meta
            st.session_state.df = merged_df

    st.subheader("2) 가맹점 분석")
    store_code = st.text_input("가맹점코드(ENCODED_MCT) 입력", placeholder="예: 00BC189C4B").strip()
    if st.button("🔍 분석 실행"):
        if not store_code:
            st.warning("가맹점 코드를 입력해주세요.")
        else:
            with st.spinner("리포트 생성 중..."):
                rep = build_report(st.session_state.model, st.session_state.meta, st.session_state.df, store_code)
                if rep is None:
                    st.error("해당 가맹점을 찾을 수 없습니다.")
                else:
                    st.session_state.last_rep = rep  # Gemini에 컨텍스트로 활용
                    m1, m2, m3 = st.columns(3)
                    m1.metric("기준월", rep["month"])
                    m2.metric("예측값", f"{rep['prediction']:.2f}" if rep['prediction'] is not None else "N/A")
                    m3.metric("실제값", f"{rep['actual']:.2f}" if rep['actual'] is not None else "N/A")
                    st.markdown("### 📘 자동 분석 리포트")
                    st.markdown(rep["report_md"])

with right:
    st.subheader("3) (선택) Gemini 상담")
    if not _HAS_GEMINI:
        st.info("`google-genai`가 설치되지 않아 Gemini 상담을 비활성화했습니다. 필요 시: `pip install google-genai`")
    api_key = st.text_input("Gemini API Key", type="password", help="환경변수 GOOGLE_API_KEY 사용 가능")
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY", "")

    user_q = st.text_area("상담 질문", placeholder="예: 매출을 빠르게 10% 올리려면 무엇부터 해야 하나요?")
    if st.button("🤖 Gemini에게 물어보기", disabled=not _HAS_GEMINI):
        if not user_q:
            st.warning("질문을 입력해주세요.")
        elif not api_key:
            st.error("Gemini API Key를 입력하거나 환경변수 GOOGLE_API_KEY를 설정해주세요.")
        else:
            rep = st.session_state.get("last_rep", {"report_md":"최근 분석 컨텍스트가 없습니다. 상단에서 가맹점 분석을 먼저 실행하세요."})
            try:
                genai.configure(api_key=api_key)
                llm = genai.GenerativeModel("gemini-2.0-flash-exp")
                prompt = f"""너는 소상공인 매출 컨설턴트다.
다음은 특정 점포의 자동 분석 리포트다:

{rep['report_md']}

사용자 질문: {user_q}

[지시사항]
- 위 리포트의 강점/약점을 근거로 3~5개의 **구체적이고 실행가능한** 마케팅 액션을 제안하라.
- 점주가 이해하기 쉬운 한국어로, 근거를 간단히 덧붙여라.
- 광고/리뷰/세트구성/단골전환/운영개선 등 다양한 관점에서 제안하되, 과도하게 복잡하지 않게 쓰라.
"""
                with st.spinner("Gemini가 답변 생성 중..."):
                    res = llm.generate_content(prompt)
                answer = getattr(res, "text", "응답 생성 실패")
                st.success("Gemini 응답")
                st.markdown(answer)
            except Exception as e:
                st.error(f"Gemini 오류: {e}")
