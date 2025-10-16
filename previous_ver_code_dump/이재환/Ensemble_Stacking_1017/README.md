# Stacking Ensemble Model with LLM Integration

이 프로젝트는 스태킹 앙상블 모델(XGBoost, LightGBM, CatBoost, RandomForest)과 Gemini 2.5 Flash API를 통합하여 머신러닝 모델의 결과를 분석하고 인사이트를 제공하는 시스템입니다.

## 주요 기능

- **스태킹 앙상블 모델**: 다중 Base Models와 Meta Model을 활용한 고성능 예측 모델
  - Base Models: XGBoost, LightGBM, CatBoost, RandomForest
  - Meta Model: XGBoost 또는 Ridge/LogisticRegression
- **Gemini API 통합**: 모델 결과를 LLM이 분석하여 인사이트 제공
- **Streamlit 인터페이스**: 웹 기반 사용자 인터페이스
- **동적 설정**: config 파일을 통한 하이퍼파라미터 및 설정 관리
- **시각화**: 모델 성능 및 학습 과정 시각화
- **모듈화**: 유지보수가 용이한 모듈 구조

## 프로젝트 코드 흐름

### 1. 데이터 로드 및 전처리

**utils.py**
```
load_csv_with_encoding()
  → 인코딩 자동 감지
  → CSV 파일 로드
  → DataFrame 반환

print_dataset_info()
  → 데이터셋 기본 정보 출력
  → 결측값 확인
```

**stacking_model.py - _preprocess_data()**
```
데이터 전처리
  → 결측값 처리
     - 범주형: 'Unknown' 채움
     - 수치형: median 값 채움
  → 자동 타입 변환
     - object → numeric 시도
     - 범주형 변수 LabelEncoder 적용
  → 높은 카디널리티 처리
     - 상위 50개 카테고리 + 'Other'
  → 전처리된 DataFrame 반환
```

### 2. 데이터 분할

**stacking_model.py - prepare_data()**
```
ENCODED_MCT 컬럼 존재 확인
  │
  ├─ 있음 → _prepare_data_by_store()
  │          → 점포별 분할
  │          → TA_YM 기준 최신 데이터 테스트셋 분리
  │          → 나머지 데이터 점포 단위 train/val 분할
  │
  └─ 없음 → _prepare_data_standard()
             → 일반적인 랜덤 분할
             → train_test_split 사용

_finalize_data_preparation()
  → 전처리 적용
  → feature_names 저장
  → X_train, X_val, X_test, y_train, y_val, y_test 반환
```

### 3. 모델 빌드 및 학습

**stacking_model.py - build_model()**
```
config에서 하이퍼파라미터 로드
  → task 확인
     - regression → Base Models (Regressors), Meta Model (Regressor)
     - classification → Base Models (Classifiers), Meta Model (Classifier)

  → Base Models 생성
     - XGBoost: xgb_* 파라미터
     - LightGBM: lgbm_* 파라미터
     - CatBoost: catboost_* 파라미터
     - RandomForest: rf_* 파라미터

  → Meta Model 생성
     - meta_model_type='xgboost' → XGBoost
     - 그 외 → Ridge/LogisticRegression
```

**stacking_model.py - train_model()**
```
1단계: Base Models 학습
  → 각 모델을 독립적으로 학습
  → XGBoost, LightGBM: eval_set 사용
  → CatBoost, RandomForest: 일반 학습
  → train/val 예측값 생성 (메타 피처)

2단계: Meta Model 학습
  → Base Models의 예측값을 입력으로 사용
  → Meta Model 학습 및 최종 예측

학습 히스토리 저장
  → Meta Model의 train_loss, val_loss 저장
```

### 4. 모델 평가

**stacking_model.py - evaluate_model()**
```
예측 수행
  → predict() 호출
     - Base Models로 예측
     - Meta Model로 최종 예측
  → 전처리 자동 적용

평가 지표 계산
  │
  ├─ regression
  │   → MSE, RMSE, R2
  │
  └─ classification
      → Accuracy, Precision, Recall, F1

metrics_history 저장
metrics, predictions, probabilities 반환
```

**stacking_model.py - get_feature_importance()**
```
XGBoost Base Model의 feature_importances_ 추출
  → 각 feature의 중요도 점수
  → numpy array 반환
```

### 5. 시각화 및 저장

**stacking_model.py - plot_training_history()**
```
matplotlib 사용
  → 학습 곡선 그래프
     - Meta Model Training Loss
     - Meta Model Validation Loss
  → 평가 지표 막대그래프
     - MSE, RMSE, R2 또는
     - Accuracy, Precision, Recall, F1

config.ui 설정에 따라
  → save_plots_image: PNG 저장
  → show_plots_popup: 팝업 표시
```

**stacking_model.py - save_model()**
```
모델 저장
  → stacking_ensemble_model.pkl
     - Base Models (XGBoost, LightGBM, CatBoost, RandomForest)
     - Meta Model
  → stacking_metadata.pkl
     - train_history
     - metrics_history
     - label_encoders
     - feature_names
```

**main.py - run_training_pipeline()**
```
Feature importance JSON 저장
  → feature_names와 importance 매핑
  → 중요도 순 정렬
  → feature_importance.json 저장
```

### 6. LLM 통합

**llm_integration.py - GeminiLLM**
```
초기화
  → config.llm.api_key 확인
  → genai.configure()
  → GenerativeModel 생성

format_model_results()
  → metrics 포맷팅
  → predictions 요약
  → feature_importance 변환
  → JSON 구조로 반환

generate_insights()
  → 모델 결과를 자연어로 요약
  → system_prompt와 결합
  → Gemini API 호출
  → 분석 인사이트 반환
```

### 7. Streamlit 웹 인터페이스

**streamlit_app.py**
```
initialize_session_state()
  → config 로드
  → llm 초기화
  → session_state 설정

load_trained_model()
  → XGBoostWrapper 생성
  → 저장된 모델 로드
  → model 객체 반환

load_test_store_data()
  → CSV에서 원본 데이터 로드
  → ENCODED_MCT + TA_YM 기준 테스트셋 추출
  → 해당 점포 정보 반환

predict_store_target()
  → 점포 데이터를 DataFrame으로 변환
  → 타겟 컬럼 제거
  → feature_names 순서 맞춤
  → model.predict() 호출
  → 예측값 반환

format_store_analysis()
  → 점포 기본 정보 추출
  → 현재값 vs 예측값 비교
  → feature_importance 상위 10개
  → 주요 특징값 표시
  → 포맷된 텍스트 반환

채팅 인터페이스
  → 점포코드 입력
  → 분석 실행 버튼
  → 모델 예측 수행
  → LLM 프롬프트 구성
     - system_prompt
     - 점포 분석 정보
     - feature_importance
     - 사용자 질문
  → Gemini API 호출
  → 마케팅 전략 답변 생성
  → 채팅 히스토리 표시
```

### 8. 전체 실행 흐름

**CLI 모드**
```
python3 main.py --mode train --data ./dataset/BasicData_Prep2.csv --target RC_M1_SAA

main.py
  → argparse 파싱
  → run_training_pipeline() 호출
     → load_csv_with_encoding()
     → StackingEnsembleWrapper 생성
     → prepare_data()
     → build_model() (Base Models + Meta Model)
     → train_model() (Base Models → Meta Model)
     → evaluate_model()
     → plot_training_history()
     → save_model()
     → feature_importance.json 저장
```

**Streamlit 모드**
```
python main.py --mode streamlit

main.py
  → os.system("streamlit run streamlit_app.py")

streamlit_app.py
  → 세션 초기화
  → 모델 로드
  → 점포코드 입력
  → 데이터 분석 및 예측
  → LLM 챗봇 상담
  → 마케팅 전략 제안
```


## 프로젝트 자료 구조

```
.
├── config.py              # 설정 관리 모듈
├── config.json            # 기본 설정 파일
├── stacking_model.py      # 스태킹 앙상블 모델 래퍼 클래스
├── llm_integration.py     # Gemini API 통합 모듈
├── utils.py               # 유틸리티 함수
├── streamlit_app.py       # Streamlit 웹 인터페이스
├── main.py                # 메인 실행 스크립트
├── requirements.txt       # 필수 패키지 목록
├── dataset/               # 데이터셋 폴더
├── plots/                 # 생성된 플롯 저장 폴더
└── models/                # 학습된 모델 저장 폴더
```

## 설치 및 실행

### 1. 필수 라이브러리 설치

Python 3.8 이상 권장됨.

(mac)
```zsh
pip3 install -r requirements.txt
```

(window)
```bash
pip install -r requirements.txt
```

설치 주요 라이브러리:
- `xgboost`: XGBoost 모델
- `lightgbm`: LightGBM 모델
- `catboost`: CatBoost 모델
- `scikit-learn`: RandomForest 및 데이터 전처리/평가
- `streamlit`: 챗봇 웹 인터페이스
- `pandas`, `numpy`: 데이터 처리
- `matplotlib`, `seaborn`, `plotly`: 시각화
- `google-generativeai`: Gemini API 연동

### 2. Gemini API 키 설정

`config.json` 파일에서 개인 gemini API 키를 설정.

```json
{
  "llm": {
    "api_key": "your-gemini-api-key-here"
  }
}
```

### 3. 실행 방법

#### 웹 인터페이스 실행 (권장)

(mac)
```bash
python3 main.py --mode streamlit
```

(window)
```bash
python main.py --mode streamlit
```


브라우저에서 자동으로 열립니다 (기본 주소: http://localhost:8501)

#### CLI 모드로 학습 (기준 컬럼: 매출 구간)

(mac)
```zsh
python3 main.py --mode train --data ./dataset/BasicData_Prep2.csv --target RC_M1_SAA
```

(window)
```bash
python main.py --mode train --data ./dataset/BasicData_Prep2.csv --target RC_M1_SAA
```

다중 타겟 컬럼 지정:

```bash
python main.py --mode train --data ./dataset/your_data.csv --target col1 col2 col3
```



## 설정 옵션

### 모델 설정 (config.json의 model 섹션)

```json
{
  "model": {
    "lr": 0.001,
    "epochs": 5000,
    "task": "regression",
    "verbose": 1,
    "xgb_max_depth": 6,
    "xgb_min_child_weight": 1.0,
    "xgb_subsample": 0.8,
    "xgb_colsample_bytree": 0.8,
    "xgb_gamma": 0.0,
    "xgb_reg_alpha": 0.0,
    "xgb_reg_lambda": 1.0,
    "xgb_tree_method": "hist",
    "xgb_device": "cpu",
    "xgb_eval_metric": "rmse",
    "lgbm_num_leaves": 31,
    "lgbm_max_depth": -1,
    "lgbm_min_child_samples": 20,
    "lgbm_subsample": 0.8,
    "lgbm_colsample_bytree": 0.8,
    "lgbm_reg_alpha": 0.0,
    "lgbm_reg_lambda": 1.0,
    "catboost_depth": 6,
    "catboost_l2_leaf_reg": 3.0,
    "catboost_border_count": 128,
    "rf_max_depth": 10,
    "rf_min_samples_split": 5,
    "rf_min_samples_leaf": 2,
    "rf_max_features": "sqrt",
    "meta_model_type": "xgboost"
  }
}
```

주요 파라미터:
- `lr`: 학습률 (기본값: 0.001)
- `epochs`: 학습 반복 횟수 (기본값: 5000)
- `task`: 작업 유형 ("regression", "binary", "multiclass")
- `meta_model_type`: 메타 모델 타입 ("xgboost", "ridge", "logistic")

XGBoost 파라미터:
- `xgb_max_depth`: 트리의 최대 깊이
- `xgb_subsample`: 서브샘플 비율
- `xgb_reg_alpha`: L1 정규화
- `xgb_reg_lambda`: L2 정규화

LightGBM 파라미터:
- `lgbm_num_leaves`: 리프 노드 수
- `lgbm_max_depth`: 최대 깊이 (-1은 제한 없음)

CatBoost 파라미터:
- `catboost_depth`: 트리 깊이
- `catboost_l2_leaf_reg`: L2 정규화

RandomForest 파라미터:
- `rf_max_depth`: 최대 깊이
- `rf_max_features`: 분할에 사용할 특성 수

### LLM 설정

```json
{
  "llm": {
    "api_key": "",
    "model_name": "gemini-2.5-flash",
    "temperature": 0.7,
    "max_tokens": 1000,
    "system_prompt": "(e.g., You are an AI assistant analyzing machine learning model results."
  }
}
```

### UI 설정

```json
{
  "ui": {
    "show_plots_popup": true,
    "save_plots_image": true,
    "plots_save_dir": "./plots"
  }
}
```

### 데이터 설정

```json
{
  "data": {
    "dataset_dir": "./dataset",
    "target_columns": ["target"]
  }
}
```

## 모델 평가 지표

### Regression (회귀)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R2 Score

### Classification (분류)
- Accuracy
- Precision
- Recall
- F1 Score

## 특수 기능

### ENCODED_MCT 기반 데이터 분할

데이터셋에 `ENCODED_MCT`(점포 고유 코드) 컬럼이 있는 경우, 점포별로 데이터를 분할합니다:
- 각 점포의 최신 데이터를 테스트 셋으로 사용
- 나머지 데이터를 점포 기준으로 train/validation 분할

### 자동 데이터 전처리

- 결측값 자동 처리 (categorical: 'Unknown', numerical: median)
- 범주형 변수 자동 인코딩
- 높은 카디널리티 처리 (상위 50개 카테고리 + 'Other')

## 개발 정보

- **머신러닝 모델**: 스태킹 앙상블 (XGBoost, LightGBM, CatBoost, RandomForest)
- **LLM**: Gemini 2.5 Flash
- **웹 프레임워크**: Streamlit
- **시각화**: Matplotlib, Seaborn, Plotly
- **데이터 처리**: Pandas, NumPy, Scikit-learn

## 참고사항

- config 파일을 통해 모든 설정 관리
- LLM API 키 미설정 시에도 모델 학습 및 기본 분석 가능
- 학습된 모델은 `./models/` 디렉토리에 자동 저장
- Feature importance는 XGBoost Base Model 기준으로 `feature_importance.json`에 저장
- 스태킹 앙상블을 통해 단일 모델 대비 향상된 예측 성능 제공

