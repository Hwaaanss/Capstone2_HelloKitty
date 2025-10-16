import streamlit as st
import pandas as pd
import os
import json
from llm_integration import GeminiLLM
from config import Config
from stacking_model import StackingEnsembleWrapper

st.set_page_config(
    page_title="점포 마케팅 전략 챗봇",
    page_icon="🏪",
    layout="wide"
)

def load_column_info():
    """컬럼 정보 JSON 파일을 로드"""
    try:
        with open("column_info.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"컬럼 정보 파일 로드 실패: {str(e)}")
        return {}

def load_feature_importance():
    """Feature importance JSON 파일을 로드"""
    try:
        with open("feature_importance.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Feature importance 파일을 찾을 수 없습니다: {str(e)}")
        return None

def initialize_session_state():
    if 'config' not in st.session_state:
        st.session_state.config = Config.load()
    if 'llm' not in st.session_state:
        st.session_state.llm = GeminiLLM(st.session_state.config)
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'store_data' not in st.session_state:
        st.session_state.store_data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'column_info' not in st.session_state:
        st.session_state.column_info = load_column_info()
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = load_feature_importance()

def load_test_store_data(store_code):
    """Test 데이터에서 점포코드(ENCODED_MCT)를 기반으로 데이터를 로드 (실제 미래 예측용)"""
    try:
        from utils import load_csv_with_encoding, get_latest_preprocessed_data

        # 가장 최신 전처리 데이터 파일 자동 검색
        original_path = get_latest_preprocessed_data("./dataset")

        if not original_path or not os.path.exists(original_path):
            st.error("전처리된 데이터 파일을 찾을 수 없습니다.")
            return None

        df = load_csv_with_encoding(original_path)
        if df is None:
            st.error("데이터를 로드할 수 없습니다.")
            return None

        # 누락된 컬럼 추가 (전처리 과정에서 생성되는 컬럼들)
        if 'quaters' not in df.columns and 'TA_YM' in df.columns:
            # quaters 컬럼 생성 (분기 계산)
            df['quaters'] = ((df['TA_YM'] % 100) - 1) // 3 + 1

        # navi, vis 컬럼이 없으면 평균값으로 채우기
        if 'navi' not in df.columns:
            df['navi'] = 0  # 기본값
        if 'vis' not in df.columns:
            df['vis'] = 0  # 기본값

        # 실전 예측: 최신 월 데이터로 다음 달 예측 (진짜 미래)
        if 'TA_YM' in df.columns:
            # 점포별로 시간순 정렬
            df_sorted = df.sort_values(['ENCODED_MCT', 'TA_YM']).reset_index(drop=True)

            # 해당 점포 데이터 필터링
            store_data = df_sorted[df_sorted['ENCODED_MCT'].astype(str) == str(store_code)]

            if not store_data.empty:
                # 최신 월 데이터 사용
                latest_data = store_data.iloc[-1].to_dict()
                current_ta_ym = latest_data['TA_YM']

                # 다음 달 계산 (예: 202412 → 202501)
                year = current_ta_ym // 100
                month = current_ta_ym % 100

                if month == 12:
                    next_ta_ym = (year + 1) * 100 + 1
                else:
                    next_ta_ym = year * 100 + month + 1

                latest_data['next_month_TA_YM'] = next_ta_ym
                # 실제 미래 예측이므로 실제값은 없음
                latest_data['next_month_actual_target'] = None

                return latest_data

        return None

    except Exception as e:
        st.error(f"테스트 데이터 로드 중 오류: {str(e)}")
        return None

def get_available_test_codes():
    """사용 가능한 테스트 점포 코드들 반환 (실전 예측용: 모든 점포)"""
    try:
        from utils import load_csv_with_encoding, get_latest_preprocessed_data

        # 가장 최신 전처리 데이터 파일 자동 검색
        original_path = get_latest_preprocessed_data("./dataset")
        if not original_path or not os.path.exists(original_path):
            return []

        df = load_csv_with_encoding(original_path)
        if df is None:
            return []

        # 실전 예측: 최신 데이터가 있는 모든 점포 (survive=1인 점포)
        if 'TA_YM' in df.columns:
            df_sorted = df.sort_values(['ENCODED_MCT', 'TA_YM']).reset_index(drop=True)

            # 각 점포의 최신 데이터만 추출
            latest_data = df_sorted.groupby('ENCODED_MCT').tail(1)

            # 최신 데이터가 있는 점포 코드 반환
            return latest_data['ENCODED_MCT'].unique().tolist()

        return []

    except Exception as e:
        st.error(f"테스트 코드 로드 중 오류: {str(e)}")
        return []

def load_trained_model():
    """학습된 Stacking Ensemble 모델 로드"""
    try:
        config = Config.load()
        model = StackingEnsembleWrapper(config)
        if model.model_exists():
            success = model.load_model()
            if success:
                return model
            else:
                st.error("모델 로드에 실패했습니다.")
                return None
        else:
            st.error("학습된 모델이 없습니다. 먼저 학습을 진행해주세요.")
            return None
    except Exception as e:
        st.error(f"모델 로드 중 오류: {str(e)}")
        return None

def predict_store_target(model, store_data):
    """점포 데이터로 타겟 예측 (시계열: N-1월 데이터로 N월 예측)"""
    if model is None or store_data is None:
        return None

    try:
        # 점포 데이터를 DataFrame으로 변환
        store_df = pd.DataFrame([store_data])

        # 타겟 컬럼 및 불필요한 컬럼 제거
        target_columns = st.session_state.config.data.target_columns
        # 시계열 예측에서 추가된 컬럼도 제거
        columns_to_drop = target_columns + ['next_month_TA_YM', 'next_month_actual_target']
        features_df = store_df.drop(columns=[col for col in columns_to_drop if col in store_df.columns], errors='ignore')

        # 컬럼명을 학습 시와 동일하게 변환 (LightGBM 호환성)
        features_df.columns = [
            col.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
               .replace('"', '').replace(',', '_').replace(':', '_').replace(' ', '')
            for col in features_df.columns
        ]

        # 학습시 사용된 feature names와 일치하는지 확인
        if hasattr(model, 'feature_names_clean') and model.feature_names_clean is not None:
            # 누락된 feature 확인 및 추가 (0으로 채움)
            missing_features = [f for f in model.feature_names_clean if f not in features_df.columns]
            if missing_features:
                st.warning(f"누락된 feature: {missing_features}")
                for feat in missing_features:
                    features_df[feat] = 0

            # 학습시 사용된 feature 순서대로 재정렬
            features_df = features_df[model.feature_names_clean]

        # 예측 수행
        predictions, probabilities = model.predict(features_df)
        return predictions[0] if len(predictions) > 0 else None

    except Exception as e:
        st.error(f"예측 중 오류: {str(e)}")
        return None

def format_store_analysis(store_data, current_target, predicted_target, column_info, feature_importance=None):
    """점포 정보와 분석 결과를 마케팅 전략용으로 포맷팅 (실전 미래 예측)"""
    if store_data is None:
        return "점포 정보를 찾을 수 없습니다."

    # 주요 정보 추출
    info_text = "점포 분석 정보 (실전 미래 예측):\n"

    # 현재 월과 다음 월 정보
    current_month = store_data.get('TA_YM', 'N/A')
    next_month = store_data.get('next_month_TA_YM', 'N/A')
    info_text += f"- 분석 기준 월 (최신): {current_month}\n"
    info_text += f"- 예측 대상 월 (미래): {next_month}\n\n"

    # 기본 정보 (중요한 것만 표시)
    important_keys = ['ENCODED_MCT', 'MCT_CAT_CD', 'SIDO_CD', 'SGG_CD']
    for key in important_keys:
        if key in store_data and pd.notna(store_data[key]) and store_data[key] != '':
            description = column_info.get(key, key)
            info_text += f"- {description}: {store_data[key]}\n"

    # 예측값 표시 (실전 예측이므로 실제값 없음)
    if predicted_target is not None:
        info_text += f"\n📊 성과 예측 ({next_month}월):\n"
        info_text += f"- AI 예측 타겟값: {predicted_target:.2f}\n"
        info_text += f"- 이 값은 {current_month}월 데이터 기반 {next_month}월 예측입니다.\n"

    # Feature Importance 정보 추가 (상위 10개)
    if feature_importance:
        info_text += f"\n🎯 타겟 예측에 가장 중요한 요소들 (Top 10):\n"
        top_features = list(feature_importance.items())[:10]
        for i, (feature, importance) in enumerate(top_features, 1):
            description = column_info.get(feature, feature)
            current_value = store_data.get(feature, 'N/A')
            if isinstance(current_value, (int, float)):
                info_text += f"{i}. {description} (중요도: {importance:.4f}) - 현재값: {current_value:.2f}\n"
            else:
                info_text += f"{i}. {description} (중요도: {importance:.4f}) - 현재값: {current_value}\n"

    # 주요 특징값들 표시 (숫자 데이터만)
    info_text += f"\n📋 주요 데이터 (의미 포함):\n"
    for key, value in store_data.items():
        if key not in important_keys and pd.notna(value) and isinstance(value, (int, float)):
            description = column_info.get(key, key)
            info_text += f"- {description}: {value}\n"

    return info_text

def main():
    initialize_session_state()

    st.title("🏪 점포 마케팅 전략 챗봇 (미래 예측)")
    st.markdown("**점포코드**를 입력하면 AI가 **다음 달 성과를 예측**하고 마케팅 전략을 제안해드립니다.")
    st.info("💡 실전 예측 모드: 최신 월 데이터를 기반으로 다음 달 타겟값을 예측합니다. (예: 202412월 → 202501월 예측)")

    # 사용 가능한 테스트 코드 표시
    available_codes = get_available_test_codes()
    if available_codes:
        with st.expander("📋 사용 가능한 점포 코드들", expanded=False):
            st.write(f"총 {len(available_codes)}개의 점포가 있습니다:")
            cols = st.columns(4)
            for i, code in enumerate(available_codes):
                col_idx = i % 4
                cols[col_idx].code(code)

    st.markdown("---")

    # 모델 로드 (첫 번째 실행시에만)
    if st.session_state.model is None:
        with st.spinner("학습된 모델을 로드하는 중..."):
            st.session_state.model = load_trained_model()

    # 점포코드 입력
    col1, col2 = st.columns([2, 1])

    with col1:
        store_code = st.text_input("점포코드(ENCODED_MCT)를 입력하세요:", placeholder="예: 000F03E44A")

    with col2:
        if st.button("🔍 점포 분석 실행", type="primary"):
            if store_code:
                with st.spinner("점포 정보를 분석하고 예측하는 중..."):
                    store_data = load_test_store_data(store_code)
                    st.session_state.store_data = store_data

                    if store_data and st.session_state.model:
                        # 실전 예측: 다음 달은 미래이므로 실제값 없음
                        next_month_actual = store_data.get('next_month_actual_target', None)
                        next_month_ta_ym = store_data.get('next_month_TA_YM', 'N/A')
                        current_month_ta_ym = store_data.get('TA_YM', 'N/A')

                        # 예측 수행 (최신 월 데이터로 다음 달 예측)
                        prediction = predict_store_target(st.session_state.model, store_data)
                        st.session_state.predictions = prediction

                        if prediction is not None:
                            st.success(f"점포코드 {store_code}의 분석이 완료되었습니다!")
                            st.info(f"📅 {current_month_ta_ym}월 데이터 → {next_month_ta_ym}월 예측\n\n"
                                   f"**예측값: {prediction:.2f}**\n\n"
                                   f"💡 {next_month_ta_ym}월은 미래 시점으로 실제값이 없습니다.")
                        else:
                            st.error("예측에 실패했습니다. 위의 오류 메시지를 확인해주세요.")

                    elif store_data:
                        st.warning("점포 정보는 찾았으나 모델이 로드되지 않았습니다.")
                    else:
                        st.error(f"점포코드 {store_code}를 데이터에서 찾을 수 없습니다.")
            else:
                st.warning("점포코드를 입력해주세요.")

    # 점포 정보 표시
    if st.session_state.store_data:
        with st.expander("로드된 점포 정보", expanded=True):
            store_df = pd.DataFrame([st.session_state.store_data]).T
            store_df.columns = ['값']
            st.dataframe(store_df)

    st.markdown("---")

    # 챗봇 영역
    st.subheader("마케팅 전략 상담")

    if st.session_state.store_data:
        # 채팅 입력
        chat_input = st.text_input("마케팅 전략에 대해 질문해보세요:",
                                   placeholder="예: 이 점포의 매출을 늘릴 수 있는 마케팅 전략은 무엇인가요?")

        col_send, col_clear = st.columns([1, 1])

        with col_send:
            if st.button("질문하기") and chat_input:
                if st.session_state.config.llm.api_key:
                    with st.spinner("답변을 생성하는 중..."):
                        # 점포 정보와 분석 결과를 조합하여 프롬프트 생성 (시계열)
                        # 다음 달 실제 타겟값 사용 (비교용)
                        next_month_actual = st.session_state.store_data.get('next_month_actual_target', None)

                        store_analysis = format_store_analysis(
                            st.session_state.store_data,
                            next_month_actual,
                            st.session_state.predictions,
                            st.session_state.column_info,
                            st.session_state.feature_importance
                        )

                        # 시스템 프롬프트와 컨텍스트를 결합한 프롬프트 생성
                        system_prompt = st.session_state.config.llm.system_prompt

                        full_prompt = f"""{system_prompt}

Here is the current situation and AI analysis results for the test store:

{store_analysis}

User question: {chat_input}

**IMPORTANT INSTRUCTION:**
The "Most Important Factors for Target Prediction (Top 10)" listed above are the variables that have the GREATEST IMPACT on the target value according to our predictive model. These are data-driven insights about what actually drives performance.

When providing marketing strategy recommendations:
1. Focus on the top 3-5 factors with highest importance scores
2. Analyze the current values of these key factors
3. Provide specific, actionable marketing tactics to improve or leverage these factors
4. Think like a marketing strategist, not a data analyst - translate the data insights into business actions

For example:
- If a high-importance factor has a low current value → Suggest concrete marketing campaigns or operational changes to improve it
- If a high-importance factor already has a high value → Recommend strategies to maintain and amplify this strength
- Prioritize improvements based on both importance score and feasibility

Your response should be practical marketing advice that a restaurant owner can implement, not a technical explanation of the model or data. Focus on customer acquisition, retention, promotional strategies, operational improvements, and competitive positioning.

Write your entire response in Korean.
                        """

                        try:
                            response = st.session_state.llm.model.generate_content(
                                full_prompt,
                                safety_settings=[
                                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                                ]
                            )

                            if response.candidates and response.candidates[0].finish_reason == 1:
                                assistant_response = response.text
                            else:
                                assistant_response = "죄송합니다. 응답을 생성할 수 없습니다. 다른 질문을 시도해보세요."

                        except Exception as e:
                            assistant_response = f"오류가 발생했습니다: {str(e)}"

                        # 채팅 히스토리에 추가
                        st.session_state.chat_history.append({
                            'user': chat_input,
                            'assistant': assistant_response
                        })
                else:
                    st.error("Gemini API 키가 설정되지 않았습니다. config.json에서 api_key를 설정해주세요.")

        with col_clear:
            if st.button("대화 초기화"):
                st.session_state.chat_history = []
                st.success("대화 내역이 초기화되었습니다.")
                st.rerun()

        # 채팅 히스토리 표시
        if st.session_state.chat_history:
            st.markdown("### 대화 내역")

            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    st.markdown(f"**🙋‍♂️ 질문 {len(st.session_state.chat_history) - i}:**")
                    st.info(chat['user'])

                    st.markdown("**🤖 답변:**")
                    st.success(chat['assistant'])
                    st.markdown("---")

    else:
        st.info("💡 점포코드를 입력하고 '점포 정보 로드' 버튼을 클릭하면 해당 점포에 맞는 마케팅 전략 상담이 가능합니다.")

if __name__ == "__main__":
    main()