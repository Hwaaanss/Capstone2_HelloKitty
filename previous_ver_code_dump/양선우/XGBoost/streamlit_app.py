import streamlit as st
import pandas as pd
import os
import json
from llm_integration import GeminiLLM
from config import Config
from xgboost_model import XGBoostWrapper

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

def load_test_store_data(store_code):
    """Test 데이터에서 점포코드(ENCODED_MCT)를 기반으로 데이터를 로드"""
    try:
        from utils import load_csv_with_encoding

        # 원본 데이터에서 test 데이터를 동적으로 추출
        original_path = "./dataset/BasicData_Prep.csv"

        if not os.path.exists(original_path):
            st.error("❌ 데이터 파일을 찾을 수 없습니다.")
            return None

        df = load_csv_with_encoding(original_path)
        if df is None:
            st.error("❌ 데이터를 로드할 수 없습니다.")
            return None

        # Test 데이터 추출 (각 점포의 최신 데이터)
        if 'TA_YM' in df.columns:
            max_ta_ym_per_store = df.groupby('ENCODED_MCT')['TA_YM'].max().reset_index()
            max_ta_ym_per_store.columns = ['ENCODED_MCT', 'max_TA_YM']

            # 원본 데이터와 merge하여 최신 데이터만 추출
            df_with_max = df.merge(max_ta_ym_per_store, on='ENCODED_MCT')
            test_data = df_with_max[df_with_max['TA_YM'] == df_with_max['max_TA_YM']]

            # 해당 점포의 테스트 데이터 찾기
            store_info = test_data[test_data['ENCODED_MCT'].astype(str) == str(store_code)]
            if not store_info.empty:
                return store_info.iloc[0].to_dict()

        return None

    except Exception as e:
        st.error(f"테스트 데이터 로드 중 오류: {str(e)}")
        return None

def get_available_test_codes():
    """사용 가능한 테스트 점포 코드들 반환"""
    try:
        from utils import load_csv_with_encoding

        original_path = "./dataset/BasicData_Prep.csv"
        if not os.path.exists(original_path):
            return []

        df = load_csv_with_encoding(original_path)
        if df is None:
            return []

        # Test 데이터 추출 (각 점포의 최신 데이터)
        if 'TA_YM' in df.columns:
            max_ta_ym_per_store = df.groupby('ENCODED_MCT')['TA_YM'].max().reset_index()
            max_ta_ym_per_store.columns = ['ENCODED_MCT', 'max_TA_YM']

            df_with_max = df.merge(max_ta_ym_per_store, on='ENCODED_MCT')
            test_data = df_with_max[df_with_max['TA_YM'] == df_with_max['max_TA_YM']]

            return test_data['ENCODED_MCT'].unique().tolist()

        return []

    except Exception as e:
        st.error(f"테스트 코드 로드 중 오류: {str(e)}")
        return []

def load_trained_model():
    """학습된 XGBoost 모델 로드"""
    try:
        config = Config.load()
        model = XGBoostWrapper(config)
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
    """점포 데이터로 타겟 예측"""
    if model is None or store_data is None:
        return None

    try:
        # 점포 데이터를 DataFrame으로 변환
        store_df = pd.DataFrame([store_data])

        # 타겟 컬럼 및 불필요한 컬럼 제거
        target_columns = st.session_state.config.data.target_columns
        columns_to_drop = target_columns + ['max_TA_YM']  # max_TA_YM은 데이터 분리시 추가된 컬럼
        features_df = store_df.drop(columns=[col for col in columns_to_drop if col in store_df.columns], errors='ignore')

        # 학습시 사용된 feature names와 일치하는지 확인
        if model.feature_names is not None:
            # 학습시 사용된 feature만 선택
            missing_features = [f for f in model.feature_names if f not in features_df.columns]
            if missing_features:
                st.warning(f"누락된 feature: {missing_features}")

            # 학습시 사용된 feature 순서대로 재정렬
            features_df = features_df[[f for f in model.feature_names if f in features_df.columns]]

        # 예측 수행
        predictions, probabilities = model.predict(features_df)
        return predictions[0] if len(predictions) > 0 else None

    except Exception as e:
        st.error(f"예측 중 오류: {str(e)}")
        return None

def format_store_analysis(store_data, current_target, predicted_target, column_info):
    """점포 정보와 분석 결과를 마케팅 전략용으로 포맷팅"""
    if store_data is None:
        return "점포 정보를 찾을 수 없습니다."

    # 주요 정보 추출
    info_text = "점포 분석 정보:\n"

    # 기본 정보 (중요한 것만 표시)
    important_keys = ['ENCODED_MCT', 'TA_YM', 'MCT_CAT_CD', 'SIDO_CD', 'SGG_CD']
    for key in important_keys:
        if key in store_data and pd.notna(store_data[key]) and store_data[key] != '':
            description = column_info.get(key, key)
            info_text += f"- {description}: {store_data[key]}\n"

    # 현재 타겟값과 예측값 비교
    if current_target is not None and predicted_target is not None:
        info_text += f"\n📊 성과 분석:\n"
        info_text += f"- 현재 타겟값: {current_target:.2f}\n"
        info_text += f"- AI 예측값: {predicted_target:.2f}\n"

        diff = predicted_target - current_target
        if diff > 0:
            info_text += f"- 예상 개선 효과: +{diff:.2f} (긍정적)\n"
        else:
            info_text += f"- 예상 변화: {diff:.2f} (주의 필요)\n"

    # 주요 특징값들 표시 (숫자 데이터만)
    info_text += f"\n📋 주요 데이터 (의미 포함):\n"
    for key, value in store_data.items():
        if key not in important_keys and pd.notna(value) and isinstance(value, (int, float)):
            description = column_info.get(key, key)
            info_text += f"- {description}: {value}\n"

    return info_text

def main():
    initialize_session_state()

    st.title("🏪 점포 마케팅 전략 챗봇")
    st.markdown("**테스트 점포코드**를 입력하면 AI가 분석한 결과를 바탕으로 마케팅 전략을 제안해드립니다.")
    st.info("💡 테스트용 점포코드만 사용 가능합니다. 학습 완료 후 출력된 코드를 사용하세요!")

    # 사용 가능한 테스트 코드 표시
    available_codes = get_available_test_codes()
    if available_codes:
        with st.expander("📋 사용 가능한 테스트 점포 코드들", expanded=False):
            st.write(f"총 {len(available_codes)}개의 테스트 점포가 있습니다:")
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
        store_code = st.text_input("📍 테스트 점포코드(ENCODED_MCT)를 입력하세요:", placeholder="예: 00BC189C4B")

    with col2:
        if st.button("🔍 점포 분석 실행", type="primary"):
            if store_code:
                with st.spinner("점포 정보를 분석하고 예측하는 중..."):
                    store_data = load_test_store_data(store_code)
                    st.session_state.store_data = store_data

                    if store_data and st.session_state.model:
                        # 현재 타겟값 추출
                        target_columns = st.session_state.config.data.target_columns
                        current_target = None
                        if target_columns and target_columns[0] in store_data:
                            current_target = store_data[target_columns[0]]

                        # 예측 수행
                        prediction = predict_store_target(st.session_state.model, store_data)
                        st.session_state.predictions = prediction

                        if current_target is not None:
                            st.success(f"✅ 점포코드 {store_code}의 분석이 완료되었습니다!")
                            st.info(f"📊 현재값: {current_target:.2f} → 예측값: {prediction:.2f}")
                        else:
                            st.success(f"✅ 점포코드 {store_code}의 분석이 완료되었습니다!")

                    elif store_data:
                        st.warning("⚠️ 점포 정보는 찾았으나 모델이 로드되지 않았습니다.")
                    else:
                        st.error(f"❌ 점포코드 {store_code}를 테스트 데이터에서 찾을 수 없습니다.")
            else:
                st.warning("점포코드를 입력해주세요.")

    # 점포 정보 표시
    if st.session_state.store_data:
        with st.expander("📊 로드된 점포 정보", expanded=True):
            store_df = pd.DataFrame([st.session_state.store_data]).T
            store_df.columns = ['값']
            st.dataframe(store_df)

    st.markdown("---")

    # 챗봇 영역
    st.subheader("💬 마케팅 전략 상담")

    if st.session_state.store_data:
        # 채팅 입력
        chat_input = st.text_input("마케팅 전략에 대해 질문해보세요:",
                                   placeholder="예: 이 점포의 매출을 늘릴 수 있는 마케팅 전략은 무엇인가요?")

        col_send, col_clear = st.columns([1, 1])

        with col_send:
            if st.button("💭 질문하기") and chat_input:
                if st.session_state.config.llm.api_key:
                    with st.spinner("답변을 생성하는 중..."):
                        # 점포 정보와 분석 결과를 조합하여 프롬프트 생성
                        target_columns = st.session_state.config.data.target_columns
                        current_target = None
                        if target_columns and target_columns[0] in st.session_state.store_data:
                            current_target = st.session_state.store_data[target_columns[0]]

                        store_analysis = format_store_analysis(
                            st.session_state.store_data,
                            current_target,
                            st.session_state.predictions,
                            st.session_state.column_info
                        )

                        # 시스템 프롬프트와 컨텍스트를 결합한 프롬프트 생성
                        system_prompt = st.session_state.config.llm.system_prompt

                        full_prompt = f"""{system_prompt}

다음은 테스트 점포의 현재 상황과 AI 분석 결과입니다:

{store_analysis}

사용자 질문: {chat_input}

현재 타겟값과 AI 예측값의 차이를 분석하여, 어떤 요소들을 개선하면 타겟값을 향상시킬 수 있을지 구체적이고 실용적인 마케팅 전략을 한국어로 제안해주세요. 점포의 특징과 현재 성과를 고려한 맞춤형 전략을 제시해주세요.
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
            if st.button("🗑️ 대화 초기화"):
                st.session_state.chat_history = []
                st.success("대화 내역이 초기화되었습니다.")
                st.rerun()

        # 채팅 히스토리 표시
        if st.session_state.chat_history:
            st.markdown("### 📝 대화 내역")

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