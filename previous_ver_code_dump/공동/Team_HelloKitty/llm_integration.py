import google.generativeai as genai
import json
import numpy as np
from typing import List, Dict, Any, Optional
from config import Config

class GeminiLLM:
    def __init__(self, config: Config):
        self.config = config
        if self.config.llm.api_key:
            genai.configure(api_key=self.config.llm.api_key)
            self.model = genai.GenerativeModel(self.config.llm.model_name)
        else:
            self.model = None
            print("Warning: Gemini API key not set in config")

    def format_model_results(self, predictions, probabilities, metrics, feature_importance, feature_names):
        results_data = {
            "model_type": "SAINT",
            "task": self.config.model.task,
            "metrics": metrics,
            "predictions_summary": {
                "count": len(predictions),
                "mean": float(np.mean(predictions)) if self.config.model.task == 'regression' else None,
                "std": float(np.std(predictions)) if self.config.model.task == 'regression' else None,
                "prediction_distribution": self._get_prediction_distribution(predictions)
            },
            "feature_importance": dict(zip(feature_names, feature_importance.tolist())) if feature_importance is not None else None,
            "class_probabilities": self._format_probabilities(probabilities) if probabilities is not None else None
        }
        return results_data

    def _get_prediction_distribution(self, predictions):
        if self.config.model.task == 'regression':
            return {
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
                "quartiles": np.percentile(predictions, [25, 50, 75]).tolist()
            }
        else:
            unique, counts = np.unique(predictions, return_counts=True)
            return dict(zip(unique.astype(str).tolist(), counts.tolist()))

    def _format_probabilities(self, probabilities):
        if probabilities is None:
            return None

        mean_probs = np.mean(probabilities, axis=0)
        return {f"class_{i}": float(prob) for i, prob in enumerate(mean_probs)}

    def generate_insights(self, model_results: Dict[str, Any], custom_user_prompt: Optional[str] = None) -> str:
        if not self.model:
            return "Gemini API not configured. Please set API key in config."

        try:
            # 더 안전한 프롬프트로 수정
            system_prompt = self.config.llm.system_prompt

            # 매우 간단하고 안전한 텍스트 형태로 요약
            metrics = model_results.get("metrics", {})

            # 기본 모델 정보만 포함
            summary_text = "머신러닝 모델 결과 분석입니다."

            # 성능 지표 중 가장 기본적인 것만 포함
            if metrics:
                if 'r2' in metrics:
                    r2_val = metrics['r2']
                    if isinstance(r2_val, (int, float)):
                        summary_text += f" 모델 정확도는 {r2_val:.2f}입니다."
                elif 'mse' in metrics:
                    mse_val = metrics['mse']
                    if isinstance(mse_val, (int, float)):
                        summary_text += f" 오차값은 {mse_val:.3f}입니다."

            if custom_user_prompt:
                user_prompt = custom_user_prompt.format(results=summary_text)
            else:
                user_prompt = f"{summary_text} 이 결과에 대해 간단한 분석을 해주세요."

            # 안전 설정만 사용 (GenerationConfig 제거)
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]

            response = self.model.generate_content(
                user_prompt,
                safety_settings=safety_settings
            )

            # 응답 처리 개선
            if response.candidates:
                candidate = response.candidates[0]

                # finish_reason 확인
                if candidate.finish_reason == 1:  # STOP (정상)
                    return response.text
                elif candidate.finish_reason == 2:  # SAFETY
                    return "Gemini API 안전 필터에 의해 응답이 차단되었습니다. 더 간단한 데이터로 다시 시도해주세요."
                elif candidate.finish_reason == 3:  # RECITATION
                    return "저작권 문제로 응답이 차단되었습니다."
                elif candidate.finish_reason == 4:  # OTHER
                    return "알 수 없는 이유로 응답 생성이 중단되었습니다."
                else:
                    return f"응답 생성 중단 (이유: {candidate.finish_reason})"
            else:
                return "API에서 응답을 생성하지 못했습니다."

        except Exception as e:
            return f"LLM 분석 중 오류 발생: {str(e)}"

    def chat_with_results(self, model_results: Dict[str, Any], user_message: str) -> str:
        if not self.model:
            return "Gemini API not configured. Please set API key in config."

        try:
            # 매우 간단한 컨텍스트 생성
            context_prompt = f"머신러닝 모델에 대한 질문입니다: {user_message}. 간단한 답변을 해주세요."

            # 안전 설정만 사용 (GenerationConfig 제거)
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]

            response = self.model.generate_content(
                context_prompt,
                safety_settings=safety_settings
            )

            # 응답 처리 개선
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.finish_reason == 1:  # STOP (정상)
                    return response.text
                elif candidate.finish_reason == 2:  # SAFETY
                    return "안전 필터에 의해 응답이 제한되었습니다. 다른 질문을 시도해보세요."
                else:
                    return f"응답 생성 중단 (이유: {candidate.finish_reason})"
            else:
                return "응답을 생성할 수 없습니다."

        except Exception as e:
            return f"채팅 중 오류 발생: {str(e)}"

    def update_system_prompt(self, new_prompt: str):
        self.config.llm.system_prompt = new_prompt


    def test_api_connection(self) -> bool:
        if not self.model:
            print("Gemini API not configured")
            return False

        try:
            # 간단한 한국어 테스트 메시지
            test_message = "안녕하세요. API 연결 테스트입니다."

            response = self.model.generate_content(
                test_message,
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            )

            if response.candidates and response.candidates[0].finish_reason == 1:
                print("Gemini API 연결 성공")
                print(f"Test response: {response.text[:50]}...")
                return True
            else:
                print(f"API 연결됐지만 응답 제한됨 (reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'})")
                return False

        except Exception as e:
            print(f"API 연결 테스트 실패: {str(e)}")
            return False

class ModelResultsAnalyzer:
    def __init__(self, config: Config):
        self.llm = GeminiLLM(config)

    def analyze_results(self, predictions, probabilities, metrics, feature_importance,
                       feature_names, custom_prompt=None):
        model_results = self.llm.format_model_results(
            predictions, probabilities, metrics, feature_importance, feature_names
        )

        insights = self.llm.generate_insights(model_results, custom_prompt)

        return {
            'model_results': model_results,
            'insights': insights
        }

    def interactive_chat(self, model_results, user_message):
        return self.llm.chat_with_results(model_results, user_message)

