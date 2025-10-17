"""
매장 건강도 점수 계산 모듈
"""
import pandas as pd
import numpy as np


class StoreHealthCalculator:
    """매장 건강도 점수를 계산하는 클래스"""

    def __init__(self):
        self.weights = {
            'sales_amount': 0.30,      # 매출금액 가중치 30%
            'sales_count': 0.20,       # 매출건수 가중치 20%
            'customer_count': 0.20,    # 고객수 가중치 20%
            'revisit_rate': 0.15,      # 재방문율 가중치 15%
            'industry_ratio': 0.15     # 동일업종비율 가중치 15%
        }

    def calculate_comprehensive_health_score(self, store_data):
        """
        종합 매장 건강도 계산 (여러 지표 활용)

        Args:
            store_data: dict 형태의 매장 데이터
                {
                    'RC_M1_SAA': 매출금액 구간,
                    'RC_M1_TO_UE_CT': 매출건수 구간,
                    'RC_M1_UE_CUS_CN': 고객수 구간,
                    'MCT_UE_CLN_REU_RAT': 재방문율,
                    'M1_SME_RY_SAA_RAT': 동일업종 매출비율
                }

        Returns:
            health_score: 0~100점 스케일의 종합 건강도 점수
        """
        scores = []

        # 1. 매출금액 점수 (구간을 점수로 변환, 1~6 구간에서 1이 최고)
        if 'RC_M1_SAA' in store_data and not pd.isna(store_data['RC_M1_SAA']):
            sales_amount_score = ((7 - store_data['RC_M1_SAA']) / 6) * 100
            scores.append(sales_amount_score * self.weights['sales_amount'])

        # 2. 매출건수 점수
        if 'RC_M1_TO_UE_CT' in store_data and not pd.isna(store_data['RC_M1_TO_UE_CT']):
            sales_count_score = ((7 - store_data['RC_M1_TO_UE_CT']) / 6) * 100
            scores.append(sales_count_score * self.weights['sales_count'])

        # 3. 고객수 점수
        if 'RC_M1_UE_CUS_CN' in store_data and not pd.isna(store_data['RC_M1_UE_CUS_CN']):
            customer_score = ((7 - store_data['RC_M1_UE_CUS_CN']) / 6) * 100
            scores.append(customer_score * self.weights['customer_count'])

        # 4. 재방문율 점수 (0~100% 그대로 사용)
        if 'MCT_UE_CLN_REU_RAT' in store_data and not pd.isna(store_data['MCT_UE_CLN_REU_RAT']):
            revisit_score = store_data['MCT_UE_CLN_REU_RAT']
            scores.append(revisit_score * self.weights['revisit_rate'])

        # 5. 동일업종 매출비율 점수 (100%를 기준으로 정규화)
        if 'M1_SME_RY_SAA_RAT' in store_data and not pd.isna(store_data['M1_SME_RY_SAA_RAT']):
            # 100% = 평균, 200% = 평균의 2배 -> 100점으로 변환
            industry_score = min(store_data['M1_SME_RY_SAA_RAT'], 200) / 2
            scores.append(industry_score * self.weights['industry_ratio'])

        if not scores:
            return 0.0

        # 가중 평균 계산
        total_score = sum(scores)
        return round(total_score, 1)

    def get_health_grade(self, health_score):
        """
        건강도 점수를 등급으로 변환

        Args:
            health_score: 건강도 점수 (0~100)

        Returns:
            grade: 등급 문자열
            color: 색상 (Streamlit용)
            emoji: 이모지
        """
        if health_score >= 80:
            return "우수", "green", "🟢"
        elif health_score >= 60:
            return "양호", "lightgreen", "🟡"
        elif health_score >= 40:
            return "보통", "yellow", "🟡"
        elif health_score >= 20:
            return "주의", "orange", "🟠"
        else:
            return "위험", "red", "🔴"

    def analyze_store_status(self, store_data):
        """
        매장 상태 종합 분석

        Args:
            store_data: dict 형태의 매장 데이터

        Returns:
            analysis: dict 형태의 분석 결과
        """
        # 건강도 계산
        health_score = self.calculate_comprehensive_health_score(store_data)
        grade, color, emoji = self.get_health_grade(health_score)

        # 주요 지표 추출
        analysis = {
            'health_score': health_score,
            'grade': grade,
            'color': color,
            'emoji': emoji,
            'key_metrics': {}
        }

        # 주요 지표 추가
        if 'RC_M1_SAA' in store_data and not pd.isna(store_data['RC_M1_SAA']):
            analysis['key_metrics']['매출금액_구간'] = int(store_data['RC_M1_SAA'])

        if 'M1_SME_RY_SAA_RAT' in store_data and not pd.isna(store_data['M1_SME_RY_SAA_RAT']):
            analysis['key_metrics']['업종평균_대비'] = f"{store_data['M1_SME_RY_SAA_RAT']:.1f}%"

        if 'MCT_UE_CLN_REU_RAT' in store_data and not pd.isna(store_data['MCT_UE_CLN_REU_RAT']):
            analysis['key_metrics']['재방문율'] = f"{store_data['MCT_UE_CLN_REU_RAT']:.1f}%"

        return analysis
