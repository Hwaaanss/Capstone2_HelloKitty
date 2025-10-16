#!/usr/bin/env python3

import gc
import os
import calendar
import pandas as pd
import numpy as np
from pathlib import Path


def run_preprocessing(raw_data_dir="./dataset/raw", output_dir="./dataset"):
    """
    prepro.ipynb의 전처리 로직을 적용하여 데이터를 전처리합니다.

    Args:
        raw_data_dir: 원본 데이터가 있는 디렉토리
        output_dir: 전처리된 데이터를 저장할 디렉토리

    Returns:
        전처리된 DataFrame과 저장된 파일 경로
    """
    print("=" * 60)
    print("Starting Preprocessing Pipeline")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[Step 1] Loading raw data files...")
    data1_path = os.path.join(raw_data_dir, "big_data_set1_f.csv")
    data2_path = os.path.join(raw_data_dir, "big_data_set2_f.csv")
    data3_path = os.path.join(raw_data_dir, "big_data_set3_f.csv")

    missing_files = []
    for path in [data1_path, data2_path, data3_path]:
        if not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        error_msg = f"Required files not found:\n"
        for f in missing_files:
            error_msg += f"  - {f}\n"
        error_msg += f"\nPlease place the following files in '{raw_data_dir}':\n"
        error_msg += "  - big_data_set1_f.csv\n"
        error_msg += "  - big_data_set2_f.csv\n"
        error_msg += "  - big_data_set3_f.csv\n"
        raise FileNotFoundError(error_msg)

    data1 = pd.read_csv(data1_path, encoding='cp949')
    print(f"  - Loaded {data1_path}: {data1.shape}")

    data2 = pd.read_csv(data2_path, encoding='cp949')
    print(f"  - Loaded {data2_path}: {data2.shape}")

    data3 = pd.read_csv(data3_path, encoding='cp949')
    print(f"  - Loaded {data3_path}: {data3.shape}")

    # 2. 데이터 병합
    print("\n[Step 2] Merging datasets...")
    data = pd.merge(data2, data3, on=['ENCODED_MCT', 'TA_YM'], how='inner')
    print(f"  - After merging data2 and data3: {data.shape}")

    data = pd.merge(data, data1, on='ENCODED_MCT', how='left')
    print(f"  - After merging with data1: {data.shape}")

    del data1, data2, data3
    gc.collect()

    # 3. 불필요한 컬럼 제거
    print("\n[Step 3] Dropping unnecessary columns...")
    columns_to_drop = ['MCT_BRD_NUM', 'MCT_SIGUNGU_NM', 'MCT_NM', 'MCT_BSE_AR']
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    if 'MCT_BSE_AR' in existing_columns_to_drop:
        # MCT_BSE_AR이 중복으로 있을 경우를 대비
        existing_columns_to_drop = list(set(existing_columns_to_drop))

    data_prep = data.drop(columns=existing_columns_to_drop)
    print(f"  - Dropped columns: {existing_columns_to_drop}")
    print(f"  - Shape after dropping: {data_prep.shape}")

    # 4. 구간 데이터 숫자로 변환
    print("\n[Step 4] Converting interval columns to numeric...")
    columns_to_split = ['MCT_OPE_MS_CN', 'RC_M1_SAA', 'RC_M1_TO_UE_CT',
                        'RC_M1_UE_CUS_CN', 'RC_M1_AV_NP_AT', 'APV_CE_RAT']

    for c in columns_to_split:
        if c in data_prep.columns:
            mask = data_prep[c].notna()
            data_prep.loc[mask, c] = data_prep.loc[mask, c].astype(str).str.split('_').str[0].astype(int)
            print(f"  - Converted {c}")

    # 5. HPSN_MCT_BZN_CD_NM 결측값 처리
    print("\n[Step 5] Filling missing values in HPSN_MCT_BZN_CD_NM...")
    if 'HPSN_MCT_BZN_CD_NM' in data_prep.columns:
        before_na = data_prep['HPSN_MCT_BZN_CD_NM'].isna().sum()
        data_prep['HPSN_MCT_BZN_CD_NM'] = data_prep['HPSN_MCT_BZN_CD_NM'].fillna('비상권')
        print(f"  - Filled {before_na} missing values with '비상권'")

    # 6. 날짜 컬럼 처리
    print("\n[Step 6] Processing date columns...")
    data_prep['ARE_D'] = pd.to_numeric(data_prep['ARE_D'], errors='coerce')
    data_prep['MCT_ME_D'] = pd.to_numeric(data_prep['MCT_ME_D'], errors='coerce')

    data_prep['ARE_D'] = data_prep['ARE_D'].astype('Int64')
    data_prep['MCT_ME_D'] = data_prep['MCT_ME_D'].astype('Int64')

    # 7. survive 컬럼 생성
    print("\n[Step 7] Creating survive column...")
    na_rows = data_prep['MCT_ME_D'].isna()
    data_prep['survive'] = na_rows.astype(int)

    # survive 값 재계산
    print("  - Calculating survive status...")
    for idx in data_prep.index:
        tmp = data_prep.loc[idx]
        try:
            if pd.notna(tmp['MCT_ME_D']) and tmp['TA_YM'] == tmp['MCT_ME_D'] // 100:
                data_prep.loc[idx, 'survive'] = 0
            else:
                data_prep.loc[idx, 'survive'] = 1
        except:
            data_prep.loc[idx, 'survive'] = 1

    survive_count = data_prep['survive'].sum()
    print(f"  - Surviving stores: {survive_count} / {len(data_prep)} ({survive_count/len(data_prep)*100:.1f}%)")

    # 8. day_diff 계산
    print("\n[Step 8] Calculating day_diff...")

    def get_last_day_from_yyyymm(yyyymm):
        """YYYYMM 형식에서 해당 월의 마지막 날짜를 YYYYMMDD로 반환"""
        if pd.isna(yyyymm):
            return None
        y = int(yyyymm // 100)
        m = int(yyyymm % 100)
        d = calendar.monthrange(y, m)[1]
        return y * 10000 + m * 100 + d

    replacement_dates = data_prep.loc[na_rows, 'TA_YM'].apply(get_last_day_from_yyyymm)
    data_prep.loc[na_rows, 'MCT_ME_D'] = replacement_dates

    date1_series = pd.to_datetime(data_prep['ARE_D'].astype(str), format='%Y%m%d', errors='coerce')
    date2_series = pd.to_datetime(data_prep['MCT_ME_D'].astype(str), format='%Y%m%d', errors='coerce')

    data_prep['day_diff'] = (date2_series - date1_series).dt.days.abs()
    print(f"  - Calculated day_diff for {len(data_prep)} rows")

    # 9. ARE_D, MCT_ME_D 제거
    print("\n[Step 9] Dropping temporary date columns...")
    data_prep = data_prep.drop(columns=['ARE_D', 'MCT_ME_D'])

    # 10. quaters 컬럼 생성
    print("\n[Step 10] Creating quarters column...")

    def calculate_quarter(row):
        """TA_YM에서 분기 계산"""
        mm = int(str(row['TA_YM'])[-2:])
        return ((mm - 1) // 3) + 1

    data_prep['quaters'] = data_prep.apply(calculate_quarter, axis=1)
    print(f"  - Created quarters column")

    # 11. 데이터 정렬 (TA_YM 기준)
    print("\n[Step 11] Sorting data by TA_YM...")
    data_prep = data_prep.sort_index().reset_index(drop=True)

    # 12. 내비게이션 데이터 병합
    print("\n[Step 12] Merging navigation search data...")
    navi_path = os.path.join(raw_data_dir, "2023~2024 성동구 내비게이션 목적지 유형별 검색량.csv")
    if os.path.exists(navi_path):
        navi = pd.read_csv(navi_path)
        navi = navi.loc[navi['목적지 유형'] == '음식'].sort_values(by='기준연월').drop(columns='목적지 유형')
        navi.columns = ['TA_YM', 'navi']
        print(f"  - Loaded navigation data: {navi.shape}")

        data_prep = data_prep.merge(navi, on='TA_YM', how='left')
        print(f"  - Merged navigation data. Shape: {data_prep.shape}")
    else:
        print(f"  - Warning: Navigation data file not found at {navi_path}")

    # 13. 관광객 수 데이터 병합
    print("\n[Step 13] Merging visitor data...")
    vis_path = os.path.join(raw_data_dir, "2023~2024 성동구 관광객 수.csv")
    if os.path.exists(vis_path):
        vis = pd.read_csv(vis_path)
        vis.drop(columns=['전년동월방문자수', '방문자수증감률'], inplace=True)
        vis.columns = ['TA_YM', 'vis']
        print(f"  - Loaded visitor data: {vis.shape}")

        data_prep = data_prep.merge(vis, on='TA_YM', how='left')
        print(f"  - Merged visitor data. Shape: {data_prep.shape}")
    else:
        print(f"  - Warning: Visitor data file not found at {vis_path}")

    # 14. 결과 저장
    print("\n[Step 14] Saving preprocessed data...")

    # 기존 파일 확인하여 버전 넘버링
    os.makedirs(output_dir, exist_ok=True)
    existing_files = list(Path(output_dir).glob("data_prep*.csv"))

    if existing_files:
        # 파일명에서 숫자 추출
        numbers = []
        for f in existing_files:
            try:
                num = int(f.stem.replace('data_prep', ''))
                numbers.append(num)
            except:
                pass
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1

    output_path = os.path.join(output_dir, f"data_prep{next_num}.csv")
    data_prep.to_csv(output_path, encoding='cp949', index=False)

    print(f"  - Saved to: {output_path}")
    print(f"  - Final shape: {data_prep.shape}")
    print(f"  - Columns: {list(data_prep.columns)}")

    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print("=" * 60)

    return data_prep, output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess raw data for XGBoost training')
    parser.add_argument('--raw-dir', type=str, default='./dataset/raw',
                       help='Directory containing raw data files')
    parser.add_argument('--output-dir', type=str, default='./dataset',
                       help='Directory to save preprocessed data')

    args = parser.parse_args()

    try:
        data_prep, output_path = run_preprocessing(args.raw_dir, args.output_dir)
        print(f"\nPreprocessed data saved to: {output_path}")
    except Exception as e:
        print(f"\nPreprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
