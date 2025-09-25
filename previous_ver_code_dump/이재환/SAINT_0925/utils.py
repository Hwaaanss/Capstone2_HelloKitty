import pandas as pd
import chardet
import os

def detect_encoding(file_path, sample_size=10000):
    """파일의 인코딩을 자동 감지"""
    try:
        with open(file_path, 'rb') as f:
            sample = f.read(sample_size)
        result = chardet.detect(sample)
        return result['encoding']
    except:
        return None

def load_csv_with_encoding(file_path):
    """다양한 인코딩을 시도하여 CSV 파일 로드"""
    # 1. 자동 감지 시도
    detected_encoding = detect_encoding(file_path)
    if detected_encoding:
        try:
            df = pd.read_csv(file_path, encoding=detected_encoding)
            print(f"✅ Data loaded with detected encoding '{detected_encoding}': {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except:
            pass

    # 2. 일반적인 인코딩들 순서대로 시도
    encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig', 'latin-1', 'iso-8859-1']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"✅ Data loaded with {encoding} encoding: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Warning: Failed to load with {encoding}: {str(e)}")
            continue

    # 3. 모든 인코딩 실패 시
    raise ValueError(f"Error: Could not load {file_path} with any supported encoding")

def load_csv_from_uploaded_file(uploaded_file):
    """Streamlit 업로드 파일에서 CSV 로드"""
    encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig', 'latin-1']

    for encoding in encodings:
        try:
            uploaded_file.seek(0)  # 파일 포인터 초기화
            df = pd.read_csv(uploaded_file, encoding=encoding)
            return df, encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            continue

    return None, None

def print_dataset_info(df):
    """데이터셋 정보 출력"""
    print(f"📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"📋 Columns: {list(df.columns)}")

    print(f"🔢 Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"   {col}: {dtype}")

    print(f"❓ Missing Values:")
    missing = df.isnull().sum()
    for col, missing_count in missing.items():
        if missing_count > 0:
            print(f"   {col}: {missing_count} ({missing_count/len(df)*100:.1f}%)")

def suggest_target_columns(df):
    """타겟 컬럼 후보 제안"""
    possible_targets = []

    # 일반적인 타겟 컬럼명들
    target_keywords = ['target', '타겟', 'label', '라벨', 'y', 'output', '결과',
                      '매출', '수익', '점수', 'score', 'price', '가격', '금액']

    for col in df.columns:
        col_lower = col.lower()
        for keyword in target_keywords:
            if keyword in col_lower:
                possible_targets.append(col)
                break

    return possible_targets