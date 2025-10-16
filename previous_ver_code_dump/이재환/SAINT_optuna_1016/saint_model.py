import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os
import sys
import pickle
import json

sys.path.append('./SAINT')
from SAINT.models import SAINT
from SAINT.data_openml import DataSetCatCon
from SAINT.augmentations import embed_data_mask
from SAINT.utils import classification_scores, mean_sq_error

from config import Config

class CustomSAINTDataset(Dataset):
    def __init__(self, X, y, cat_idxs, con_idxs, continuous_mean_std=None, task='reg'):
        self.X = X.values.astype(np.float32)
        self.y = y.reshape(-1, 1).astype(np.float32) if task == 'reg' else y.astype(np.int64)
        self.cat_idxs = cat_idxs
        self.con_idxs = con_idxs
        self.task = task

        # 카테고리컬과 연속형 데이터 분리
        if len(cat_idxs) > 0:
            self.X_cat = self.X[:, cat_idxs].astype(np.int64)
        else:
            self.X_cat = np.zeros((self.X.shape[0], 1), dtype=np.int64)

        if len(con_idxs) > 0:
            self.X_con = self.X[:, con_idxs].astype(np.float32)
            # 연속형 변수 정규화
            if continuous_mean_std is not None:
                mean, std = continuous_mean_std
                self.X_con = (self.X_con - mean) / std
        else:
            self.X_con = np.zeros((self.X.shape[0], 1), dtype=np.float32)

        # 마스크 생성 (모든 데이터가 유효하다고 가정)
        self.cat_mask = np.ones_like(self.X_cat, dtype=np.int64)
        self.con_mask = np.ones_like(self.X_con, dtype=np.int64)

        # CLS 토큰 추가
        self.cls = np.zeros((self.X.shape[0], 1), dtype=np.int64)
        self.cls_mask = np.ones((self.X.shape[0], 1), dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # CLS 토큰과 카테고리컬 데이터 결합
        x_categ = np.concatenate([self.cls[idx], self.X_cat[idx]])
        cat_mask = np.concatenate([self.cls_mask[idx], self.cat_mask[idx]])

        return (
            torch.tensor(x_categ, dtype=torch.long),  # x_categ
            torch.tensor(self.X_con[idx], dtype=torch.float32),  # x_cont
            torch.tensor(self.y[idx], dtype=torch.float32 if self.task == 'reg' else torch.long),  # y
            torch.tensor(cat_mask, dtype=torch.long),  # cat_mask
            torch.tensor(self.con_mask[idx], dtype=torch.long)  # con_mask
        )

class SAINTWrapper:
    def __init__(self, config: Config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = None
        self.train_history = {'loss': [], 'val_loss': []}
        self.metrics_history = {}
        self.model_save_path = "saint_model.pth"
        self.metadata_save_path = "saint_metadata.pkl"

        # SAINT 특정 변수들
        self.cat_dims = None
        self.cat_idxs = None
        self.con_idxs = None
        self.continuous_mean_std = None
        self.label_encoders = {}

    def _preprocess_data(self, df):
        """데이터 전처리: 타입 변환 및 정리"""
        df = df.copy()

        for col in df.columns:
            # 1. 결측값 처리
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['object', 'category']:
                    df[col].fillna('Unknown', inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)

            # 2. 데이터 타입 자동 감지 및 변환
            if df[col].dtype == 'object':
                # 숫자로 변환 가능한지 시도
                try:
                    # 모든 값이 숫자인지 확인
                    pd.to_numeric(df[col], errors='raise')
                    df[col] = pd.to_numeric(df[col])
                    print(f"🔢 Converted {col} to numeric")
                except:
                    # 변환 불가능하면 카테고리컬로 처리
                    unique_ratio = df[col].nunique() / len(df)

                    # ID같은 고유값이 너무 많으면 제외 고려
                    if unique_ratio > 0.9 and df[col].nunique() > 100:
                        print(f"⚠️ Warning: Column '{col}' has too many unique values ({df[col].nunique()}), might be ID column")
                        # 고유값이 너무 많으면 해당 컬럼 제거를 고려하거나 처리 방법 변경
                        # 여기서는 일단 카테고리컬로 처리하되, 상위 빈도 값들만 유지
                        top_categories = df[col].value_counts().head(50).index
                        df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
                        print(f"   -> Reduced to top 50 categories + 'Other'")

                    # 카테고리컬 데이터를 숫자로 인코딩
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    else:
                        # 이미 학습된 인코더로 변환 (예측 시)
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))

                    print(f"🏷️ Processed {col} as categorical ({df[col].nunique()} categories)")

            # 3. 매우 큰 숫자값들 스케일링 체크
            elif df[col].dtype in ['int64', 'float64']:
                if df[col].max() > 1e6:
                    print(f"ℹ️ Info: Large values detected in {col} (max: {df[col].max():.2e})")

        return df

    def prepare_data(self, df, target_columns, test_size=0.2, val_size=0.1):
        # ENCODED_MCT 컬럼이 있는지 확인
        if 'ENCODED_MCT' in df.columns:
            print("🏪 ENCODED_MCT 기준으로 데이터셋 분리")
            return self._prepare_data_by_store(df, target_columns, test_size, val_size)
        else:
            print("📊 일반적인 방법으로 데이터셋 분리")
            return self._prepare_data_standard(df, target_columns, test_size, val_size)

    def _prepare_data_by_store(self, df, target_columns, test_size=0.2, val_size=0.1):
        """ENCODED_MCT 기준으로 데이터셋을 분리 (점포별로 분리)"""
        X = df.drop(columns=target_columns)
        y = df[target_columns].values

        # 각 점포별로 최신 데이터(TA_YM 최대값) 찾기
        if 'TA_YM' in df.columns:
            max_ta_ym_per_store = df.groupby('ENCODED_MCT')['TA_YM'].max().reset_index()
            max_ta_ym_per_store.columns = ['ENCODED_MCT', 'max_TA_YM']

            # 원본 데이터와 merge하여 최신 데이터만 추출
            df_with_max = df.merge(max_ta_ym_per_store, on='ENCODED_MCT')
            test_indices = df_with_max[df_with_max['TA_YM'] == df_with_max['max_TA_YM']].index

            # Test 데이터 추출
            X_test = X.iloc[test_indices].copy()
            y_test = y[test_indices]

            # 나머지 데이터 (test에 없는 데이터들)
            remaining_indices = df.index.difference(test_indices)
            X_remaining = X.iloc[remaining_indices].copy()
            y_remaining = y[remaining_indices]

            print(f"🧪 Test set: {X_test.shape} (각 점포의 최신 데이터)")
            print(f"📋 Remaining data: {X_remaining.shape}")

            # 나머지 데이터를 점포 기준으로 train/validation 분리
            unique_stores = X_remaining['ENCODED_MCT'].unique()
            val_ratio = val_size / (1 - test_size)
            train_stores, val_stores = train_test_split(
                unique_stores, test_size=val_ratio, random_state=42
            )

            # 점포별로 데이터 분리
            train_mask = X_remaining['ENCODED_MCT'].isin(train_stores)
            X_train = X_remaining[train_mask].copy()
            y_train = y_remaining[train_mask]

            val_mask = X_remaining['ENCODED_MCT'].isin(val_stores)
            X_val = X_remaining[val_mask].copy()
            y_val = y_remaining[val_mask]

            print(f"🏋️ Train set: {X_train.shape} ({len(train_stores)} 점포)")
            print(f"✅ Validation set: {X_val.shape} ({len(val_stores)} 점포)")

        else:
            # TA_YM이 없으면 점포별로만 분리
            unique_stores = X['ENCODED_MCT'].unique()

            # 점포를 train/val/test로 분리
            test_stores, temp_stores = train_test_split(
                unique_stores, test_size=test_size, random_state=42
            )
            val_ratio = val_size / (1 - test_size)
            train_stores, val_stores = train_test_split(
                temp_stores, test_size=val_ratio, random_state=42
            )

            # 각 점포별로 데이터 분리
            X_train = X[X['ENCODED_MCT'].isin(train_stores)].copy()
            X_val = X[X['ENCODED_MCT'].isin(val_stores)].copy()
            X_test = X[X['ENCODED_MCT'].isin(test_stores)].copy()

            y_train = y[X['ENCODED_MCT'].isin(train_stores)]
            y_val = y[X['ENCODED_MCT'].isin(val_stores)]
            y_test = y[X['ENCODED_MCT'].isin(test_stores)]

        return self._finalize_data_preparation(X, X_train, X_val, X_test, y_train, y_val, y_test)

    def _prepare_data_standard(self, df, target_columns, test_size=0.2, val_size=0.1):
        """일반적인 방법으로 데이터셋 분리"""
        X = df.drop(columns=target_columns)
        y = df[target_columns].values

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42
        )

        return self._finalize_data_preparation(X, X_train, X_val, X_test, y_train, y_val, y_test)

    def _finalize_data_preparation(self, X, X_train, X_val, X_test, y_train, y_val, y_test):
        # 데이터 전처리 및 타입 정리 (분할 전에 전체 데이터로 기준 설정)
        X_processed = self._preprocess_data(X)

        # 전처리된 데이터로 다시 분할
        X_train_processed = X_processed.iloc[X_train.index]
        X_val_processed = X_processed.iloc[X_val.index]
        X_test_processed = X_processed.iloc[X_test.index]

        # 카테고리컬과 연속형 컬럼 분리
        # 인코딩된 카테고리컬 컬럼들을 찾기 위해 label_encoders를 사용
        categorical_columns = [col for col in X_processed.columns if col in self.label_encoders]
        continuous_columns = [col for col in X_processed.columns if col not in self.label_encoders]

        categorical_columns = pd.Index(categorical_columns)
        continuous_columns = pd.Index(continuous_columns)

        self.cat_idxs = [X_processed.columns.get_loc(col) for col in categorical_columns]
        self.con_idxs = [X_processed.columns.get_loc(col) for col in continuous_columns]

        print(f"🏷️ Categorical columns ({len(categorical_columns)}): {list(categorical_columns)}")
        print(f"🔢 Continuous columns ({len(continuous_columns)}): {list(continuous_columns)}")

        # 카테고리컬 차원 계산
        self.cat_dims = []
        for col in categorical_columns:
            unique_vals = X_processed[col].nunique()
            self.cat_dims.append(unique_vals)
            print(f"   {col}: {unique_vals} categories")

        # 연속형 변수 정규화를 위한 통계
        if len(continuous_columns) > 0:
            train_mean = X_train_processed[continuous_columns].mean().values.astype(np.float32)
            train_std = X_train_processed[continuous_columns].std().values.astype(np.float32)
            # std가 0인 경우 1로 설정
            train_std = np.where(train_std == 0, 1, train_std)
            self.continuous_mean_std = np.array([train_mean, train_std])
        else:
            self.continuous_mean_std = np.array([[0], [1]]).astype(np.float32)

        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test

    def create_data_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # 타겟 데이터 형태 맞추기
        if len(y_train.shape) == 2 and y_train.shape[1] == 1:
            y_train = y_train.flatten()
            y_val = y_val.flatten()
            y_test = y_test.flatten()

        # 마스크 생성 (모든 값이 존재한다고 가정)
        X_train_mask = np.ones(X_train.shape, dtype=np.int64)
        X_val_mask = np.ones(X_val.shape, dtype=np.int64)
        X_test_mask = np.ones(X_test.shape, dtype=np.int64)

        # SAINT DataSetCatCon 형식으로 변환
        X_train_dict = {
            'data': X_train.values.astype(np.float32),
            'mask': X_train_mask
        }
        y_train_dict = {'data': y_train.reshape(-1, 1).astype(np.float32)}

        X_val_dict = {
            'data': X_val.values.astype(np.float32),
            'mask': X_val_mask
        }
        y_val_dict = {'data': y_val.reshape(-1, 1).astype(np.float32)}

        X_test_dict = {
            'data': X_test.values.astype(np.float32),
            'mask': X_test_mask
        }
        y_test_dict = {'data': y_test.reshape(-1, 1).astype(np.float32)}

        # 태스크 타입 결정
        dtask = 'reg' if self.config.model.task == 'regression' else 'clf'

        try:
            train_ds = DataSetCatCon(X_train_dict, y_train_dict, self.cat_idxs, dtask, self.continuous_mean_std)
            trainloader = DataLoader(train_ds, batch_size=self.config.model.batch_size, shuffle=True, num_workers=4, pin_memory=True)

            valid_ds = DataSetCatCon(X_val_dict, y_val_dict, self.cat_idxs, dtask, self.continuous_mean_std)
            validloader = DataLoader(valid_ds, batch_size=self.config.model.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            test_ds = DataSetCatCon(X_test_dict, y_test_dict, self.cat_idxs, dtask, self.continuous_mean_std)
            testloader = DataLoader(test_ds, batch_size=self.config.model.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            return trainloader, validloader, testloader

        except Exception as e:
            print(f"⚠️ Warning: DataSetCatCon failed: {e}")
            print("🔄 Using custom DataLoader...")
            # 커스텀 DataLoader 사용
            return self._create_custom_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test)

    def _create_custom_data_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """커스텀 DataLoader 생성"""
        dtask = 'reg' if self.config.model.task == 'regression' else 'clf'

        train_ds = CustomSAINTDataset(X_train, y_train, self.cat_idxs, self.con_idxs, self.continuous_mean_std, dtask)
        trainloader = DataLoader(train_ds, batch_size=self.config.model.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        valid_ds = CustomSAINTDataset(X_val, y_val, self.cat_idxs, self.con_idxs, self.continuous_mean_std, dtask)
        validloader = DataLoader(valid_ds, batch_size=self.config.model.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        test_ds = CustomSAINTDataset(X_test, y_test, self.cat_idxs, self.con_idxs, self.continuous_mean_std, dtask)
        testloader = DataLoader(test_ds, batch_size=self.config.model.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        print("✅ Custom DataLoader created successfully")
        return trainloader, validloader, testloader

    def build_model(self, input_dim, output_dim=1):
        # y_dim 계산
        if self.config.model.task == 'regression':
            y_dim = 1
        else:
            y_dim = output_dim

        # cat_dims 준비 (CLS 토큰을 위해 앞에 1 추가)
        if not self.cat_dims:
            cat_dims_tuple = tuple([1])
        else:
            cat_dims_tuple = tuple(np.append(np.array([1]), np.array(self.cat_dims)).astype(int))

        self.model = SAINT(
            categories=cat_dims_tuple,
            num_continuous=len(self.con_idxs),
            dim=self.config.model.embedding_size,
            dim_out=output_dim,
            depth=self.config.model.transformer_depth,
            heads=self.config.model.attention_heads,
            attn_dropout=self.config.model.attention_dropout,
            ff_dropout=self.config.model.ff_dropout,
            mlp_hidden_mults=(4, 2),
            cont_embeddings='MLP',
            attentiontype='colrow',
            final_mlp_style='sep',
            y_dim=y_dim
        ).to(self.device)

        # 옵티마이저 설정
        if self.config.model.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.model.lr)
        elif self.config.model.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.model.lr)
        elif self.config.model.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.model.lr, momentum=0.9)

        # Loss function 설정
        if self.config.model.task == 'regression':
            self.criterion = nn.MSELoss().to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_model(self, X_train, y_train, X_val, y_val):
        trainloader, validloader, testloader = self.create_data_loaders(X_train, y_train, X_val, y_val, X_val, y_val)

        self.model.train()

        for epoch in range(self.config.model.epochs):
            running_loss = 0.0

            for i, data in enumerate(trainloader):
                self.optimizer.zero_grad()

                x_categ, x_cont, y_gts, cat_mask, con_mask = (
                    data[0].to(self.device),
                    data[1].to(self.device),
                    data[2].to(self.device),
                    data[3].to(self.device),
                    data[4].to(self.device)
                )

                # 데이터를 임베딩으로 변환
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model, False)

                # Transformer를 통과
                reps = self.model.transformer(x_categ_enc, x_cont_enc)

                # CLS 토큰에 해당하는 표현 선택
                y_reps = reps[:, 0, :]

                # MLP를 통해 예측
                y_outs = self.model.mlpfory(y_reps)

                if self.config.model.task == 'regression':
                    loss = self.criterion(y_outs, y_gts)
                else:
                    loss = self.criterion(y_outs, y_gts.squeeze().long())

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Validation
            val_loss = self.validate(validloader)

            self.train_history['loss'].append(running_loss / len(trainloader))
            self.train_history['val_loss'].append(val_loss)

            if self.config.model.verbose and epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{self.config.model.epochs}, Loss: {running_loss/len(trainloader):.4f}, Val Loss: {val_loss:.4f}')

    def validate(self, validloader):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in validloader:
                x_categ, x_cont, y_gts, cat_mask, con_mask = (
                    data[0].to(self.device),
                    data[1].to(self.device),
                    data[2].to(self.device),
                    data[3].to(self.device),
                    data[4].to(self.device)
                )

                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model, False)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = self.model.mlpfory(y_reps)

                if self.config.model.task == 'regression':
                    loss = self.criterion(y_outs, y_gts)
                else:
                    loss = self.criterion(y_outs, y_gts.squeeze().long())

                val_loss += loss.item()

        self.model.train()
        return val_loss / len(validloader)

    def predict(self, X):
        # 예측을 위한 더미 타겟 생성
        dummy_y = np.zeros(X.shape[0])

        # 전처리 적용
        X_processed = self._preprocess_data(X)

        # 커스텀 DataLoader 사용
        dtask = 'reg' if self.config.model.task == 'regression' else 'clf'
        test_ds = CustomSAINTDataset(X_processed, dummy_y, self.cat_idxs, self.con_idxs, self.continuous_mean_std, dtask)
        testloader = DataLoader(test_ds, batch_size=self.config.model.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        self.model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for data in testloader:
                x_categ, x_cont, _, cat_mask, con_mask = (
                    data[0].to(self.device),
                    data[1].to(self.device),
                    data[2].to(self.device),
                    data[3].to(self.device),
                    data[4].to(self.device)
                )

                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model, False)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = self.model.mlpfory(y_reps)

                if self.config.model.task == 'regression':
                    predictions.extend(y_outs.cpu().numpy().flatten())
                else:
                    probs = torch.softmax(y_outs, dim=1)
                    probabilities.extend(probs.cpu().numpy())
                    predictions.extend(torch.argmax(y_outs, dim=1).cpu().numpy())

        if self.config.model.task == 'regression':
            return np.array(predictions), None
        else:
            return np.array(predictions), np.array(probabilities)

    def evaluate_model(self, X_test, y_test):
        predictions, probabilities = self.predict(X_test)

        # y_test가 2차원이면 1차원으로 변환
        if len(y_test.shape) == 2 and y_test.shape[1] == 1:
            y_test = y_test.flatten()

        metrics = {}

        if self.config.model.task == 'regression':
            metrics['mse'] = mean_squared_error(y_test, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_test, predictions)
        else:
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            try:
                metrics['f1'] = f1_score(y_test, predictions, average='weighted')
                metrics['precision'] = precision_score(y_test, predictions, average='weighted')
                metrics['recall'] = recall_score(y_test, predictions, average='weighted')
            except:
                metrics['f1'] = 0.0
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0

        self.metrics_history = metrics

        if self.config.model.verbose:
            print("📊 Model Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")

        return metrics, predictions, probabilities

    def get_feature_importance(self, X):
        # SAINT는 attention weights를 통한 feature importance 계산이 복잡하므로
        # 임시로 랜덤 importance 생성 (실제로는 attention weights를 분석해야 함)
        importances = np.random.random(X.shape[1])
        importances = importances / importances.sum()
        return importances

    def plot_training_history(self):
        if not os.path.exists(self.config.ui.plots_save_dir):
            os.makedirs(self.config.ui.plots_save_dir)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 학습 곡선
        ax1.plot(self.train_history['loss'], label='Training Loss')
        ax1.plot(self.train_history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # 메트릭
        if self.metrics_history:
            metrics_names = list(self.metrics_history.keys())
            metrics_values = list(self.metrics_history.values())

            ax2.bar(metrics_names, metrics_values)
            ax2.set_title('Model Metrics')
            ax2.set_ylabel('Score')

            for i, v in enumerate(metrics_values):
                ax2.text(i, v + max(metrics_values) * 0.01, f'{v:.3f}', ha='center')

        plt.tight_layout()

        if self.config.ui.save_plots_image:
            plt.savefig(f"{self.config.ui.plots_save_dir}/saint_training_results.png", dpi=300, bbox_inches='tight')

        if self.config.ui.show_plots_popup:
            plt.show()
        else:
            plt.close()

        return fig

    def save_model(self, save_dir="./models"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, "saint_model.pth")
        metadata_path = os.path.join(save_dir, "saint_metadata.pkl")

        if self.model is None:
            print("⚠️ Warning: No model to save!")
            return False

        try:
            torch.save(self.model.state_dict(), model_path)

            metadata = {
                'cat_idxs': self.cat_idxs,
                'con_idxs': self.con_idxs,
                'cat_dims': self.cat_dims,
                'continuous_mean_std': self.continuous_mean_std,
                'train_history': self.train_history,
                'metrics_history': self.metrics_history,
                'label_encoders': self.label_encoders,
            }

            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            print(f"💾 Model saved to {model_path}")
            print(f"📋 Metadata saved to {metadata_path}")
            return True

        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False

    def load_model(self, save_dir="./models"):
        model_path = os.path.join(save_dir, "saint_model.pth")
        metadata_path = os.path.join(save_dir, "saint_metadata.pkl")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            print(f"❌ Error: Model files not found in {save_dir}")
            return False

        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.cat_idxs = metadata['cat_idxs']
            self.con_idxs = metadata['con_idxs']
            self.cat_dims = metadata['cat_dims']
            self.continuous_mean_std = metadata['continuous_mean_std']
            self.train_history = metadata['train_history']
            self.metrics_history = metadata['metrics_history']
            self.label_encoders = metadata.get('label_encoders', {})

            # 모델 재구축
            output_dim = 1 if self.config.model.task == 'regression' else 2
            self.build_model(len(self.cat_idxs) + len(self.con_idxs), output_dim)

            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

            print(f"✅ Model loaded from {model_path}")
            print(f"📋 Metadata loaded from {metadata_path}")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False

    def model_exists(self, save_dir="./models"):
        model_path = os.path.join(save_dir, "saint_model.pth")
        metadata_path = os.path.join(save_dir, "saint_metadata.pkl")
        return os.path.exists(model_path) and os.path.exists(metadata_path)