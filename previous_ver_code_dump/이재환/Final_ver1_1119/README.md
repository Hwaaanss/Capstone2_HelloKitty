# 모델 사용법
- vscode 환경에서 실행 권장
- 모든 작업 실행은 터미널에 커멘트를 입력 후 실행하는 것으로 작동하도록 설계해놓음. 그게 나중에 관리하기 편함.
- 터미널 명령어는 기기 환경에 따라 mac 환경은 python3 / pip3으로 시작하고, window 환경은 python/pip으로 시작하고, 이후는 동일하게 입력 후 실행하면 됨. 작성은 mac 환경 기준으로 작성됨.
- 안되면 캡스톤 단톡이나 개인톡으로 나한테 연락


## Install Libarary
pip3 install -r requirements.txt

만약 위 커멘드 실행 후 아래의 명령어들로 코드 실행 중 No module name ~ 뜨면 없다는 라이브러리만 따로 pip3 install {라이브러리 이름} 입력하면 됨.


## Preprocessing
python3 main.py --mode preprocessing

여기서 생성된 전처리 파일명을 ./dataset 디렉토리에 data_prep{n}.csv 명으로 n을 확인해야함. 
매 실행마다 파일명이 겹치지 않게 n을 바꾸도록 해놓고 전처리 과정 수정 시 반영된 별도 파일 생성을 위함. 이후 설명에서는 예시를 위해 "data_prep3.csv"를 사용.


## Hyperparameter Tuning with Optuna
python3 main.py --mode optuna --data {파일명} --n-trials {하이퍼파라미터 튜닝 시도 횟수}
(e.g.,) python3 main.py --mode optuna --data ./dataset/data_prep3.csv --n-trials 100

위 과정이 끝나면 ./optuna_results 디렉토리에 아래의 항목들이 저장됨.
1. best_params_{날짜}_{시각}.json : 최적의 하이퍼파라미터 조합
2. optimization_history_20251118_202602.png : 최적화 과정 시각화. x축은 trials 번호, y축은 loss, 파란점은 각 trial의 loss값, 빨간 선은 현재까지 최적값 의미.
3. param_importances_20251024_005251.png : 하이퍼파라미터 중요도
4. study_history_20251023_202557.csv : 각 학습 기록


## Training
python3 main.py --mode train --data {파일명} --config {내가 직접 설정한 하이퍼파라미터(config.json) or optuna로 찾은 최적 하이퍼파라미터(config_optimized.json)}
(e.g.,) python3 main.py --mode train --data ./dataset/data_prep3.csv --config config_optimized.json

위 과정이 끝나면 아래의 항목들이 저장됨.

### ./plots
1. xgboost_stacking_results.png : 학습 과정 그래프와 test score plot

### ./models
1. stacking_model.pkl : Stacking 앙상블 모델이 저장된 파일
2. stacking_metadata.pkl : Stacking 모델의 메타데이터가 저장된 파일

### ./
1. feature_importance.json : 베이스 모델 중 성능이 가장 좋은 모델의 feature importance


## Chat Bot
python3 main.py --mode streamlit

1. 실행 후 상단 점포 목록을 펼치고 임의의 점포코드 하나를 복사 후 하단 입력란에 붙혀넣고 우측 분석버튼 클릭
2. 매출 건강도 gauge plot 하단 채팅창에 질문 입력 후 아래 질문하기 버튼 클릭 (엔터로 입력이 안되는데, 디버깅 실패)
3. 조금 기다리면 하단에 답변 출력
4. 종료 시 웹 탭 닫은 다음 vscode 터미널에서 ctrl + c 로 keyboard interrupt 종료