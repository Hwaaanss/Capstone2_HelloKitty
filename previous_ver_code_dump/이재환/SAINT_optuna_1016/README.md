## Info
### Model
Tabular 데이터 특화 Transformer 기반 딥러닝 모델인 SAINT 사용

### Setting
config.json 파일 안에 시스템 프롬프트, epoch, token 등 각종 설정 값들 다 있음.

### 사용 및 테스트 방법
1. SAINT_0925 폴더째로 다운로드
2. 구글 코랩과 같은 cell based IDE 가 아닌, vscode 처럼 .py 와 터미널(리눅스) 환경을 지원하는 IDE 에서 SAINT 폴더 열기
3. 모델 학습 시 (mac) python3 main.py --mode train --data ./dataset/BasicData_Prep.csv --target {예측 목표 컬럼명 ex. RC_M1_SAA}  (window) python3 -> python
4. 챗봇 테스트 시 (mac) python3 main.py --mode streamlit (window) python3 -> python
5. Train, validation 에 사용되지 않은 test 데이터셋의 가맹점 고유 코드를 먼저 확인하고 복사할 수 있는 창이 뜨면 아무 코드나 복사한 후 아래 입력창에 입력 후 분석버튼 클릭
6. 아래에 해당 가맹점 정보가 출력(실제 사용 시 신한카드 가맹점 정보 관련 DB에서 코드에 맞는 가맹점의 가장 최신 정보를 로드함을 가정하고 설계했음)
7. 아래에 질문 입력창이 나오면 마케팅 전략이나 이벤트 추천 등 질문 입력
8. 해당 매장 정보와 모델의 가중치를 바탕으로 답변 출력
