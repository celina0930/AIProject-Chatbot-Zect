- 딥러닝 챗봇 시스템
- (프론트엔드) html,css,javascript
- (백엔드) Flask로 웹 서버 구현

----------

파일 실행 관련

env : ``py-study``

``python main.py``

- 구현 완료
  - 계정 생성 (정보입력: 이메일, 비밀번호, 연령대, 성별)
  - 로그인, 로그아웃
  - 사용자 텍스트 입력받고 감정 분류
  - 챗봇 대화

-------------------

[System Configuration]

<img src="https://github.com/celina0930/Chatbot_v1/blob/main/static/images/AIproject_model-System%20Configuration.drawio.png">

------

사용한 데이터셋

- AIHUB- 감성대화말뭉치 (https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)
- AIHUB - 웰니스 대화 스크립트 데이터셋(https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=267)
- GITHUB - 챗봇데이터(https://github.com/songys/Chatbot_data)

딥러닝 모델

- 감정 분류 :  GPT2
- 챗봇 대화 : Transformers
