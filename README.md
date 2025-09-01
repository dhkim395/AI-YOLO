# 😃 Emotion Recognition with YOLO & ResNet

---

## 📌 프로젝트 정보
- **프로젝트명**: Emotion Recognition with YOLO & ResNet  
- **기간**: 2025.08.14 ~ 2025.08.21  
- **설명**:  
  - 논문을 읽고 YOLOv1 또는 YOLOv2를 직접 구현하여 얼굴을 탐지하고,  
  - ResNet으로 감정을 분류하는 모델 구축하고,
  - opencv로 카메라 연결하여 실시간으로 얼굴을 인식하고 감정을 분류
 
- **팀원**:
   - 김동현:YOLOv1 구현, 얼굴 데이터 학습
   - 양하용:FastYOLO 구현, 얼굴 데이터 학습
   - 최리준: YOLOv2구현, 얼굴 데이터 학습
   - 이찬환: ResNet-18 모델 학습, opencv 연결
   - 김하경: 분류기 모델 학습, 결과 비교

---

## 🛠 기술 스택
- **Environment**: Git, VS Code, Jupyter Notebook / Colab  
- **Config / Build**: Python 3.10+, pip/conda  
- **Development**: PyTorch, ResNet-18, OpenCV, NumPy, Pandas, Matplotlib  
- **Communication**: Slack

---

## 📊 데이터셋
- **AI-Hub: 감정 분류** (감정 분류: 행복, 슬픔, 분노, 놀람, 중립 등)  
- **WIDER FACE** (얼굴 탐지 → YOLO 학습)  

---

## ✨ 주요 기능
- 🔍 **YOLO 기반 얼굴 탐지** (이미지/영상 내 얼굴 영역 추출)  
- 😃 **ResNet 기반 감정 분류** (Happy, Sad, Angry, Neutral 등 다중 클래스 분류)  
- 📈 **결과 시각화** (Bounding box + 감정 Label + 확률) , 카메라로 실시간 시각화
  
---

## 📑 발표자료 링크
- [yolo구현 발표과제.pdf](https://github.com/user-attachments/files/22070627/yolo.pdf)
