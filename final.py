# webcam_emotion_yolo_my.py
import os
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ▶ 너의 YOLO 패키지에서 불러오기
from yolo import load_model
from yolo.utils.postprocess import decode_yolov1

# ----------------------------
# 1) 장치 설정
# ----------------------------
device = (
    torch.device("cuda") if torch.cuda.is_available() else
    (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
)
print(f"[INFO] Using device: {device}")

# ----------------------------
# 2) ResNet18Dropout 정의 & 로드
# ----------------------------
class ResNet18Dropout(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# RESNET_PATH = r"C:\Gukbi\MyDetector\resnet_emotion_dropout_faces.pth"
RESNET_PATH = r"C:\Gukbi\MyDetector\yolo_face_test.pth" # 새로운 가중치

num_classes_emotion = 7
class_names = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

resnet = ResNet18Dropout(num_classes_emotion)
resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device))
resnet.to(device).eval()

# 감정 분류기 입력 전처리 (학습과 동일)
emo_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ----------------------------q
# 3) 너의 YOLOv1-FCN 로드
# ----------------------------
YOLO_WEIGHTS = r"C:\Gukbi\MyDetector\weights_last.pt"  # 네 가중치 경로로 교체
yolo_num_classes = 1  # 얼굴 1 클래스
yolo_B = 2
yolo_img_size = 448   # 학습과 동일 권장(64의 배수)

yolo_model = load_model(
    weights=YOLO_WEIGHTS,
    num_classes=yolo_num_classes,
    B=yolo_B,
    device=str(device),
    eval_mode=True
)

# YOLO 입력 전처리 (stretch resize: 기존 파이프라인과 동일)
def preprocess_for_yolo(frame_bgr, img_size):
    H, W = frame_bgr.shape[:2]
    img = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))[None, ...]  # (1,C,H,W)
    tensor = torch.from_numpy(img)
    return tensor, (W, H)   # 원본 크기 반환

# ----------------------------
# 4) 웹캠 / 영상 캡처
# ----------------------------
cap = cv2.VideoCapture(0)  # 0=내장 카메라, 경로 넣으면 파일
if not cap.isOpened():
    raise FileNotFoundError("Cannot open camera/video source 0")

CONF_THR = 0.60
IOU_THR  = 0.45

with torch.inference_mode():
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ---- YOLO 탐지 ----
        inp, (origW, origH) = preprocess_for_yolo(frame, yolo_img_size)
        inp = inp.to(device, non_blocking=True)
        pred = yolo_model(inp)[0].detach().cpu()  # (S,S,B*5+C)

        # 원본 좌표계로 복원하여 디코드
        dets = decode_yolov1(
            pred,
            img_size=(origW, origH),
            B=yolo_B,
            C=yolo_num_classes,
            conf_thr=CONF_THR,
            iou_thr=IOU_THR
        )
        # 좌표 정수화(선택)
        if dets.numel():
            dets[:, :4] = dets[:, :4].round()

        # ---- 감정 분류 + 표시 ----
        if dets.numel():
            for x1, y1, x2, y2, score, cls in dets.tolist():
                x1 = max(0, min(origW-1, int(x1)))
                y1 = max(0, min(origH-1, int(y1)))
                x2 = max(0, min(origW-1, int(x2)))
                y2 = max(0, min(origH-1, int(y2)))
                if x2 <= x1 or y2 <= y1:
                    continue

                # 얼굴 크롭 (RGB로 변환해서 PIL 생성)
                face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[y1:y2, x1:x2]
                if face_rgb.size == 0:
                    continue
                face_pil = Image.fromarray(face_rgb)

                face_tensor = emo_transform(face_pil).unsqueeze(0).to(device)
                outputs = resnet(face_tensor)
                probs = F.softmax(outputs, dim=1)
                conf_score, pred_idx = torch.max(probs, 1)
                label = class_names[pred_idx.item()]
                conf_score = conf_score.item()

                # 박스 + 레이블 그리기
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf_score:.2f}",
                            (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,0), 2)

        cv2.imshow("Webcam Emotion (YOLOv1-FCN + ResNet18Dropout)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()