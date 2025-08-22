from yolo import train_yolov1_fn, infer_folder_fn, load_model
import torch
from pathlib import Path

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print(torch.version.cuda)   # 빌드된 CUDA 버전
print(torch.backends.cudnn.version())  # cuDNN 버전
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 네가 선언한 경로
train_images_dir = r"C:\Gukbi\MyDetector\Dataset\images\train"
train_labels_dir = r"C:\Gukbi\MyDetector\Dataset\labels\train"
val_images_dir   = None    
val_labels_dir   = None
# val_images_dir   = "/Dataset/images/val"    # 없으면 None로 두면 됨
# val_labels_dir   = "/Dataset/labels/val"

# 1) 학습 (YOLO 정규화 라벨 -> label_format='yolo_center')
def main():
    model, history = train_yolov1_fn(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        val_images_dir=val_images_dir,          # 검증셋 없으면 None
        val_labels_dir=val_labels_dir,          # 검증셋 없으면 None

        num_classes=1,      # 얼굴 1클래스
        img_size=448,
        label_format='yolo_center',   # ★ 중요: YOLO 정규화 라벨
        normalized_labels=False,      # yolo_center일 땐 무시됨
        B=2,

        batch=16,
        epochs=50,
        optimizer="sgd",
        lr=0.01,
        weight_decay=5e-4,
        momentum=0.9,
        scheduler_kind='cosine',
        warmup_epochs=3,
        amp=True,

        workers=4,
        val_workers=2,
        device=None,                  # 자동 선택(cuda/CPU)
        save_path="weights_last.pt",  # 가중치 저장
        plot_out_dir="runs/train_face",
        return_history=True           # history도 함께 받기
    )

# infer_images_dir =r"C:\Gukbi\MyDetector\Dataset\infer_images"
infer_images_dir =r"C:\Gukbi\MyDetector\Dataset\sample"
# 2) 추론 (가중치 경로 또는 model 객체 사용 가능)
def infer():
    results = infer_folder_fn(
        model_or_weights="weights_last.pt",      # 또는 model
        images_dir=infer_images_dir,               # 검증 폴더
        recursive=True,
        # out_dir="runs/infer_face",
        out_dir=".",
        num_classes=1,
        B=2,
        img_size=448,
        conf=0.51,
        iou=0.45,
        draw=True,
        class_names=["face"]                     # 시각화 라벨 텍스트
    )
    return results #["/path/to/img.jpg"] -> Tensor [M,6]  (x1,y1,x2,y2,score,cls)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()                       # 윈도우 권장
    # mp.set_start_method("spawn", force=True)  # 선택: 명시적 spawn
    # main()
    result = infer()
    boxes=[]
    for ip, t in result.items():
        if t.numel():
            t[:, :4] = torch.round(t[:, :4])  
            boxes.append(t[:, :4])
    # boxes = result.values()
    # print("결과:", result)
    print("결과:", boxes)