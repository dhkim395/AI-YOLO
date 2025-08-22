# yolo/api.py
import os, glob
from typing import Optional, Union, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw

from .models.fast_yolov1_fcn import FastYOLOv1_FCN
from .loss.yolov1_loss import YoloV1Loss
from .data.dataset import YoloV1TxtDataset
from .engine.train import train_yolov1
from .utils.postprocess import decode_yolov1
from .utils.plot import plot_training_curves
from pathlib import Path

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def train_yolov1_fn(
    images_dir: str,
    labels_dir: str,
    val_images_dir: Optional[str] = None,
    val_labels_dir: Optional[str] = None,
    *,
    # 데이터/모델
    num_classes: int = 1,
    img_size: int = 448,
    label_format: str = "voc_corners",  # 'voc_corners' | 'yolo_center'
    normalized_labels: bool = False,     # label_format='voc_corners'일 때만 의미 있음
    B: int = 2,
    # 러닝 하이퍼파라미터
    batch: int = 16,
    epochs: int = 100,
    optimizer: str = "sgd",              # 'sgd' | 'adam'
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    scheduler_kind: str = "cosine",      # 'cosine' | 'multistep' | None
    warmup_epochs: int = 3,
    amp: bool = True,
    # 시스템/입출력
    workers: int = 2,
    val_workers: int = 2,
    device: Optional[str] = None,
    save_path: Optional[str] = None,     # 'weights_last.pt' 등
    plot_out_dir: str = "runs/train",
    return_history: bool = False,
):
    """
    YOLOv1-FCN 학습 함수.
    - label_format:
        'yolo_center' : 라벨이 [class cx cy w h], 모두 [0,1] 정규화
        'voc_corners' : 라벨이 [class x1 y1 x2 y2], 픽셀 또는 [0,1] (normalized_labels로 제어)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert img_size % 64 == 0, "입력 해상도는 64의 배수 권장 (예: 448→S=7)"
    S = img_size // 64

    # Datasets & Loaders
    train_set = YoloV1TxtDataset(
        images_dir, labels_dir, S=S, C=num_classes,
        img_size=img_size, normalized=normalized_labels, label_format=label_format, recursive=True
    )
    train_loader = DataLoader(
        train_set, batch_size=batch, shuffle=True,
        num_workers=workers, pin_memory=True
    )

    val_loader = None
    if val_images_dir and val_labels_dir:
        val_set = YoloV1TxtDataset(
            val_images_dir, val_labels_dir, S=S, C=num_classes,
            img_size=img_size, normalized=normalized_labels, label_format=label_format, recursive=True
        )
        val_loader = DataLoader(
            val_set, batch_size=batch, shuffle=False,
            num_workers=val_workers, pin_memory=True
        )
    
    if len(train_set) == 0:
        raise ValueError(f"[train_yolov1_fn] No training images found under: {images_dir}")

    if val_images_dir and val_labels_dir and len(val_set) == 0:
        print(f"[train_yolov1_fn] Warning: no validation images found under: {val_images_dir}")

    # Model & Loss
    model = FastYOLOv1_FCN(num_classes=num_classes, B=B)
    criterion = YoloV1Loss(S=S, B=B, C=num_classes, lambda_coord=5.0, lambda_noobj=0.5)

    # Train (히스토리 반환)
    history = train_yolov1(
        model, criterion, train_loader, val_loader=val_loader,
        epochs=epochs, optimizer_kind=optimizer, base_lr=lr,
        weight_decay=weight_decay, momentum=momentum,
        scheduler_kind=scheduler_kind, warmup_epochs=warmup_epochs,
        amp=amp, device=device
    )

    # 곡선/CSV 저장
    plot_training_curves(history, out_dir=plot_out_dir)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({"model": model.state_dict()}, save_path)
        print(f"[train_yolov1_fn] saved: {save_path}")

    return (model, history) if return_history else model


def load_model(
    weights: Union[str, Dict[str, torch.Tensor]],
    num_classes: int = 1,
    B: int = 2,
    device: Optional[str] = None,
    eval_mode: bool = True
) -> FastYOLOv1_FCN:
    """
    가중치에서 FastYOLOv1_FCN 모델을 생성/로드.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FastYOLOv1_FCN(num_classes=num_classes, B=B).to(device)
    if isinstance(weights, str):
        ckpt = torch.load(weights, map_location=device)
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    else:
        state = weights
    model.load_state_dict(state)
    if eval_mode:
        model.eval()
    return model


def infer_folder_fn(
    model_or_weights,
    images_dir: str,
    *,
    out_dir: Optional[str] = "runs/infer",
    num_classes: int = 1,
    B: int = 2,
    img_size: int = 448,
    # conf: float = 0.25,
    # iou: float = 0.45,
    conf: float = 0.30,
    iou: float = 0.55,
    device: Optional[str] = None,
    draw: bool = True,
    class_names: Optional[List[str]] = None,
    recursive: bool = False,     # ★ 추가
) -> Dict[str, torch.Tensor]:

    if images_dir is None:
        raise ValueError("[infer] images_dir is None. Pass a valid folder or file path.")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 모델 준비
    if isinstance(model_or_weights, FastYOLOv1_FCN):
        model = model_or_weights.to(device)
        model.eval()
    else:
        model = load_model(model_or_weights, num_classes=num_classes, B=B, device=device, eval_mode=True)

    # 전처리
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    os.makedirs(out_dir or "runs/infer", exist_ok=True)
    results: Dict[str, torch.Tensor] = {}

    # ★ 폴더/파일 경로 안전 처리 + 재귀/확장자 필터
    p = Path(images_dir)
    if not p.exists():
        raise FileNotFoundError(f"[infer] images_dir not found: {images_dir}")

    if p.is_file():
        img_paths = [str(p)]
    else:
        pattern = "**/*" if recursive else "*"
        img_paths = [str(q) for q in p.glob(pattern) if q.suffix.lower() in VALID_EXTS]

    if len(img_paths) == 0:
        raise ValueError(f"[infer] No images found in '{images_dir}' (recursive={recursive}, valid_exts={sorted(VALID_EXTS)})")

    with torch.no_grad():
        for ip in sorted(img_paths):
            img = Image.open(ip).convert("RGB")
            W, H = img.size
            x = tf(img).unsqueeze(0).to(device)
            pred = model(x)[0].cpu()

            dets = decode_yolov1(pred, img_size=(W, H), B=B, C=num_classes, conf_thr=conf, iou_thr=iou)
            results[ip] = dets

            if draw:
                vis = img.copy()
                draw_ = ImageDraw.Draw(vis)
                for x1, y1, x2, y2, score, cls in dets.tolist():
                    cls_int = int(cls)
                    label = class_names[cls_int] if (class_names and 0 <= cls_int < len(class_names)) else str(cls_int)
                    draw_.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw_.text((x1, max(0, y1 - 12)), f"{label} {score:.2f}", fill="red")
                vis.save(os.path.join(out_dir or "runs/infer", os.path.basename(ip)))

    if draw:
        print(f"[infer_folder_fn] saved visualizations to: {out_dir}")

    return results