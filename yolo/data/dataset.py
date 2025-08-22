import os, glob
import torch
from PIL import Image
from torchvision import transforms
from .encoding import corners_to_center_norm, encode_targets_yolov1
from pathlib import Path

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

class YoloV1TxtDataset(torch.utils.data.Dataset):
    """
    images_dir: path/to/images
    labels_dir: path/to/labels (same stem name .txt)
    txt line:
      - 'voc_corners': class x1 y1 x2 y2   (픽셀 또는 [0,1], normalized 플래그로 제어)
      - 'yolo_center': class cx cy w h     (항상 [0,1] 가정)
    """
    def __init__(self, images_dir, labels_dir, S=7, C=20, img_size=448, normalized=False, label_format='voc_corners', recursive: bool = False):
        self.img_paths = sorted(glob.glob(os.path.join(images_dir, '*')))
        self.labels_dir = labels_dir
        self.S, self.C = S, C
        self.img_size = img_size
        self.normalized = normalized
        self.label_format = label_format  # ▶ 추가
        self.recursive = recursive

        # ★ 이미지 목록 만들기 (재귀/확장자 필터)
        p = Path(images_dir)
        if not p.exists():
            raise FileNotFoundError(f"images_dir not found: {images_dir}")

        pattern = "**/*" if recursive else "*"
        self.img_paths = sorted(
            [str(q) for q in p.glob(pattern) if q.suffix.lower() in VALID_EXTS]
        )

        if len(self.img_paths) == 0:
            raise ValueError(
                f"No images found in '{images_dir}'. "
                f"(recursive={recursive}, valid_exts={sorted(VALID_EXTS)})"
            )

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self): return len(self.img_paths)

    def _read_labels(self, stem):
        p = os.path.join(self.labels_dir, stem + '.txt')
        boxes = []
        if os.path.exists(p):
            with open(p,'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                    cid,x1,y1,x2,y2 = parts
                    boxes.append([float(cid), float(x1), float(y1), float(x2), float(y2)])
        return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,5), dtype=torch.float32)

    def __getitem__(self, idx):
        ip = self.img_paths[idx]
        stem = os.path.splitext(os.path.basename(ip))[0]
        img = Image.open(ip).convert('RGB')
        W,H = img.size
        lab_raw = self._read_labels(stem)  # (M,5)

        if self.label_format == 'yolo_center':
            # YOLO 형식: 이미 [0,1]의 (class, cx, cy, w, h)
            lab_cxcywh = lab_raw.clone().float()
            # 안전 클램프
            if lab_cxcywh.numel() > 0:
                lab_cxcywh[:,1:5].clamp_(min=1e-6, max=1-1e-6)
        else:
            # VOC 코너 → 센터 정규화로 변환
            lab_cxcywh = corners_to_center_norm(lab_raw, (W,H), normalized=self.normalized)

        img = self.tf(img)  # (3, img_size, img_size)
        target = encode_targets_yolov1(lab_cxcywh, S=self.S, C=self.C)  # (S,S,5+C)
        return img, target