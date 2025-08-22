# import torch
# from .nms import nms_per_class, box_iou_xyxy  # box_iou_xyxy는 통합 NMS용

# def decode_yolov1(pred, img_size, B=2, C=20, conf_thr=0.25, iou_thr=0.45, per_class_nms=True):
#     """
#     pred: (S,S,B*5+C) channels-last
#     return: (M,6) [x1,y1,x2,y2,score,cls]
#     """
#     S = pred.shape[0]
#     W,H = img_size
#     box = pred[...,:B*5].view(S,S,B,5)
#     cls_logits = pred[...,B*5:]
#     probs = torch.softmax(cls_logits, dim=-1)  # (S,S,C)

#     dets=[]
#     for j in range(S):
#         for i in range(S):
#             pcls = probs[j,i]
#             for b in range(B):
#                 x,y,w,h,conf = box[j,i,b]
#                 if conf <= 0: continue
#                 cx = (i + x)/S * W; cy = (j + y)/S * H
#                 bw = w*W; bh = h*H
#                 x1 = cx - bw/2; y1 = cy - bh/2; x2 = cx + bw/2; y2 = cy + bh/2
#                 score_vec = conf * pcls
#                 score, cls = torch.max(score_vec, dim=0)
#                 if score.item() >= conf_thr:
#                     dets.append([x1.item(),y1.item(),x2.item(),y2.item(),score.item(), float(int(cls.item()))])
#     if not dets:
#         return torch.zeros((0,6), dtype=torch.float32)
#     dets = torch.tensor(dets, dtype=torch.float32)
#     return nms_per_class(dets, iou_thr=iou_thr) if per_class_nms else dets


#####################################################################버전2

import torch
from .nms import nms_per_class

def decode_yolov1(
    pred, img_size, B=2, C=1,
    conf_thr=0.10,          # 일단 낮춰서 많이 남기고
    iou_thr=0.60,           # NMS 임계 조금 올려서 큰박스가 작은얼굴 지우는 거 완화
    per_class_nms=True,
    *,
    use_sigmoid_xy=True,    # x,y를 [0,1]로 보정
    use_sigmoid_conf=True,  # conf를 [0,1]로 보정
    use_sqrt_wh=True        # 학습이 √w,h였다면 True
):
    """
    pred: (S,S,B*5+C) channels-last
    return: (M,6) [x1,y1,x2,y2,score,cls]
    """
    S = pred.shape[0]
    W, H = img_size

    box = pred[...,:B*5].view(S,S,B,5).clone()
    cls_logits = pred[...,B*5:]

    # 클래스 확률
    if C > 1:
        probs = torch.softmax(cls_logits, dim=-1)  # (S,S,C)
    else:
        # 단일 클래스면 항상 1
        probs = torch.ones(S, S, 1, dtype=pred.dtype, device=pred.device)

    dets = []
    for j in range(S):      # y-index (row)
        for i in range(S):  # x-index (col)
            pcls = probs[j,i]  # (C,)
            for b in range(B):
                tx, ty, tw, th, tobj = box[j,i,b]

                # 활성화/복원
                x = torch.sigmoid(tx) if use_sigmoid_xy else tx
                y = torch.sigmoid(ty) if use_sigmoid_xy else ty

                if use_sqrt_wh:
                    # 음수/NaN 방지
                    w = torch.clamp(tw, min=1e-6) ** 2
                    h = torch.clamp(th, min=1e-6) ** 2
                else:
                    w = torch.clamp(tw, min=1e-6)

                    h = torch.clamp(th, min=1e-6)

                conf = torch.sigmoid(tobj) if use_sigmoid_conf else tobj

                # 센터/크기 to 픽셀
                cx = (i + x) / S * W
                cy = (j + y) / S * H
                bw = w * W
                bh = h * H

                x1 = cx - bw/2
                y1 = cy - bh/2
                x2 = cx + bw/2
                y2 = cy + bh/2

                # 점수(클래스 포함)
                score_vec = conf * pcls  # (C,)
                score, cls = torch.max(score_vec, dim=0)

                s = float(score.item())
                if s >= conf_thr and bw > 1 and bh > 1:
                    dets.append([x1.item(), y1.item(), x2.item(), y2.item(), s, float(int(cls.item()))])

    if not dets:
        return torch.zeros((0,6), dtype=torch.float32)

    dets = torch.tensor(dets, dtype=torch.float32)
    return nms_per_class(dets, iou_thr=iou_thr) if per_class_nms else dets

################################################### 버전 3

# import torch
# # torchvision이 가능하면 그걸 써서 NMS 검증도 쉽게 가능
# try:
#     import torchvision
#     HAS_TV = True
# except Exception:
#     HAS_TV = False

# def nms_per_class_safe(dets: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
#     """
#     dets: [N,6] (x1,y1,x2,y2,score,cls)
#     """
#     if dets.numel() == 0:
#         return dets
#     out = []
#     classes = dets[:, 5].unique()
#     for c in classes:
#         m = dets[:, 5] == c
#         d = dets[m]
#         boxes, scores = d[:, :4], d[:, 4]
#         if HAS_TV:
#             keep = torchvision.ops.nms(boxes, scores, iou_thr)
#             out.append(d[keep])
#         else:
#             # 너의 기존 nms_per_class()로 대체해도 됨
#             out.append(d)  # 최소 보존 (임시)
#     return torch.cat(out, 0) if out else dets


# def decode_yolov1(
#     pred, img_size, B=2, C=1,
#     conf_thr=0.25, iou_thr=0.60, per_class_nms=True,
#     *,
#     use_sigmoid_xy=True,      # x,y를 [0,1]
#     use_sigmoid_conf=True,    # conf를 [0,1]
#     use_sqrt_wh=True,         # √w,h 회귀였다면 True
#     sigmoid_wh=True,          # tw,th에도 sigmoid 적용 후 제곱 → [0,1] 보장
#     clip_boxes=True,          # 이미지 경계로 클램프
#     min_wh=2,                 # 너무 작은 박스 제거(px)
#     max_area_ratio=0.95,      # 이미지의 95% 초과 박스 제거
#     topk_pre_nms=300          # NMS 전에 score 상위 K개만 사용
# ):
#     """
#     pred: (S,S,B*5+C)  channels-last
#     return: (M,6)  [x1,y1,x2,y2,score,cls]
#     """
#     S = pred.shape[0]
#     W, H = img_size

#     box = pred[..., :B*5].view(S, S, B, 5).clone()
#     cls_logits = pred[..., B*5:]

#     # 클래스 확률
#     if C > 1:
#         probs = torch.softmax(cls_logits, dim=-1)
#     else:
#         probs = torch.ones(S, S, 1, dtype=pred.dtype, device=pred.device)

#     dets = []
#     for gy in range(S):
#         for gx in range(S):
#             pcls = probs[gy, gx]  # (C,)
#             for b in range(B):
#                 tx, ty, tw, th, tobj = box[gy, gx, b]

#                 # 1) 좌표/크기 복원
#                 x = torch.sigmoid(tx) if use_sigmoid_xy else tx
#                 y = torch.sigmoid(ty) if use_sigmoid_xy else ty

#                 if use_sqrt_wh:
#                     if sigmoid_wh:
#                         w = torch.sigmoid(tw) ** 2
#                         h = torch.sigmoid(th) ** 2
#                     else:
#                         w = torch.clamp(tw, min=1e-6) ** 2
#                         h = torch.clamp(th, min=1e-6) ** 2
#                 else:
#                     w = torch.sigmoid(tw) if sigmoid_wh else torch.clamp(tw, min=1e-6)
#                     h = torch.sigmoid(th) if sigmoid_wh else torch.clamp(th, min=1e-6)

#                 # 2) confidence
#                 conf = torch.sigmoid(tobj) if use_sigmoid_conf else tobj

#                 # 3) 픽셀 좌표
#                 cx = (gx + x) / S * W
#                 cy = (gy + y) / S * H
#                 bw = w * W
#                 bh = h * H

#                 x1 = cx - bw / 2
#                 y1 = cy - bh / 2
#                 x2 = cx + bw / 2
#                 y2 = cy + bh / 2

#                 if clip_boxes:
#                     x1 = torch.clamp(x1, 0, W - 1)
#                     y1 = torch.clamp(y1, 0, H - 1)
#                     x2 = torch.clamp(x2, 0, W - 1)
#                     y2 = torch.clamp(y2, 0, H - 1)

#                 # 4) 점수
#                 score_vec = conf * pcls
#                 score, cls = torch.max(score_vec, dim=0)
#                 s = float(score.item())

#                 # 5) 기본 필터
#                 if s < conf_thr:
#                     continue
#                 ww = float((x2 - x1).item())
#                 hh = float((y2 - y1).item())
#                 if ww < min_wh or hh < min_wh:
#                     continue
#                 if (ww * hh) > (W * H * max_area_ratio):
#                     continue

#                 dets.append([float(x1), float(y1), float(x2), float(y2), s, float(int(cls))])

#     if not dets:
#         return torch.zeros((0, 6), dtype=torch.float32)

#     dets = torch.tensor(dets, dtype=torch.float32)

#     # 6) Pre-NMS Top-K (점수 높은 것만 남기기)
#     if dets.shape[0] > topk_pre_nms:
#         topk = torch.topk(dets[:, 4], k=topk_pre_nms).indices
#         dets = dets[topk]

#     # 7) NMS
#     if per_class_nms:
#         dets = nms_per_class_safe(dets, iou_thr=iou_thr)
#     elif HAS_TV:
#         keep = torchvision.ops.nms(dets[:, :4], dets[:, 4], iou_thr)
#         dets = dets[keep]

#     return dets

######################################################## 버전 4

# import torch

# def decode_yolov1(
#     pred,                     # (S,S,B*5+C), channels-last
#     img_size,                 # (img_w, img_h)  <-- 순서 명시!
#     B=2, C=1,
#     conf_thr=0.10,
#     iou_thr=0.60,
#     per_class_nms=True,
#     *,
#     use_sigmoid_xy=True,      # x,y in [0,1]
#     use_sigmoid_conf=True,    # conf in [0,1]
#     use_sqrt_wh=True,         # training on sqrt(w,h) assumption
#     # ⬇️ 전처리가 letterbox였다면 아래 3개를 전달 (없으면 모두 None/기본값)
#     net_input_size=None,      # (in_w, in_h) 네트워크 입력 크기(예: 448,448)
#     pad=None,                 # (pad_x, pad_y) letterbox에서 추가된 패딩 (픽셀)
#     scale=None,               # 원본->네트 입력으로의 스케일 (float)
#     nms_per_class=None,       # 외부 nms 함수 주입 가능
# ):
#     S = pred.shape[0]
#     img_w, img_h = img_size  # ⬅️ (W,H) 순서로 받는다고 가정
#     box = pred[..., :B*5].view(S, S, B, 5).clone()
#     cls_logits = pred[..., B*5:]  # (S,S,C)

#     # class prob (YOLOv1: per-cell)
#     if C > 1:
#         probs = torch.softmax(cls_logits, dim=-1)  # (S,S,C)
#     else:
#         probs = torch.ones(S, S, 1, dtype=pred.dtype, device=pred.device)

#     dets = []
#     for j in range(S):      # y(row)
#         for i in range(S):  # x(col)
#             pcls = probs[j, i]  # (C,)
#             for b in range(B):
#                 tx, ty, tw, th, tobj = box[j, i, b]

#                 # xy in [0,1] (cell-local)
#                 x = torch.sigmoid(tx) if use_sigmoid_xy else tx
#                 y = torch.sigmoid(ty) if use_sigmoid_xy else ty
#                 x = torch.clamp(x, 0.0, 1.0)
#                 y = torch.clamp(y, 0.0, 1.0)

#                 # wh in [0,1] (image-normalized)  — YOLOv1 assumes sqrt target
#                 if use_sqrt_wh:
#                     w = torch.clamp(tw, min=1e-6) ** 2
#                     h = torch.clamp(th, min=1e-6) ** 2
#                 else:
#                     # 이 경로를 쓸 거면 학습도 w,h 자체를 회귀했어야 함
#                     w = torch.clamp(tw, 0.0, 1.0)
#                     h = torch.clamp(th, 0.0, 1.0)

#                 w = torch.clamp(w, 0.0, 1.0)
#                 h = torch.clamp(h, 0.0, 1.0)

#                 conf = torch.sigmoid(tobj) if use_sigmoid_conf else tobj
#                 if use_sigmoid_conf:
#                     conf = torch.clamp(conf, 0.0, 1.0)

#                 # --- 네트 입력 평면에서의 절대 좌표 (letterbox 전) ---
#                 # grid center
#                 cx_n = (i + x) / S
#                 cy_n = (j + y) / S
#                 # size
#                 bw_n = w
#                 bh_n = h

#                 # 픽셀 (네트 입력 해상도 기준)
#                 if net_input_size is None:
#                     in_w, in_h = img_w, img_h
#                 else:
#                     in_w, in_h = net_input_size

#                 cx_px = cx_n * in_w
#                 cy_px = cy_n * in_h
#                 bw_px = bw_n * in_w
#                 bh_px = bh_n * in_h

#                 x1 = cx_px - bw_px / 2
#                 y1 = cy_px - bh_px / 2
#                 x2 = cx_px + bw_px / 2
#                 y2 = cy_px + bh_px / 2

#                 # --- 레터박스 역변환 (패딩 제거 & 원본 스케일로 복원) ---
#                 if (pad is not None) and (scale is not None):
#                     pad_x, pad_y = pad
#                     # 네트 입력 좌표 -> 패딩 제거
#                     x1 = (x1 - pad_x) / scale
#                     y1 = (y1 - pad_y) / scale
#                     x2 = (x2 - pad_x) / scale
#                     y2 = (y2 - pad_y) / scale
#                 else:
#                     # 전처리가 단순 resize라면 바로 원본 크기라 가정
#                     # 만약 net_input_size != (img_w,img_h) 이면
#                     # 여기서 비율 재조정이 필요할 수 있음
#                     if (in_w != img_w) or (in_h != img_h):
#                         sx = img_w / float(in_w)
#                         sy = img_h / float(in_h)
#                         x1, x2 = x1 * sx, x2 * sx
#                         y1, y2 = y1 * sy, y2 * sy

#                 # 경계 클램프
#                 x1 = float(torch.clamp(x1, 0, img_w - 1))
#                 y1 = float(torch.clamp(y1, 0, img_h - 1))
#                 x2 = float(torch.clamp(x2, 0, img_w - 1))
#                 y2 = float(torch.clamp(y2, 0, img_h - 1))

#                 bw = x2 - x1
#                 bh = y2 - y1

#                 # score = conf * class_prob
#                 score_vec = conf * pcls  # (C,)
#                 score, cls = torch.max(score_vec, dim=0)
#                 s = float(score.item())

#                 if s >= conf_thr and bw > 1 and bh > 1:
#                     dets.append([x1, y1, x2, y2, s, float(int(cls.item()))])

#     if not dets:
#         return torch.zeros((0, 6), dtype=torch.float32)

#     dets = torch.tensor(dets, dtype=torch.float32)

#     # NMS
#     if per_class_nms and nms_per_class is not None:
#         return nms_per_class(dets, iou_thr=iou_thr)
#     elif per_class_nms:
#         # 네가 쓰던 nms_per_class를 가져다 써
#         from .nms import nms_per_class as _nms
#         return _nms(dets, iou_thr=iou_thr)
#     else:
#         return dets