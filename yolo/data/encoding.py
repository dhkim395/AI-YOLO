import torch

def corners_to_center_norm(labels, img_size, normalized=False, eps=1e-6):
    """
    labels: (M,5) [class_id, x1, y1, x2, y2]
    img_size: (W,H)
    return: (K,5) [class_id, x, y, w, h], all in [0,1]
    """
    if labels.numel() == 0:
        return labels.new_zeros((0,5))
    W, H = img_size
    lab = labels.clone().float()
    if not normalized:
        lab[:,1] /= W; lab[:,2] /= H; lab[:,3] /= W; lab[:,4] /= H
    x1 = torch.minimum(lab[:,1], lab[:,3])
    y1 = torch.minimum(lab[:,2], lab[:,4])
    x2 = torch.maximum(lab[:,1], lab[:,3])
    y2 = torch.maximum(lab[:,2], lab[:,4])
    w = (x2-x1).clamp(min=eps); h=(y2-y1).clamp(min=eps)
    x = (x1+x2)*0.5; y=(y1+y2)*0.5
    x = x.clamp(eps, 1-eps); y = y.clamp(eps, 1-eps)
    w = w.clamp(eps, 1-eps); h = h.clamp(eps, 1-eps)
    out = torch.stack([lab[:,0], x, y, w, h], dim=1)
    keep = (w>0) & (h>0)
    return out[keep]

def encode_targets_yolov1(boxes, S=7, C=20):
    """
    boxes: (M,5) [class_id, x, y, w, h] in [0,1]
    return: (S,S,5+C) with [tx,ty,w,h,obj, one-hot(C)]
    """
    target = torch.zeros((S,S,5+C), dtype=torch.float32)
    if boxes.numel()==0: return target
    chosen_area = torch.zeros((S,S), dtype=torch.float32)
    for b in boxes:
        cid, x, y, w, h = b.tolist()
        i = min(S-1, int(x*S)); j = min(S-1, int(y*S))
        area = w*h
        if area > chosen_area[j,i]:
            tx = x*S - i; ty = y*S - j
            target[j,i,0:5] = torch.tensor([tx,ty,w,h,1.0])
            target[j,i,5+int(cid)] = 1.0
            chosen_area[j,i] = area
    return target