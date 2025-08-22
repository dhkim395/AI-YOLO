import torch

def box_iou_xyxy(a, b, eps=1e-12):
    x1 = torch.max(a[:,None,0], b[None,:,0])
    y1 = torch.max(a[:,None,1], b[None,:,1])
    x2 = torch.min(a[:,None,2], b[None,:,2])
    y2 = torch.min(a[:,None,3], b[None,:,3])
    iw = (x2-x1).clamp(min=0); ih=(y2-y1).clamp(min=0)
    inter = iw*ih
    area_a = ((a[:,2]-a[:,0]).clamp(0)*(a[:,3]-a[:,1]).clamp(0))[:,None]
    area_b = ((b[:,2]-b[:,0]).clamp(0)*(b[:,3]-b[:,1]).clamp(0))[None,:]
    union = area_a + area_b - inter + eps
    return inter/union

def nms_per_class(dets, iou_thr=0.45):
    """
    dets: (N,6) [x1,y1,x2,y2,score,cls]
    """
    if dets.numel()==0: return dets
    out = []
    for c in dets[:,5].unique():
        dc = dets[dets[:,5]==c]
        order = dc[:,4].argsort(descending=True)
        dc = dc[order]
        keep=[]
        while dc.size(0):
            keep.append(dc[0:1])
            if dc.size(0)==1: break
            ious = box_iou_xyxy(dc[0:1,:4], dc[1:,:4]).squeeze(0)
            dc = dc[1:][ious<=iou_thr]
        out.append(torch.cat(keep, dim=0))
    return torch.cat(out, dim=0) if out else dets.new_zeros((0,6))