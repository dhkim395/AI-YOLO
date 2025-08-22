import torch
import torch.nn as nn
import torch.nn.functional as F

class YoloV1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5, eps=1e-9):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.l_coord, self.l_noobj, self.eps = lambda_coord, lambda_noobj, eps

    @staticmethod
    def _xywh_to_xyxy(cx, cy, w, h):
        x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
        return x1,y1,x2,y2

    def _iou(self, a_xywh, b_xywh):
        x1a,y1a,x2a,y2a = self._xywh_to_xyxy(a_xywh[...,0], a_xywh[...,1], a_xywh[...,2], a_xywh[...,3])
        x1b,y1b,x2b,y2b = self._xywh_to_xyxy(b_xywh[...,0], b_xywh[...,1], b_xywh[...,2], b_xywh[...,3])
        ix1 = torch.maximum(x1a, x1b); iy1 = torch.maximum(y1a, y1b)
        ix2 = torch.minimum(x2a, x2b); iy2 = torch.minimum(y2a, y2b)
        iw = (ix2-ix1).clamp(min=0); ih=(iy2-iy1).clamp(min=0)
        inter = iw*ih
        area_a = (x2a-x1a).clamp(0)*(y2a-y1a).clamp(0)
        area_b = (x2b-x1b).clamp(0)*(y2b-y1b).clamp(0)
        union = area_a + area_b - inter + 1e-12
        return inter/union

    def forward(self, pred, target):
        """
        pred: (N,S,S,B*5+C) channels-last
        target: (N,S,S,5+C)
        """
        N,S,S2,D = pred.shape
        assert S==self.S and S2==self.S and D==self.B*5+self.C

        pred_boxes = pred[...,:self.B*5].view(N,S,S,self.B,5)
        pred_cls   = pred[..., self.B*5:]

        px,py = pred_boxes[...,0], pred_boxes[...,1]
        pw,ph = torch.clamp(pred_boxes[...,2],min=self.eps), torch.clamp(pred_boxes[...,3],min=self.eps)
        pconf  = pred_boxes[...,4]

        tx,ty = target[...,0], target[...,1]
        tw,th = torch.clamp(target[...,2],min=self.eps), torch.clamp(target[...,3],min=self.eps)
        tobj  = target[...,4].bool()
        tcls  = target[...,5:]

        # grid indices
        gy,gx = torch.meshgrid(torch.arange(S, device=pred.device), torch.arange(S, device=pred.device), indexing='ij')
        gx = gx.view(1,S,S,1).float(); gy = gy.view(1,S,S,1).float()

        abs_px = (gx+px)/S; abs_py = (gy+py)/S; abs_pw=pw; abs_ph=ph
        abs_tx = (gx.squeeze(-1)+tx)/S; abs_ty=(gy.squeeze(-1)+ty)/S; abs_tw=tw; abs_th=th

        ious = self._iou(torch.stack([abs_px,abs_py,abs_pw,abs_ph],dim=-1),
                         torch.stack([abs_tx,abs_ty,abs_tw,abs_th],dim=-1).unsqueeze(3))  # (N,S,S,B)
        iou_best, best_idx = ious.max(dim=3)
        resp_mask = F.one_hot(best_idx, num_classes=self.B).bool() & tobj.unsqueeze(-1)

        # coordinate loss
        L_xy = F.mse_loss(px[resp_mask], tx[tobj], reduction='sum') \
             + F.mse_loss(py[resp_mask], ty[tobj], reduction='sum')
        L_wh = F.mse_loss(torch.sqrt(torch.clamp(pw[resp_mask], min=self.eps)), torch.sqrt(tw[tobj]), reduction='sum') \
             + F.mse_loss(torch.sqrt(torch.clamp(ph[resp_mask], min=self.eps)), torch.sqrt(th[tobj]), reduction='sum')
        L_coord = L_xy + L_wh

        # objectness
        L_obj   = F.mse_loss(pconf[resp_mask], iou_best[tobj], reduction='sum')
        noobj_mask_all = (~resp_mask & tobj.unsqueeze(-1)) | (~tobj).unsqueeze(-1)
        L_noobj = F.mse_loss(pconf[noobj_mask_all], torch.zeros_like(pconf[noobj_mask_all]), reduction='sum')

        # class (only obj cells)
        if tobj.any():
            prob = torch.softmax(pred_cls[tobj], dim=-1)
            L_cls = F.mse_loss(prob, tcls[tobj], reduction='sum')
            # ▶ 추가: 셀 단위 클래스 정확도(참고용 proxy metric)
            pred_lbl = prob.argmax(dim=-1)
            true_lbl = tcls[tobj].argmax(dim=-1)
            cls_acc = (pred_lbl == true_lbl).float().mean()
        else:
            L_cls = torch.tensor(0.0, device=pred.device)
            cls_acc = torch.tensor(0.0, device=pred.device)

        loss = (self.l_coord*L_coord + L_obj + self.l_noobj*L_noobj + L_cls)/max(N,1)

        logs = {
            'coord': (self.l_coord*L_coord/max(N,1)).detach(),
            'obj':   (L_obj/max(N,1)).detach(),
            'noobj': (self.l_noobj*L_noobj/max(N,1)).detach(),
            'cls':   (L_cls/max(N,1)).detach(),
            'iou_pos_mean': (iou_best[tobj].mean() if tobj.any() else torch.tensor(0.0, device=pred.device)).detach(),
            'cls_acc': cls_acc.detach(),   # ▶ 추가
        }
        return loss, logs