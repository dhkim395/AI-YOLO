import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

def build_optimizer(model, kind='sgd', lr=1e-2, momentum=0.9, weight_decay=5e-4, betas=(0.9,0.999)):
    if kind.lower()=='sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif kind.lower()=='adam':
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        raise ValueError(kind)

def build_scheduler(optimizer, kind='cosine', epochs=100, warmup_epochs=3, base_lr=1e-2, min_lr_ratio=0.02, milestones=(60,90), gamma=0.1):
    if kind=='cosine':
        sched_warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=max(1,warmup_epochs))
        sched_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,epochs-warmup_epochs), eta_min=base_lr*min_lr_ratio)
        return [sched_warm, sched_main]
    elif kind=='multistep':
        return [torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(milestones), gamma=gamma)]
    return []

def train_yolov1(model, criterion, train_loader, val_loader=None, epochs=100,
                 optimizer_kind='sgd', base_lr=1e-2, weight_decay=5e-4, momentum=0.9, betas=(0.9,0.999),
                 scheduler_kind='cosine', warmup_epochs=3, amp=True, grad_clip=10.0, device='cuda'):
    model.to(device); criterion.to(device)
    optim = build_optimizer(model, optimizer_kind, base_lr, momentum, weight_decay, betas)
    scheds = build_scheduler(optim, scheduler_kind, epochs, warmup_epochs, base_lr)
    device_type = 'cuda' if (device.startswith('cuda') and torch.cuda.is_available()) else 'cpu'
    scaler = GradScaler(device=device_type, enabled=(amp and device_type=='cuda'))

    history = {  # ▶ 에폭별 기록
        'epoch': [], 'loss': [], 'coord': [], 'obj': [], 'noobj': [], 'cls': [], 'iou_pos_mean': [], 'cls_acc': [],
        'val_loss': []
    }

    def step_sched(epoch):
        if scheduler_kind=='cosine' and len(scheds)==2:
            (scheds[0] if epoch < warmup_epochs else scheds[1]).step()
        elif len(scheds)==1: scheds[0].step()

    for epoch in range(epochs):
        model.train()
        tot=0.0; aggr={'coord':0,'obj':0,'noobj':0,'cls':0,'iou_pos_mean':0,'cls_acc':0}; n=0

        for imgs, targets in train_loader:
            imgs=imgs.to(device, non_blocking=True); targets=targets.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast(device_type=device_type, enabled=(amp and device_type=='cuda')):
                preds = model(imgs)
                loss, ldict = criterion(preds, targets)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip>0:
                scaler.unscale_(optim); nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim); scaler.update()

            tot += loss.item()
            for k in aggr: aggr[k] += ldict[k].item()
            n += 1

        step_sched(epoch)
        avg = tot/max(1,n); logs = {k: aggr[k]/max(1,n) for k in aggr}

        # 검증 손실
        vloss = None
        if val_loader and (epoch+1)%1==0:
            model.eval(); vtot=0.0; vn=0
            with torch.no_grad(), autocast(device_type=device_type, enabled=(amp and device_type=='cuda')):
                for vi, vt in val_loader:
                    vi=vi.to(device); vt=vt.to(device)
                    l,_ = criterion(model(vi), vt); vtot+=l.item(); vn+=1
            vloss = vtot/max(1,vn)

        # 로그 출력
        if vloss is None:
            print(f"[{epoch+1:03d}/{epochs}] loss={avg:.4f} | coord={logs['coord']:.4f} obj={logs['obj']:.4f} noobj={logs['noobj']:.4f} cls={logs['cls']:.4f} | IoU={logs['iou_pos_mean']:.4f} acc={logs['cls_acc']:.4f}")
        else:
            print(f"[{epoch+1:03d}/{epochs}] loss={avg:.4f} (val={vloss:.4f}) | coord={logs['coord']:.4f} obj={logs['obj']:.4f} noobj={logs['noobj']:.4f} cls={logs['cls']:.4f} | IoU={logs['iou_pos_mean']:.4f} acc={logs['cls_acc']:.4f}")

        # ▶ 히스토리 저장
        history['epoch'].append(epoch+1)
        history['loss'].append(avg)
        history['coord'].append(logs['coord'])
        history['obj'].append(logs['obj'])
        history['noobj'].append(logs['noobj'])
        history['cls'].append(logs['cls'])
        history['iou_pos_mean'].append(logs['iou_pos_mean'])
        history['cls_acc'].append(logs['cls_acc'])
        history['val_loss'].append(vloss if vloss is not None else float('nan'))

    return history  # ▶ 학습 내역 반환