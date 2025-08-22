import os, csv
import matplotlib.pyplot as plt

def _plot_curve(y, title, ylabel, out_path):
    plt.figure()                 # ▶ 한 그림당 하나의 차트
    plt.plot(range(1, len(y)+1), y)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_history_csv(history: dict, out_path: str):
    keys = list(history.keys())
    rows = zip(*[history[k] for k in keys])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow(r)

def plot_training_curves(history: dict, out_dir: str = "runs/train"):
    os.makedirs(out_dir, exist_ok=True)
    _plot_curve(history['loss'],      'Training Loss', 'loss',         os.path.join(out_dir, 'loss_train.png'))
    if any(x==x for x in history['val_loss']):  # NaN 체크
        _plot_curve([v for v in history['val_loss'] if v==v], 'Validation Loss', 'loss', os.path.join(out_dir, 'loss_val.png'))
    _plot_curve(history['coord'],     'Coord Loss (λcoord 포함)', 'loss', os.path.join(out_dir, 'loss_coord.png'))
    _plot_curve(history['obj'],       'Obj Loss',  'loss',             os.path.join(out_dir, 'loss_obj.png'))
    _plot_curve(history['noobj'],     'NoObj Loss','loss',             os.path.join(out_dir, 'loss_noobj.png'))
    _plot_curve(history['cls'],       'Class Loss','loss',             os.path.join(out_dir, 'loss_cls.png'))
    _plot_curve(history['iou_pos_mean'], 'IoU (positive cells mean)', 'IoU', os.path.join(out_dir, 'metric_iou.png'))
    _plot_curve(history['cls_acc'],   'Class Accuracy (cell-level)', 'accuracy', os.path.join(out_dir, 'metric_cls_acc.png'))
    # CSV도 남김
    save_history_csv(history, os.path.join(out_dir, 'history.csv'))