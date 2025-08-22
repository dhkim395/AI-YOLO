import torch
import torch.nn as nn

class ConvBNLeaky(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, negative_slope=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class FastBackbone9(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvBNLeaky(3, 16), nn.MaxPool2d(2,2),      # 448->224
            ConvBNLeaky(16, 32), nn.MaxPool2d(2,2),     # 224->112
            ConvBNLeaky(32, 64), nn.MaxPool2d(2,2),     # 112->56
            ConvBNLeaky(64,128), nn.MaxPool2d(2,2),     # 56->28
            ConvBNLeaky(128,256), nn.MaxPool2d(2,2),    # 28->14
            ConvBNLeaky(256,512), nn.MaxPool2d(2,2),    # 14->7
            ConvBNLeaky(512,1024),
            ConvBNLeaky(1024,1024),
            ConvBNLeaky(1024,1024),
        ])
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.1)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # (N,1024,H/64,W/64)

class YOLOv1FCNHead(nn.Module):
    def __init__(self, in_ch=1024, num_classes=20, B=2, mid_ch=1024):
        super().__init__()
        self.C, self.B = num_classes, B
        self.refine = nn.Sequential(
            ConvBNLeaky(in_ch, mid_ch),
            ConvBNLeaky(mid_ch, mid_ch),
        )
        self.head_box = nn.Conv2d(mid_ch, B*5, kernel_size=1)
        self.head_cls = nn.Conv2d(mid_ch, num_classes, kernel_size=1)
        nn.init.normal_(self.head_box.weight, 0, 0.01); nn.init.zeros_(self.head_box.bias)
        nn.init.normal_(self.head_cls.weight, 0, 0.01); nn.init.zeros_(self.head_cls.bias)

    def forward(self, x):
        f = self.refine(x)
        box = self.head_box(f)  # (N,B*5,S,S)
        cls = self.head_cls(f)  # (N,C,S,S)
        out = torch.cat([box, cls], dim=1)
        return out

class FastYOLOv1_FCN(nn.Module):
    def __init__(self, num_classes=20, B=2):
        super().__init__()
        self.backbone = FastBackbone9()
        self.head     = YOLOv1FCNHead(in_ch=1024, num_classes=num_classes, B=B)
        self.C, self.B = num_classes, B

    def forward(self, x, channels_last=True):
        f = self.backbone(x)             # (N,1024,S,S)
        y = self.head(f)                 # (N,B*5+C,S,S)
        if channels_last:
            y = y.permute(0,2,3,1).contiguous()  # (N,S,S,B*5+C)
        return y