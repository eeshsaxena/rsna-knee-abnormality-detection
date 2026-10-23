# fleet_model_f2.py -- fleet FAMILY 2 (BatchNorm CNN backbones: tf_efficientnetv2_s.in21k_ft_in1k, resnet50.a1_in1k)
# as a subclass of fleet_model.KneeFleetNet. fleet_model.py itself is NOT edited: fleet_worker.py asserts
# ckpt['args']['model_sha'] == sha(fleet_model.py source), so any edit there would make every family-1 (ConvNeXt-T)
# checkpoint unloadable. forward(), the slice/view attention pooling, the heads and mirror_views are inherited
# unchanged; only the backbone input path differs:
#   F2-1 input norm = the timm pretrained_cfg of the tag (tf_efficientnetv2_s.in21k* = mean/std 0.5,
#        resnet50.a1_in1k = ImageNet), registered as PERSISTENT buffers 'm'/'s' so the norm travels inside the
#        state_dict (KneeFleetNet hard-codes ImageNet in non-persistent buffers). The worker builds the net with
#        pretrained=False and its strict load restores the trained norm (incl. RadImageNet 0.5/0.5).
#   F2-2 freeze_bn: every BatchNorm stays in eval mode under train() (running stats frozen, affine still trained).
#        Why: encode() runs chunks of 32 images that come from ONE study (24 slices of view j + 8 of view j+1) plus
#        all-zero images for absent views, so train-mode BN would normalise with single-study, zero-polluted batch
#        statistics (absent views would leak into present ones); grad-checkpoint recompute updates running stats
#        twice per step; and fleet_train's EMA (AveragedModel, use_buffers=False) never copies BN buffers, so the
#        EMA checkpoint would carry stale running stats under adapted weights.
#   F2-3 init_ckpt: optional backbone init from a local state_dict. The RadImageNet R50 file is a torchvision
#        children()[:-2] Sequential ('0.' conv1, '1.' bn1, '4.'..'7.' layer1..4): remapped to timm resnet50 keys,
#        loaded strictly; its input range is [-1, 1] (our crop code: x/127.5-1), i.e. mean = std = 0.5.
#   F2-4 channels_last optional (default on, as in family 1).
# With a BN-free backbone (ConvNeXt), ImageNet norm and channels_last on, KneeFleetNetF2 computes exactly what
# KneeFleetNet computes (fleet2/tests/t1_default_parity.py: max |diff| 0 on logits and on every gradient).
import torch
import torch.nn as nn
from fleet_model import KneeFleetNet, mirror_views  # noqa: F401  (mirror_views re-exported)

_TV_SEQ = {'0.': 'conv1.', '1.': 'bn1.', '4.': 'layer1.', '5.': 'layer2.', '6.': 'layer3.', '7.': 'layer4.'}


def remap_tv_sequential_resnet(sd):
    """torchvision resnet50 children()[:-2] Sequential keys -> timm resnet50 keys (identical tensor shapes).
    The real ~/rsna_train/ResNet50_radimagenet.pt (sha256 08629f7e..., 318 tensors) stores them as 'backbone.0.weight',
    'backbone.4.0.conv1.weight', ... (the crop code's Enc.backbone): that prefix is stripped first."""
    if sd and all(k.startswith('backbone.') for k in sd):
        sd = {k[len('backbone.'):]: v for k, v in sd.items()}
    out = {}
    for k, v in sd.items():
        for a, b in _TV_SEQ.items():
            if k.startswith(a):
                out[b + k[len(a):]] = v
                break
        else:
            out[k] = v
    return out


def has_bn(module):
    return any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in module.modules())


class KneeFleetNetF2(KneeFleetNet):
    def __init__(self, backbone='tf_efficientnetv2_s.in21k_ft_in1k', freeze_bn=True, init_ckpt='',
                 channels_last=True, pretrained=True, **kw):
        super().__init__(backbone=backbone, pretrained=(pretrained and not init_ckpt), **kw)
        cfg = self.cnn.pretrained_cfg
        mean = tuple(float(v) for v in cfg.get('mean', (0.485, 0.456, 0.406)))
        std = tuple(float(v) for v in cfg.get('std', (0.229, 0.224, 0.225)))
        if init_ckpt:                                                       # F2-3
            sd = torch.load(init_ckpt, map_location='cpu', weights_only=True)
            sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
            sd = remap_tv_sequential_resnet(sd) if any(k.startswith(('0.', 'backbone.0.')) for k in sd) else sd
            miss, unexp = self.cnn.load_state_dict(sd, strict=False)
            assert not unexp and not [k for k in miss if not k.startswith('fc.')], (miss[:4], unexp[:4])
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        self.register_buffer('m', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=True)   # F2-1
        self.register_buffer('s', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=True)
        self.freeze_bn, self.channels_last = bool(freeze_bn), bool(channels_last)
        self.norm_mean, self.norm_std = mean, std

    def train(self, mode=True):                                             # F2-2
        super().train(mode)
        if mode and self.freeze_bn:
            for mod in self.cnn.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        return self

    def encode(self, x_u8):                                                 # F2-4, else = KneeFleetNet.encode
        outs = []
        fmt = torch.channels_last if self.channels_last else torch.contiguous_format
        for i in range(0, x_u8.shape[0], self.chunk):
            x = x_u8[i:i + self.chunk].float().div_(255.0)
            x = ((x - self.m) / self.s).contiguous(memory_format=fmt)
            outs.append(self.cnn(x))
        return torch.cat(outs)
