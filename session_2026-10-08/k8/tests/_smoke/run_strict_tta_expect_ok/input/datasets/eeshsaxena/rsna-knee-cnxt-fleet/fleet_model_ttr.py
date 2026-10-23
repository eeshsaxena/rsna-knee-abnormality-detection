# fleet_model_ttr.py -- TTR family (BIG_JUMP_PLAN sec 3.1, build item B1): the family-1 KneeFleetNet with ONE change in
# encode(): every uint8 slice image is resized to ttr_size x ttr_size (default 224) with bilinear + antialias AFTER the
# uint8 -> float [0, 1] step and BEFORE the ImageNet normalisation. Everything else (backbone, slice/view attention
# pooling, heads, mirror_views) is inherited unchanged. fleet_model.py itself is NOT edited (the worker asserts
# ckpt['args']['model_sha'] == sha(fleet_model.py source) for every family-1 member; fa35880fca963652).
#   TTR-1 the store stays 320 px (contract r3): train and serve feed the same 320 px uint8 triplets, the model resizes
#         inside, so the worker, the pack tool and fleet_common need no change and train == serve by construction
#   TTR-2 the resize runs in fp32 in both paths (x is float32 here; F.interpolate is not an autocast fp16 op)
#   TTR-3 drop_path / head dropout are constructor args as in KneeFleetNet (recipe: 0.2 / 0.3); they do nothing in
#         eval mode, so the worker builds the net with drop_path 0.0 like every other family
#   TTR-4 model_sha = sha(fleet_model_ttr.py source + fleet_model.py source), recomputed by the trainer, the worker and
#         the pack tool (ckpt args model_mod == 'fleet_model_ttr', ttr_size)
#   TTR-5 a store image that is already ttr_size x ttr_size is passed through untouched (no identity interpolate)
#   TTR-6 persistent int64 buffer ttr_size_t: a TTR state_dict no longer strict-loads into a plain KneeFleetNet
#         (unexpected key), so a loader that skips the model_mod / model_sha check cannot serve a TTR ckpt at 320 px.
#         The trainer's EMA (AveragedModel, use_buffers False) never averages it; the worker asserts it == args ttr_size
import torch
import torch.nn.functional as F
from fleet_model import KneeFleetNet, mirror_views  # noqa: F401  (mirror_views re-exported)


class KneeFleetNetTTR(KneeFleetNet):
    def __init__(self, ttr_size=224, **kw):
        super().__init__(**kw)
        self.ttr_size = int(ttr_size)
        self.register_buffer('ttr_size_t', torch.tensor(self.ttr_size, dtype=torch.int64), persistent=True)   # TTR-6

    def encode(self, x_u8):                                                 # TTR-1/2/5, else = KneeFleetNet.encode
        outs = []
        S = self.ttr_size
        for i in range(0, x_u8.shape[0], self.chunk):
            x = x_u8[i:i + self.chunk].float().div_(255.0)
            if x.shape[-2] != S or x.shape[-1] != S:
                x = F.interpolate(x, size=(S, S), mode='bilinear', align_corners=False, antialias=True)
            x = ((x - self.m) / self.s).contiguous(memory_format=torch.channels_last)
            outs.append(self.cnn(x))
        return torch.cat(outs)
