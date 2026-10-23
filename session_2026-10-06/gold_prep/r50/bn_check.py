# bn_check.py: frozen-BN proof on the dry-run checkpoints (BN running stats == timm pretrained; BN affine + conv weights moved),
# EMA vs raw buffers, persistent norm buffers, and the drop_path wiring of the training-time model.
# r50verify fix: HF hub forced offline BEFORE timm/huggingface_hub import (no runner script set it; the pretrained
# reference must come from the local cache only, never a hub metadata request).
import os
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, torch, timm
H = os.path.expanduser('~'); RUN = sys.argv[1]
ref = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0).state_dict()
for tag in ('raw', 'ema'):
    p = [f for f in os.listdir(RUN) if f.startswith(f'f0_s243_{tag}_') and f.endswith('_best.pt')][0]
    sd = torch.load(f'{RUN}/{p}', map_location='cpu', weights_only=False)['model']
    rs = [k for k in ref if k.endswith(('running_mean', 'running_var'))]
    same = all(torch.equal(sd['cnn.' + k], ref[k]) for k in rs)
    nbt = sorted({int(sd['cnn.' + k]) for k in ref if k.endswith('num_batches_tracked')})
    aff = [k for k in ref if k.startswith(('bn1.', 'layer')) and '.bn' in k and k.endswith('weight')] + ['bn1.weight']
    moved_aff = sum(not torch.equal(sd['cnn.' + k], ref[k]) for k in aff)
    convs = [k for k in ref if 'conv' in k and k.endswith('weight')]
    moved_conv = sum(not torch.equal(sd['cnn.' + k], ref[k]) for k in convs)
    extra = sorted(k for k in sd if not k.startswith('cnn.'))
    print(f'{tag}: BN running stats == pretrained: {same} ({len(rs)} tensors) | num_batches_tracked {nbt} | BN affine moved '
          f'{moved_aff}/{len(aff)} | conv weights moved {moved_conv}/{len(convs)} | norm m {sd["m"].flatten().tolist()} '
          f's {[round(v, 3) for v in sd["s"].flatten().tolist()]} | head keys {extra}')
sys.path.insert(0, f'{H}/fast_build/code2')
from fleet_model_f2 import KneeFleetNetF2
m = KneeFleetNetF2(backbone='resnet50.a1_in1k', pretrained=False, drop_path=0.1, dropout=0.3, grad_ckpt=True, freeze_bn=True)
m.train()
dp = [n for n, q in m.cnn.named_modules() if type(q).__name__ == 'DropPath']
bn_train = sum(q.training for q in m.cnn.modules() if isinstance(q, torch.nn.modules.batchnorm._BatchNorm))
print(f'train-mode model: DropPath modules {len(dp)} (e.g. {dp[:2]}) | BN modules in training mode {bn_train} | '
      f'grad_checkpointing {getattr(m.cnn, "grad_checkpointing", None)}')
