import sys, glob, os, torch
for p in sys.argv[1:]:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    a = ck['args']
    keys = ['fold', 'seed', 'model_mod', 'model_sha', 'ttr_size', 'train_labels', 'sel_labels', 'eval_labels', 'backbone',
            'amp', 'n_slices', 'train_slices', 'aug', 'contract']
    print(os.path.basename(p), 'epoch0', ck.get('epoch'), 'cv', ck.get('cv'), {k: a.get(k) for k in keys})
    print('   decode_probe', sorted(a.get('decode_probe', {})), 'ttr_size_t', ck['model'].get('ttr_size_t'))
