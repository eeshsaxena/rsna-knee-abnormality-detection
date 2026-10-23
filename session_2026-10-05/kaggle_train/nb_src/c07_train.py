# [step 6] choose E from the probe, launch fold 3 (GPU0) and fold 4 (GPU1) for real, watch them, enforce the hard
# wall, then write the receipt. Outputs go straight to /kaggle/working/f{F}_s1043_ttr_mix50/ (+ .log next to it):
# the trainer rewrites its best raw/ema ckpt + OOF csv whenever val improves, so a kill keeps best-so-far.
_el0 = el()
_iters = max(COUNTS[F]['iters_per_epoch'] for F in FOLDS)
_ec = choose_epochs(S_IT, _iters, _el0)
E = _ec['E']
SAVE_EP = save_epochs_arg(E)
PLAN = dict(E=E, save_epochs=SAVE_EP, s_it=S_IT, iters_per_epoch=_iters, eval_frac=CFG['EVAL_FRAC'],
            epoch_min_est=round(_ec['epoch_s'] / 60, 2), elapsed_at_launch_h=round(_el0 / 3600, 3),
            avail_h=round(_ec['avail_s'] / 3600, 3), budget_h=CFG['BUDGET_H'], wall_h=CFG['WALL_H'],
            est_finish_h=round(_ec['finish_elapsed_s'] / 3600, 2), est_finish_ist=ist(T0 + _ec['finish_elapsed_s']),
            wall_ist=ist(T0 + CFG['WALL_H'] * 3600), box_recipe_epochs=_a['epochs'])
REC['plan'] = PLAN
log(f'E = {E} epochs (min({CFG["EPOCHS_MAX"]}, floor({_ec["avail_s"] / 3600:.2f} h / {_ec["epoch_s"] / 60:.1f} min))), '
    f'--save_epochs {SAVE_EP}; est. finish {PLAN["est_finish_ist"]} (+{PLAN["est_finish_h"]} h), wall {PLAN["wall_ist"]}')
if E < CFG['EPOCHS_WARN']:
    warn(f'E = {E} < {CFG["EPOCHS_WARN"]}: the cosine schedule is {E}/24 of the box recipe; the operator decides '
         f'whether these fold-3/4 members are used')
save_receipt()

if not wait_gpus_free(120):
    warn(f'GPU memory in use before the real launch: {gpu_query()}')
for F in FOLDS:
    if os.path.exists(f'{WORK}/{RUN[F]}'):
        shutil.rmtree(f'{WORK}/{RUN[F]}')
PROCS = {F: launch(F, g, E, SAVE_EP, f'{WORK}/{RUN[F]}', f'{WORK}/{RUN[F]}.log', LBL, CFG['WORKERS'])
         for F, g in zip(FOLDS, CFG['GPU_IDS'])}
T_LAUNCH = time.time()
REC['train'] = dict(launch_ist=ist(T_LAUNCH), launch_h=round(el() / 3600, 3),
                    cmd={F: ' '.join(PROCS[F].kt_cmd) for F in FOLDS}, folds={})
REC['status'] = 'training'
save_receipt()
ENDED, _hdr_ok, _proj_warned, _last_status = {}, {}, False, 0.0
WALL = T0 + CFG['WALL_H'] * 3600


def _snapshot():
    out = {}
    for F in FOLDS:
        txt = read_text(f'{WORK}/{RUN[F]}.log')
        pr = progress(txt)
        its = parse_its(txt, max(0, pr['last_ep_done'] + 1))
        ep_min = [m for ep, tag, _, m in pr['evals'] if tag == 'ema']
        out[F] = dict(epochs_done=pr['last_ep_done'] + 1, best=pr['best'], epoch_min=ep_min,
                      train_only_min=pr['train_only_min'], cur_it=its[-1][0] if its else None,
                      alive=PROCS[F].poll() is None, rc=PROCS[F].returncode)
    return out


def finalize():
    import torch
    snap = _snapshot()
    total, folds = 0, {}
    exp_keys = [k for k, _, _ in BOX['state_dict']]
    for F in FOLDS:
        d, lp = f'{WORK}/{RUN[F]}', f'{WORK}/{RUN[F]}.log'
        txt = read_text(lp)
        files = []
        for root, _, fs in os.walk(d):
            for f in sorted(fs):
                p = os.path.join(root, f)
                rel = os.path.relpath(p, WORK).replace(os.sep, '/')
                e = dict(path=rel, bytes=os.path.getsize(p), sha256=sha256_file(p))
                if f.endswith('.pt'):
                    try:
                        ck = torch.load(p, map_location='cpu', weights_only=False)
                        e.update(load_ok=True, keys_match=list(ck['model'].keys()) == exp_keys, cv=ck.get('cv'),
                                 epoch=ck.get('epoch'), args_epochs=ck['args'].get('epochs'),
                                 args_fold=ck['args'].get('fold'), args_seed=ck['args'].get('seed'))
                        del ck
                    except Exception as ex:
                        e.update(load_ok=False, error=f'{type(ex).__name__}: {str(ex)[:200]}')
                        warn(f'{rel} does not load: {e["error"]}')
                files.append(e)
                total += e['bytes']
        cl = closing_line(txt)
        best_pt = [x['path'] for x in files if x['path'].endswith('_best.pt') and '/epochs/' not in x['path']]
        folds[F] = dict(run=RUN[F], out_dir=d, log=lp, rc=PROCS[F].returncode, ended=ENDED.get(F),
                        closing_line=cl, complete=cl is not None, best_ckpts=best_pt,
                        oof=[x['path'] for x in files if os.path.basename(x['path']).startswith('oof_')
                             and '/epochs/' not in x['path']],
                        progress=snap[F], files=files, log_tail=tail(lp, 6))
        if os.path.exists(lp):
            total += os.path.getsize(lp)
    if total > CFG['OUT_LIMIT_GB'] * 1e9:                 # never expected (~1.4 GB); keep best ckpts + all csv
        for F in FOLDS:
            for p in glob.glob(f'{WORK}/{RUN[F]}/epochs/*.pt'):
                os.remove(p)
        warn(f'outputs {total / 1e9:.1f} GB > {CFG["OUT_LIMIT_GB"]} GB: removed epochs/*.pt')
    REC['train']['folds'] = folds
    REC['train']['end_ist'] = ist()
    REC['train']['hours'] = round((time.time() - T_LAUNCH) / 3600, 3)
    REC['outputs_gb'] = round(total / 1e9, 3)
    if not str(REC['status']).startswith('ABORTED'):
        REC['status'] = ('DONE' if all(folds[F]['complete'] for F in FOLDS) else
                         'PARTIAL (no closing line for folds ' +
                         ', '.join(str(F) for F in FOLDS if not folds[F]['complete']) + ')')
    save_receipt()
    for F in FOLDS:
        x = folds[F]
        log(f'fold {F}: {x["ended"]} rc {x["rc"]} | closing {x["closing_line"]} | epochs done '
            f'{x["progress"]["epochs_done"]}/{E} | best {x["progress"]["best"]} | {len(x["files"])} files')
    log(f'STATUS {REC["status"]} | outputs {REC["outputs_gb"]} GB | receipt {WORK}/run_receipt.json')


try:
    while True:
        now = time.time()
        for F in FOLDS:
            if F not in _hdr_ok:
                txt = read_text(f'{WORK}/{RUN[F]}.log')
                if RE_IT.search(txt):
                    _pb = header_problems(parse_header(txt), COUNTS[F], BOX)
                    if _pb:
                        for G in FOLDS:
                            ENDED[G] = stop_group(PROCS[G], 10)
                        abort(f'fold {F}: trainer header differs from the box: {_pb}')
                    _hdr_ok[F] = True
        snap = _snapshot()
        REC['train']['progress'] = snap
        if not any(s['alive'] for s in snap.values()):
            for F in FOLDS:
                ENDED[F] = stop_group(PROCS[F], 5)
            break
        if now >= WALL:
            warn(f'hard wall {CFG["WALL_H"]} h reached: terminating the trainers (best-so-far ckpts are kept)')
            _wq = wait_quiet([f'{WORK}/{RUN[F]}' for F in FOLDS])
            log(f'outputs quiet after {_wq:.0f} s (no ckpt/OOF write in flight): sending SIGTERM')
            REC['train']['wall_quiet_wait_s'] = round(_wq, 1)
            for F in FOLDS:
                ENDED[F] = stop_group(PROCS[F], CFG['KILL_GRACE_S'])
            break
        _done = [s['epochs_done'] for s in snap.values() if s['epochs_done'] > 0]
        if _done and not _proj_warned:
            _per = max(sum(s['epoch_min']) / len(s['epoch_min']) for s in snap.values() if s['epoch_min']) * 60
            _fin = T_LAUNCH + CFG['STARTUP_S'] + E * _per
            REC['train']['projected_finish_ist'] = ist(_fin)
            if _fin > WALL:
                _proj_warned = True
                warn(f'measured {_per / 60:.1f} min/epoch -> projected finish {ist(_fin)} is past the wall '
                     f'{ist(WALL)}: the last epochs will be cut (best-so-far is kept)')
        if now - _last_status >= CFG['STATUS_EVERY_S']:
            _last_status = now
            log(' | '.join(f'f{F}: ep {s["epochs_done"]}/{E} it {s["cur_it"]} best ' +
                           ','.join(f'{t} {v[0]:.4f}@{v[1]}' for t, v in sorted(s['best'].items()))
                           for F, s in snap.items()) +
                f' | proj. finish {REC["train"].get("projected_finish_ist", "?")}')
        save_receipt()
        time.sleep(CFG['POLL_S'])
finally:
    for F in FOLDS:
        if F not in ENDED:
            ENDED[F] = stop_group(PROCS[F], CFG['KILL_GRACE_S'])
    finalize()
