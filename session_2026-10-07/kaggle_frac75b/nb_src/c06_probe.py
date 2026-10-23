# [step 5] timing probe: BOTH trainers at once with the real recipe (NB_PLAN trainers[0] on GPU0, trainers[1] on GPU1) until each has
# printed 'it PROBE_ITERS' (or timeout), then killed. Steady s/it = window between the last two progress lines.
# The probe logs also prove, before any real training, that the trainer itself prints the box contract and the
# expected fold counts (abort otherwise).
if not wait_gpus_free(60):
    abort(f'GPUs not free before the probe: {gpu_query()}')
shutil.rmtree(PROBE_DIR, ignore_errors=True)
os.makedirs(PROBE_DIR)
_W = CFG['WORKERS']
_pp = {F: launch(F, g, CFG['EPOCHS_MAX'], save_epochs_arg(CFG['EPOCHS_MAX']), f'{PROBE_DIR}/out_f{F}',
                 f'{PROBE_DIR}/f{F}.log', LBL, _W) for F, g in zip(FOLDS, CFG['GPU_IDS'])}
_t_start = time.time()
_first, _cpu_a, _cpu_b, _gutil, _t_last_g, _hdr_ok = {}, None, None, {}, 0.0, {}
_why = 'timeout'
while True:
    time.sleep(CFG['PROBE_POLL_S'])
    _txt = {F: read_text(f'{PROBE_DIR}/f{F}.log') for F in FOLDS}
    _its = {F: parse_its(_txt[F]) for F in FOLDS}
    for F in FOLDS:
        if F not in _hdr_ok and RE_IT.search(_txt[F]):          # header is complete once training prints
            _pb = header_problems(parse_header(_txt[F]), COUNTS[F], BOX)
            if _pb:
                for G in FOLDS:
                    stop_group(_pp[G], min(10, CFG['KILL_GRACE_S']))
                REC['probe'] = dict(header_problems={F: _pb})
                abort(f'probe fold {F}: trainer header differs from the box: {_pb}')
            _hdr_ok[F] = True
        if _its[F] and F not in _first:
            _first[F] = time.time()
    if _cpu_a is None and len(_first) == len(FOLDS):
        _cpu_a, _t_cpu_a = cpu_times(), time.time()
    if _cpu_a is not None and time.time() - _t_last_g >= 10:
        for g in gpu_query():
            _gutil.setdefault(g['index'], []).append(float(g['util_pct']))
        _t_last_g = time.time()
    if all(_its[F] and _its[F][-1][0] >= CFG['PROBE_ITERS'] for F in FOLDS):
        _why = 'reached'
        break
    _dead = [F for F in FOLDS if _pp[F].poll() is not None]
    if _dead:
        _why = f'trainer exited early: folds {_dead}'
        break
    if time.time() - _t_start > CFG['PROBE_TIMEOUT_S']:
        break
_cpu_b, _t_cpu_b = cpu_times(), time.time()
_how = {F: stop_group(_pp[F], min(20, CFG['KILL_GRACE_S'])) for F in FOLDS}
_txt = {F: read_text(f'{PROBE_DIR}/f{F}.log') for F in FOLDS}
if _why.startswith('trainer exited'):
    for F in FOLDS:
        log(f'--- probe fold {F} log tail ---\n' + tail(f'{PROBE_DIR}/f{F}.log', 40))
    REC['probe'] = dict(why=_why, rc={F: _pp[F].returncode for F in FOLDS})
    abort(f'probe: {_why} (see the log tails above)')
_sit, _src = {}, {}
for F in FOLDS:
    _its = parse_its(_txt[F])
    if F not in _hdr_ok:
        _pb = header_problems(parse_header(_txt[F]), COUNTS[F], BOX)
        if _pb:
            abort(f'probe fold {F}: header problems {_pb}')
    w = sit_window(_its)
    if w is not None and w > 0:
        _sit[F], _src[F] = w, f'window it {_its[-2][0]}->{_its[-1][0]}'
    elif _its and F in _first:
        _sit[F] = 1.5 * (_t_cpu_b - _first[F]) / 200.0       # never reached it 200: s/it > elapsed / 200
        _src[F] = f'only it {_its[-1][0]} after {_t_cpu_b - _first[F]:.0f} s: pessimistic 1.5 x elapsed / 200'
        warn(f'probe fold {F}: {_src[F]} -> {_sit[F]:.2f} s/it')
    else:
        abort(f'probe fold {F}: no progress line within {CFG["PROBE_TIMEOUT_S"]:.0f} s:\n' + tail(f'{PROBE_DIR}/f{F}.log'))
if not wait_gpus_free(120):
    warn(f'GPU memory still in use after the probe: {gpu_query()}')
_busy = cpu_busy(_cpu_a, _cpu_b) if _cpu_a is not None else None
_gu = {k: round(sum(v) / len(v), 1) for k, v in _gutil.items() if v}
S_IT = max(_sit.values())
_studies_s = sum(2.0 / v for v in _sit.values())
_verdict = ('CPU-bound (decode/augment)' if (_busy or 0) > 0.9 and (not _gu or max(_gu.values()) < 80) else
            'GPU-bound' if _gu and min(_gu.values()) >= 85 else 'mixed')
PROBE = dict(why=_why, workers=_W, s_per_it=_sit, source=_src, s_it_used=S_IT, stopped=_how,
             probe_wall_s=round(_t_cpu_b - _t_start, 1), cpu_busy_frac=None if _busy is None else round(_busy, 3),
             gpu_util_mean_pct=_gu, train_studies_per_s_total=round(_studies_s, 2),
             decode_s_per_study_thread=BENCH['heavy_s_per_study'],
             decode_cpu_s_per_s=round(_studies_s * BENCH['heavy_s_per_study'], 2), verdict=_verdict,
             progress_lines={F: parse_its(_txt[F]) for F in FOLDS})
REC['probe'] = PROBE
os.makedirs(f'{WORK}/probe', exist_ok=True)
for F in FOLDS:
    shutil.copyfile(f'{PROBE_DIR}/f{F}.log', f'{WORK}/probe/probe_f{F}.log')
stage('probe', ok=True)
log(f'probe ({_why}, {PROBE["probe_wall_s"]:.0f} s, workers {_W}/trainer): s/it {{{", ".join(f"f{F}: {v:.3f}" for F, v in _sit.items())}}} '
    f'| CPU busy {PROBE["cpu_busy_frac"]} | GPU util {_gu} | {_studies_s:.2f} train studies/s total = '
    f'{PROBE["decode_cpu_s_per_s"]} CPU-s/s of decode+augment at the one-thread bench | {_verdict}')
