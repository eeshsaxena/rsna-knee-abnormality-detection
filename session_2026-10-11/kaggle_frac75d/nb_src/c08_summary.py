# [step 7] summary for the commit log (the receipt has everything).
shutil.rmtree(PROBE_DIR, ignore_errors=True)
print(json.dumps(dict(status=REC['status'], plan=REC.get('plan'), probe={k: REC['probe'][k] for k in
                      ('s_per_it', 's_it_used', 'cpu_busy_frac', 'gpu_util_mean_pct', 'verdict', 'workers')},
                      warnings=REC['warnings'], outputs_gb=REC.get('outputs_gb')), indent=1, default=str))
for F in FOLDS:
    x = REC['train']['folds'][F]
    print(f'fold {F}: complete={x["complete"]} ended={x["ended"]} best ckpts {x["best_ckpts"]} oof {x["oof"]}')
for root, _, fs in os.walk(WORK):
    for f in sorted(fs):
        p = os.path.join(root, f)
        print(f'{os.path.getsize(p) / 1e6:10.1f} MB  {os.path.relpath(p, WORK)}')
