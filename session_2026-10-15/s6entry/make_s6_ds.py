#!/usr/bin/env python
# make_s6_ds.py (2026-10-16, Claude): builds the private dataset dir for an S6 entry model (S6ENTRY_RUNBOOK step 5).
#   - copies the model from the box (scp, sha-checked against the box's own sha256sum)
#   - <stem>.run_config.json = nartaa's E_prior_s42_fall.run_config.json (the driver checks only the geometry keys, which are
#     identical for every S6fall-family member; recorded as such in the DONE note)
#   - <stem>.DONE.json = {model, model_sha256, epochs 16, note}; the patched driver checks model + sha (+ epochs: expect_epochs 16)
#   - dataset-metadata.json (private, licence other)
# Usage: python make_s6_ds.py --box-model ~/s6seed/soup/E_prior_s4243_fsoup.pt --slug eeshsaxena/rsna-s6-soup-s4243 --note "..."
# Prints the patch_s6_model.py arguments to use. The dataset itself is created by main (kaggle datasets create, staging-dir fix).
import os, sys, json, hashlib, argparse, subprocess, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
BOX = 'training@192.168.10.30'
ap = argparse.ArgumentParser()
ap.add_argument('--box-model', required=True)
ap.add_argument('--slug', required=True)
ap.add_argument('--note', default='')
A = ap.parse_args()
name = os.path.basename(A.box_model)
assert name.startswith('E_prior_s') and '_f' in name and name.endswith('.pt'), name
stem = name[:-3]
owner, slug = A.slug.split('/')
d = os.path.join(HERE, 'ds_' + slug)
os.makedirs(d, exist_ok=True)
box_sha = subprocess.run(['ssh', '-o', 'BatchMode=yes', BOX, 'sha256sum ' + A.box_model], capture_output=True, text=True,
                         check=True).stdout.split()[0]
dst = os.path.join(d, name)
if not (os.path.isfile(dst) and hashlib.sha256(open(dst, 'rb').read()).hexdigest() == box_sha):
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '%s:%s' % (BOX, A.box_model), dst], check=True)
sha = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
assert sha == box_sha, ('copy differs from the box file', sha, box_sha)
rc_src = os.path.join(HERE, '..', 'silver', 'pulled', 's6fall_meta', 'E_prior_s42_fall.run_config.json')
shutil.copyfile(rc_src, os.path.join(d, stem + '.run_config.json'))
done = {'model': name, 'model_sha256': sha, 'epochs': 16,
        'note': (A.note + ' | run_config = nartaa E_prior_s42_fall.run_config.json (geometry keys identical for the family)').strip(' |')}
json.dump(done, open(os.path.join(d, stem + '.DONE.json'), 'w'), indent=1)
json.dump({'title': slug.replace('-', ' '), 'id': A.slug, 'licenses': [{'name': 'other'}]},
          open(os.path.join(d, 'dataset-metadata.json'), 'w'), indent=1)
print('S6_DS OK', d, '| model', name, sha[:16], '| files', sorted(os.listdir(d)))
print('patch args: --ds %s --model-file %s --model-sha %s --run-config %s --done %s --expect-epochs 16 --expect-train none'
      % (A.slug, name, sha, stem + '.run_config.json', stem + '.DONE.json'))
