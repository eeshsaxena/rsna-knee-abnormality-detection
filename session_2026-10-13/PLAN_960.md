# Plan: Eesh wants .960 within 2 days (written 10-13 ~15:10 IST)

## Where we are
- Submitted today: HR7 (R7 verbatim, expect .956) and HFR7-R20 (R7 + our 20-member fleet at .2). Reads ~18:00-21:00.
- Public frontier: .956 = the R7 copy crowd (~334 teams). .957 = r80-119, .958 = r59-79, .960 = r32-47, top .966.
- .960 needs about +.004 over R7. Every gain we measured by blending our own models into a public blend was below
  .001 (it never moved the 3-decimal display).

## Honest assessment
- **.960 in 2 days: very unlikely (my estimate < 5%).** The teams at .960+ have stronger single models of their own
  (forum: a 27th-place team reports 5-fold OOF .937 and gold58 .938; our fleet's gold58 is ~.922). Closing that gap
  means a better model family, not more compute for the families we already have.
- **Realistic target: .957 (silver, r80-119), stretch .958.** Most of the value is on the private LB, where a
  differentiated blend jumps the whole copy crowd if it is even slightly better.

## What extra GPU can and cannot do
- OAI data must stay on the university box (NDA), so **rented GPUs can only train on RSNA data**.
- Our own model families (TTR ConvNeXt-T, Swin, ResNet) do not add on top of R7 at the 3-decimal level (measured
  10-12 / 10-13). More GPU for them = no visible gain.
- **The one GPU job worth paying for:** train extra members of the R7 recipe itself (ConvNeXt-small, 384 px, 4 folds,
  16 epochs). heliosli's training package is public (dataset `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`,
  `cloud_code/training/...`), and the R7 manifest records the full run config. Missing pieces:
  - their target files (report-derived labels + "state" targets, not public): we would use our own report labels
    (`llm_labels_v4_blend`) and switch the state head off or derive state targets ourselves;
  - the 384 px pixel cache: must be rebuilt from the DICOMs (heavy preprocessing).
- Timeline if we go: ~0.5-1 day to port and test the code on 1 GPU (smoke run), then ~8-12 h for 4 folds on
  4 GPUs, then ~4-6 h hidden scoring. That uses the whole 2 days with no room for a second try.
- Expected value: if our R7-style members reach R7 quality, adding them as a second seed of the strongest family is
  the best diversity we can buy (MODEL +.0005 to +.0015). If our targets are worse than heliosli's, they add ~0.
  Probability of a visible public gain within 2 days: ~20-30% (GUESS).

## GPU to buy (only for the R7-recipe job)
- 4 x RTX 4090 (24 GB) or 2 x A100 80 GB, ~24-30 h. Vast.ai / RunPod: ~$0.35-0.45 per 4090-hour or ~$1.2-1.6 per
  A100-hour. **Budget ~$40-60.** Disk >= 300 GB (DICOM + pixel cache). Not a China-located host (pip/wheel problems
  on 10-09).
- Do NOT buy more for our own model families: measured no visible gain.

## Free levers (no GPU), using the remaining submissions
1. Read HR7 and HFR7-R20 (~18:00-21:00). If HFR7 shows .957: our fleet adds on top of R7, so build HFR7 variants
   (w .3, + D2/D3 seeds) and make HFR7 pick B.
2. Raptor second checkpoint (nartaa ships an r224 SWA checkpoint too): 3-way blend Raptor384 + Raptor224 + R7, ~25 min
   extra runtime. Small, cheap, differentiated.
3. Final picks (10-22): A = HR7 (copy floor), B = best differentiated blend (pre-registered in PREREG_10-13.md).

## Decision for Eesh
- If you want to spend: approve ~$40-60 for the R7-recipe job and say which provider (Vast / RunPod). I start the
  code port on the box CPU right away; GPUs are only rented once the 1-GPU smoke run works.
- If not: free levers only, and the box keeps training OAI models for pick B.
