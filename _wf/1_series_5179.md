All data gathered. Here is the findings block.

---

# train_series.csv — Findings

**File:** `C:/Users/eeshs/Downloads/rsna-knee/_data/train_series.csv` · 24,371 series rows · 4,407 studies · 0 nulls · SeriesInstanceUID is unique (no dup rows).

## 1. Series-per-study distribution
- mean **5.53**, median **5**, range **3–14**, std 1.39. IQR 5–6.
- Full tail: 3→1 study, 4→675, 5→**2299**, 6→698, 7→310, 8→145, 9→176, 10→74, 11→21, 12→5, 13→2, 14→1.
- 95th pct = 9, 99th pct = 10. So ≥7 series is uncommon (~14%), ≥10 is the rare long tail (~2.3%).

## 2. Plane coverage
- Series-level: Sagittal 9,864 / Coronal 8,609 / Axial 5,898.
- **Every study (4407/4407 = 100%) has all three planes.** Zero studies missing any plane. Per-study distinct-plane count is 3 for all.
- Series-per-plane within a study: Sagittal mean 2.24 (1–6), Coronal mean 1.95 (1–5), Axial mean 1.34 (1–7). Axial is usually a single series (3313/4407 studies have exactly 1 axial); Sagittal/Coronal usually 2.

## 3. Redundancy — Fluid_Sensitive vs Fat_Suppression
- **Byte-for-byte identical: 24,371/24,371 rows equal (crosstab is perfectly diagonal, off-diagonal = 0).** They carry zero independent information — **drop one column.** Both are 1 for 14,010 rows (fat-suppressed / fluid-sensitive) and 0 for 10,361.
- No other redundancy: StudyInstanceUID (grouping key), SeriesInstanceUID (unique id), Anatomical_Plane are all independent and non-null.

## 4. Sequence-type coverage patterns
Treating the real sequence axis as **plane × fluid-sensitive (FS1 = fat-sat/fluid-sensitive e.g. STIR/T2-FS/PD-FS; FS0 = non-fat-sat e.g. T1/PD)** gives 6 canonical buckets. Per-study presence:

| Bucket | % of studies |
|---|---|
| Axial_FS1 (fat-sat axial) | **100.0%** (guaranteed) |
| Sagittal_FS0 | 96.8% |
| Coronal_FS1 | 96.4% |
| Sagittal_FS1 | 94.2% |
| Coronal_FS0 | 77.3% |
| **Axial_FS0** | **19.4%** (rare) |

- Only **12.8% of studies (566)** carry all 6 buckets; most have 5 (2578) or 4 (1259); 4 studies have just 3.
- Just **12 distinct present-set signatures** exist. Two templates dominate: `Sag(FS0+FS1) + Cor(FS0+FS1) + Ax_FS1` = **57.1%**, and `Sag(FS0+FS1) + Cor_FS1 + Ax_FS1` = **16.9%**. Together with the all-6 template these three cover **86.8%** of studies.
- Per-plane fat-sat mix: Axial is 80% fat-sat (FS0 essentially never appears alone — 0 studies have axial-FS0-only); Coronal often fat-sat-only (1001 studies); Sagittal is the most balanced (both FS present in 4009/4407).
- Gold-58 mirrors the full set: 58/58 have all 3 planes, mean 5.79 series, bucket-count dist {4:16, 5:31, 6:11} — same shape, so a design tuned on the population transfers to gold.

## 5. Implications for model design

**Per-plane 2.5D architecture**
- Build **three plane-specific branches (Sagittal / Coronal / Axial)** — every study is guaranteed to feed all three, so a 3-branch design has no missing-branch case on train. Late-fuse (concat/attention pool) the three plane embeddings into the 12-label head.
- **Do not** build separate branches per (plane × fluid) bucket: Axial_FS0 exists in only 19% of studies and 4 buckets are non-universal, so bucket-level branches would starve or need masking. Instead pass fluid-sensitive (one bit, from the deduplicated column) as **series metadata** — either an input channel/embedding or by keeping FS0/FS1 in separate 2.5D stacks *within* each plane branch and pooling.
- Multiple series per plane is the norm (Sag ~2, Cor ~2), so each plane branch must **aggregate a variable number of series** (attention/mean pooling over per-series embeddings), not assume one volume per plane.
- 2.5D stacking: form adjacent-slice channel groups within a series; pool slices → series embedding → plane embedding → study embedding. Fat-sat fluid-sensitive series (FS1, universal on axial, dominant on coronal) are the effusion/synovitis/Baker's/marrow-edema signal — weight/route them deliberately for those labels; FS0 (T1/PD) carries anatomy for ACL/MCL/meniscus/OA.

**Robust handling of variable series counts at inference**
- Design for **variable-length input per plane** via pooling over a set (masked attention or mean), never a fixed slot count. Cap/sample the long tail (≥10 series, ~2%) to bound the ≤9h T4 budget; a per-plane cap of ~3–4 series covers ~95% of studies with no loss.
- **Test set is images-only and may not honor train invariants** — do not hard-code "all 3 planes always present" or "axial fat-sat always exists." Detect plane/fluid from DICOM (or a lightweight classifier) at inference, route each series to its branch, and make each plane branch tolerate zero series (learned/zero-fill embedding + mask) so a study missing a plane still scores. Train-time augmentation should randomly drop a plane/bucket to harden against this even though train never shows it.
- Since 87% of studies fall in 3 templates, a **canonical-template fast path** (fixed routing) with a general masked-pooling fallback keeps inference simple and fast for the common case while staying correct on the tail.
- Fluid-sensitive is the only real per-series covariate here and it is a clean single bit after dedup — cheap to embed, high value for fluid-related labels.