# make_figs.py (writeup figures, 10-23). Draws the 5 writeup figures from AGGREGATE numbers already on the laptop.
# Every number below is copied from (or, for fig 5, recomputed and self-checked against) the file named next to it; FIGS.md lists them.
# OAI governance: only aggregate AUC deltas, sd, P, phi / tet / rho from laptop logs; no OAI row, id or per-knee value.
# Competition data: no images, no report text, no study-level labels appear in any figure.
# Run: py -3.12 make_figs.py   (needs data/dose_gold58.json from dose_gold58.py)
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.dirname(HERE)
DATA = os.path.join(FIGS, 'data')
DPI = 200            # 8.0 in wide -> 1600 px

# ---------- style (dataviz reference palette, validated on #ffffff: adjacent CVD dE >= 9.1, normal >= 19.6) ----------
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6'
SERIOUS = '#ec835a'
plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': True,
})
MINUS = '\u2212'
fmt = lambda v, d=3: (MINUS if v < 0 else '') + ('%.*f' % (d, abs(v)))
fmts = lambda v, d=4: ('+' if v > 0 else (MINUS if v < 0 else '')) + ('%.*f' % (d, abs(v)))


def save(fig, name):
    p = os.path.join(FIGS, name)
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print('wrote', p)


# =====================================================================================================================
# (1) pipeline.png
# =====================================================================================================================
def box(ax, x, y, w, h, fc, ec, lw=1.4, r=1.2, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=%g' % r, fc=fc, ec=ec, lw=lw, ls=ls,
                                mutation_aspect=1))


def arrow(ax, x0, y0, x1, y1, c=INK2):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=c, lw=1.6, mutation_scale=16,
                                                                     shrinkA=0, shrinkB=0))


def tint(hex_, a):
    h = hex_.lstrip('#'); rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return tuple(1 - a * (1 - rgb))


def fig_pipeline():
    fig = plt.figure(figsize=(8.0, 10.55))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(-4.2, 127.5); ax.axis('off')
    ax.text(3, 124.2, 'Final pipeline: the c4 blend plus E-OA', fontsize=15, fontweight='bold', va='center')
    ax.text(3, 120.4, 'One Kaggle code-competition notebook; all three model families use public weights. Read top to bottom.',
            fontsize=11, color=INK2, va='center')

    # decode
    box(ax, 3, 106, 94, 10.5, tint(MUTED, .10), MUTED)
    ax.text(50, 113.4, 'Test study DICOMs (sagittal, coronal and axial MRI series)', ha='center', va='center', fontsize=12,
            fontweight='bold')
    ax.text(50, 109.0, 'Custom JPEG-lossless decoder cell (pylibjpeg wheels, self-tested on embedded images):\n'
            'without it Raptor zero-fills JPEG-lossless slices and the other stages drop the whole series', ha='center', va='center', fontsize=10.5,
            color=INK2, linespacing=1.3)
    for xc in (19, 52, 83):
        arrow(ax, xc, 106, xc, 101.6)

    # members
    yb, hb = 75, 26.6
    members = [
        (3, 32, BLUE, 'Raptor CoAtNet', ['nartaa public SWA weights', 'r384, anatomical mirror TTA', '+ r224 crop model',
                                         'rank mix .75 r384 + .25 r224']),
        (37, 30, ORANGE, 'R6 compact ConvNeXt-S', ['heliosli bundle, 4 folds', 'flip TTA (M / L swapped back)',
                                                   'own Effusion and', "Baker's stackers"]),
        (69, 28, AQUA, 'S6fall', ['nartaa retrain of the', 'heliosli S6 architecture', 'own subprocess and', 'decoder dir']),
    ]
    for x, w, col, title, lines in members:
        box(ax, x, yb, w, hb, tint(col, .13), col, lw=1.8)
        ax.text(x + w / 2, yb + hb - 3.6, title, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x + w / 2, yb + hb - 11.4, '\n'.join(lines), ha='center', va='center', fontsize=10.5, color=INK2,
                linespacing=1.45)
    # canary badges (status colour always with an icon and a label)
    def badge(xc, y, txt, w=24):
        box(ax, xc - w / 2, y, w, 6.6, 'white', SERIOUS, lw=1.4, r=1.0)
        ax.text(xc - w / 2 + 1.8, y + 3.3, '!', ha='center', va='center', fontsize=12, fontweight='bold', color=SERIOUS)
        ax.text(xc - w / 2 + 3.8, y + 3.3, txt, ha='left', va='center', fontsize=9.8, color=INK, linespacing=1.25)
    badge(19, 76.4, 'canary: if r224 fails,\nFracture = 0.5 for all rows')
    badge(83, 76.4, 'canary: if S6fall fails,\nACL = 0.5 for all rows')

    arrow(ax, 19, 75.0, 19, 70.6); arrow(ax, 52, 75.0, 52, 70.6)
    arrow(ax, 83, 75.0, 83, 52.6)

    # per-label blend
    box(ax, 3, 55, 64, 15.6, tint(MUTED, .10), MUTED)
    ax.text(35, 67.2, 'Per-label rank blend', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(35, 63.3, r'$w\,\mathrm{pct}(\mathrm{Raptor}) + (1-w)\,\mathrm{pct}(\mathrm{R6})$', ha='center', va='center',
            fontsize=12)
    ax.text(35, 58.0, "Raptor share w:  Medial OA .5,  Lateral OA .2,  PF OA .5,\nSynovitis .8,  Baker's .5,  "
            "Lateral Meniscus .4,  Fracture .65,  rest .5", ha='center', va='center', fontsize=10.2, color=INK2,
            linespacing=1.35)
    arrow(ax, 35, 55, 35, 52.6)

    # final mix
    box(ax, 3, 42.5, 94, 10.1, tint(MUTED, .10), MUTED)
    ax.text(50, 49.6, 'c4 output, per label', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(50, 45.4, r'$0.8\,\mathrm{pct}(\mathrm{blend}) + 0.2\,\mathrm{pct}(\mathrm{S6fall})$'
            '      (pct = average-tie percentile rank)', ha='center', va='center', fontsize=11.5)
    arrow(ax, 50, 42.5, 50, 38.6)

    # E-OA
    box(ax, 3, 13.5, 94, 25.1, tint(VIOLET, .10), VIOLET, lw=2.4)
    ax.text(50, 35.4, 'E-OA: decorrelate the three OA columns (post-processing, no extra runtime)', ha='center',
            va='center', fontsize=12, fontweight='bold')
    ax.text(50, 30.3, r'$\mathrm{out}_t = \mathrm{pct}\left(\mathrm{pct}(f_t) - \frac{a}{2}\sum_{o \neq t}'
            r'\,w_{t,o}\,\mathrm{pct}(f_o)\right)$', ha='center', va='center', fontsize=14)
    ax.text(50, 24.6, 't, o in {Medial OA, Lateral OA, PF OA}; the other 9 columns are untouched.\n'
            'sym: w = 1.   pair-weighted (pw): w(Lateral, PF) 1.416, w(Medial, Lateral) .876, w(Medial, PF) .708',
            ha='center', va='center', fontsize=10.5, color=INK2, linespacing=1.4)
    ax.text(50, 17.6, 'Pick A, swide: c4 + pw a .25 + (Lateral OA <- Medial Meniscus .05) + (MCL <- Medial Meniscus .1)\n'
            'Pick B, r6pw20: the R6 entry (uniform blend, r224 share .35) + pw a .2',
            ha='center', va='center', fontsize=10.5, color=INK, linespacing=1.4)
    arrow(ax, 50, 13.5, 50, 10.4)

    # submission
    box(ax, 30, 4.4, 40, 6.0, 'white', INK2)
    ax.text(50, 7.4, 'submission.csv (12 probabilities per study)', ha='center', va='center', fontsize=11.5)
    ax.text(3, -0.6, 'Canaries: a failed stage sets one column to a constant, so a broken hidden rerun shows up (about 0.916 '
            'public)\ninstead of passing silently. 4 late entries hit the S6fall canary on flaky Kaggle days.',
            fontsize=9.6, color=INK2, va='center', linespacing=1.3)
    save(fig, 'pipeline.png')


# =====================================================================================================================
# (2) coupling.png  (session_2026-10-19/ex/coupling_map.md sec 2, rows 1, 2, 4)
# =====================================================================================================================
def fig_coupling():
    pairs = ['Medial OA\nvs Lateral OA', 'Medial OA\nvs PF OA', 'Lateral OA\nvs PF OA']
    series = [  # (label, color, values for MOA-LOA, MOA-PFOA, LOA-PFOA, sd or None)
        ('c4 model scores (Spearman, gold58)', BLUE, [.85, .82, .81], None),
        ('report labels, LLM-read v2 (tetrachoric)', ORANGE, [.83, .80, .77], None),
        ('expert labels, gold58 (tetrachoric)', AQUA, [.51, .56, .45], None),
        ('gold74 = gold58 + 16 gold-convention reads (\u00b11 sd)', YELLOW, [.59, .61, .39], [.16, .13, .20]),
        ('OAI MOAKS expert reads (tetrachoric)', MAGENTA, [.12, .13, -.01], None),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.27)
    x = np.arange(3); bw = 0.158; gap = 0.012
    for i, (lab, col, vals, sd) in enumerate(series):
        xs = x + (i - 2) * (bw + gap)
        ax.bar(xs, vals, bw, color=col, label=lab, zorder=3, linewidth=0)
        if sd:
            ax.errorbar(xs, vals, yerr=sd, fmt='none', ecolor=INK2, elinewidth=1.1, capsize=3, zorder=4)
        for xi, v, k in zip(xs, vals, range(3)):
            top = v + (sd[k] if sd else 0)
            yy = (top + 0.02) if v >= 0 else 0.02
            ax.text(xi, yy, fmt(v, 2).replace('0.', '.'), ha='center', va='bottom', fontsize=10, color=INK)
    ax.axhline(0, color=INK2, lw=1.0, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=11.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(-0.08, 1.0); ax.set_ylabel('pairwise coupling (latent correlation)')
    ax.grid(axis='x', visible=False)
    fig.text(0.10, 0.955, 'Report-trained models couple the OA compartments like the reports do', fontsize=14,
             fontweight='bold', ha='left', va='center')
    fig.text(0.10, 0.895, 'The rubric scores each compartment on its own. Our blend couples them (.81 to .85) about as much as the\n'
             'report labels do (.77 to .83); expert labels couple them far less (.39 to .61 on RSNA, \u2212.01 to .13 on OAI).',
             fontsize=10.5, color=INK2, ha='left', va='center', linespacing=1.35)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower left', bbox_to_anchor=(0.085, 0.0), ncol=2, frameon=False, fontsize=10.5,
               handlelength=1.2, columnspacing=1.4)
    save(fig, 'coupling.png')


# =====================================================================================================================
# (3) oai_validation.png  (g1_box_run.log [mean P_FT]; g1_replication_box.log [mean P_FT] x 2)
# =====================================================================================================================
def fig_oai():
    A = [-0.1, 0.0, 0.1, 0.2]
    models = [  # name, color, marker, deltas at a -.1/.1/.2, boot sd at -.1/.1/.2
        ('Model 1: 5-fold mean (s1043)', BLUE, 'o', [-.00659, .00624, .01154], [.00038, .00042, .00089]),
        ('Model 2: single fold (s1044)', ORANGE, 's', [-.00581, .00498, .00859], [.00040, .00044, .00092]),
        ('Model 3: single fold, 288 px (s1045)', AQUA, '^', [-.00610, .00531, .00929], [.00042, .00045, .00094]),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    fig.subplots_adjust(left=0.13, right=0.76, top=0.80, bottom=0.14)
    ax.axvspan(-0.135, -0.05, color=tint(MUTED, .10), zorder=0, lw=0)
    ax.text(-0.0925, 0.0128, 'control:\nborrowing\nmust lose', ha='center', va='top', fontsize=10, color=INK2, linespacing=1.25)
    ax.axhline(0, color=INK2, lw=1.1, zorder=2)
    for name, col, mk, d, sd in models:
        y = [d[0], 0.0, d[1], d[2]]; e = [1.96 * sd[0], 0.0, 1.96 * sd[1], 1.96 * sd[2]]
        ax.errorbar(A, y, yerr=e, color=col, marker=mk, ms=8, lw=2, capsize=3.5, elinewidth=1.2, zorder=3,
                    markeredgecolor='white', markeredgewidth=1.0, label=name)
    # direct labels at a = .2 (values spread to avoid collisions)
    ends = [(.01154, .01154, 'Model 1  ' + fmts(.01154)), (.00929, .00955, 'Model 3  ' + fmts(.00929)),
            (.00859, .00820, 'Model 2  ' + fmts(.00859))]
    cols = {'Model 1': BLUE, 'Model 2': ORANGE, 'Model 3': AQUA}
    for yv, ylab, txt in ends:
        ax.annotate(txt, xy=(0.2, yv), xytext=(0.222, ylab), fontsize=10.5, va='center', color=INK,
                    annotation_clip=False, arrowprops=dict(arrowstyle='-', color=cols[txt[:7]], lw=1.0))
    ax.set_xlim(-0.135, 0.215); ax.set_ylim(-0.0095, 0.0135)
    ax.set_xticks(A); ax.set_xticklabels([MINUS + '0.1', '0\n(no change)', '0.1', '0.2'])
    ax.set_xlabel('E-OA strength a (sym)')
    ax.set_ylabel('change in mean AUC of the 3 OA labels')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: fmts(v, 3) if abs(v) > 1e-12 else '0'))
    fig.text(0.13, 0.955, 'E-OA validated on 2,399 OAI knees read by expert radiologists', fontsize=14, fontweight='bold',
             ha='left', va='center')
    fig.text(0.13, 0.895, '3 OAI-free, report-trained models; labels = MOAKS full-thickness cartilage loss per compartment.\n'
             'Pre-registered rule; P(>0) = 1.000 for every model at a .1 and .2. Error bars: 95 % paired bootstrap.',
             fontsize=10.5, color=INK2, ha='left', va='center', linespacing=1.35)
    ax.legend(loc='lower right', frameon=True, framealpha=1.0, edgecolor='white', fontsize=10.2, handlelength=1.8)
    save(fig, 'oai_validation.png')


# =====================================================================================================================
# (4) lb_progress.png  (session_2026-10-23_scores.json, Kaggle API display values)
# =====================================================================================================================
def fig_lb():
    S = json.load(open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-23_scores.json'))
    seq = [('57201874', 'S6F', 'Raptor + R7\n+ S6fall .2'), ('57248084', 'R6', 'R7 -> R6\nswap'),
           ('57262115', 'HW', 'per-label\ntable'), ('57316395', 'a25', 'r224\nshare .25'), ('57343435', 'c4', 'table\nv2'),
           ('57413380', 'eoa10', 'E-OA\nsym .1'), ('57413368', 'eoa20', 'E-OA\nsym .2'), ('57425970', 'pw30', 'E-OA\npw .3'),
           ('57464326', 'swide', 'pw .25 +\nmeniscus')]
    pub = [float(S[i][0]) for i, _, _ in seq]; prv = [float(S[i][1]) for i, _, _ in seq]
    x = np.arange(len(seq))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.0, 7.3), sharex=True)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.835, bottom=0.12, hspace=0.30)
    for ax, ys, col, mk, ttl, lo in ((a1, pub, BLUE, 'o', 'Public leaderboard', 0.955),
                                    (a2, prv, ORANGE, 's', 'Private leaderboard (final standings)', 0.9395)):
        ax.axvspan(4.5, 8.5, color=tint(VIOLET, .08), lw=0, zorder=0)
        ax.plot(x, ys, color=col, lw=2, marker=mk, ms=8, markeredgecolor='white', markeredgewidth=1.0, zorder=3)
        for xi, v in zip(x, ys):
            ax.text(xi, v + 0.00042, '%.3f' % v, ha='center', va='bottom', fontsize=10.5, color=INK)
        ax.set_ylim(lo, lo + 0.0065)
        ax.set_yticks(np.round(np.arange(lo + 0.0005, lo + 0.0066, 0.001), 4) if lo == 0.9395 else
                      np.round(np.arange(lo, lo + 0.0066, 0.001), 3))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
        ax.set_title(ttl, fontsize=12.5, pad=6)
        ax.set_ylabel('macro AUC')
        ax.grid(axis='x', visible=False)
    a1.text(6.5, 0.9556, 'E-OA family', ha='center', va='center', fontsize=10.5, color=VIOLET, fontweight='bold')
    a1.annotate('E-OA: +0.001', xy=(5, pub[5]), xytext=(3.2, 0.9593), fontsize=10.5, color=INK, ha='center',
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.1))
    a2.annotate('E-OA: +0.001\n(held on private)', xy=(5, prv[5]), xytext=(3.0, 0.9448), fontsize=10.5, color=INK,
                ha='center', arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.1))
    a1.annotate('final pick A', xy=(8, pub[8] - 0.00012), xytext=(8, 0.9577), fontsize=10.5, ha='center', color=INK,
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.1))
    a2.set_xticks(x); a2.set_xticklabels(['%s\n%s' % (k, d) for _, k, d in seq], fontsize=10.2)
    a2.tick_params(axis='x', length=0)
    for t in a2.get_xticklabels():
        t.set_linespacing(1.15)
    fig.text(0.11, 0.955, 'Pre-registered entries in order: E-OA moved both boards',
             fontsize=13.5, fontweight='bold', ha='left', va='center')
    fig.text(0.11, 0.905, 'Kaggle display values (3 decimals) on the path to the final pick, in submission order. Final: private '
             '0.944,\nrank 246 / 6,425, silver (cut at rank 321). Pick B, r6pw20 copy 3, also scored 0.959 / 0.944.',
             fontsize=10.5, color=INK2, ha='left', va='center', linespacing=1.35)
    save(fig, 'lb_progress.png')


# =====================================================================================================================
# (5) dose_curve.png  (data/dose_gold58.json from dose_gold58.py; self-check vs gate logs PASS)
# =====================================================================================================================
def fig_dose():
    D = json.load(open(os.path.join(DATA, 'dose_gold58.json')))
    assert D['self_check'] == 'PASS'
    sub = {'sym': {.1: 'eoa10', .2: 'eoa20', .3: 'eoa30'}, 'pw': {.2: 'pw20', .25: 'pw25', .3: 'pw30'}}
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.80, bottom=0.14)
    ax.axhline(0, color=INK2, lw=1.1, zorder=2)
    for mode, col, mk, lab in (('sym', BLUE, 'o', 'sym (w = 1)'), ('pw', ORANGE, 's', 'pair-weighted (pw)')):
        a = np.array([r['a'] for r in D['curve'][mode]]); d = np.array([r['delta'] for r in D['curve'][mode]])
        ax.plot(a, d, color=col, lw=2, zorder=3, label=lab)
        filled = np.array([any(abs(x - k) < 1e-9 for k in sub[mode]) for x in a])
        ax.scatter(a[~filled], d[~filled], s=46, marker=mk, facecolor='white', edgecolor=col, linewidth=1.6, zorder=4)
        ax.scatter(a[filled], d[filled], s=64, marker=mk, facecolor=col, edgecolor='white', linewidth=1.0, zorder=5)
    # direct labels
    ax.text(0.325, 0.00158, 'sym', color=INK, fontsize=11.5, ha='center')
    ax.text(0.35, 0.00292, 'pw', color=INK, fontsize=11.5, ha='center')
    ax.annotate('swide dose (pw .25)', xy=(0.25, 0.00255), xytext=(0.14, 0.0035), fontsize=10.5, ha='center',
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.1))
    ax.set_xlim(-0.01, 0.61); ax.set_ylim(-0.0034, 0.0042)
    ax.set_xlabel('E-OA strength a')
    ax.set_ylabel('macro AUC change vs c4')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: fmts(v, 3) if abs(v) > 1e-12 else '0'))
    fig.text(0.13, 0.955, 'Dose curve on 58 hand-labelled studies (info only)', fontsize=14, fontweight='bold', ha='left',
             va='center')
    fig.text(0.13, 0.895, 'gold58 = the 58 expert-labelled train studies; c4 base macro 0.9411. Filled markers = doses that were\n'
             'submitted. Too small to choose a dose (paired bootstrap sd about 0.001 at a .2); used as a code gate only.',
             fontsize=10.5, color=INK2, ha='left', va='center', linespacing=1.35)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=BLUE, lw=2, marker='o', ms=7, label='sym (w = 1)'),
          Line2D([], [], color=ORANGE, lw=2, marker='s', ms=7, label='pair-weighted (pw)'),
          Line2D([], [], color=MUTED, lw=0, marker='o', ms=7, label='submitted dose'),
          Line2D([], [], color=MUTED, lw=0, marker='o', ms=7, markerfacecolor='white', label='offline only')]
    ax.legend(handles=hs, loc='lower left', frameon=False, fontsize=10.5)
    save(fig, 'dose_curve.png')


if __name__ == '__main__':
    fig_pipeline(); fig_coupling(); fig_oai(); fig_lb(); fig_dose()
