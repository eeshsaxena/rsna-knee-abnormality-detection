<div style="border-radius: 16px; background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a2333 100%); border: 1px solid #30363d; padding: 28px 32px; margin-bottom: 24px; box-shadow: 0 12px 36px rgba(0,0,0,0.6); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
    <div>
      <span style="background: linear-gradient(90deg, #38bdf8, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 13px; font-weight: 800; letter-spacing: 1.8px; text-transform: uppercase;">RSNA 2026 Frontier SOTA Engine</span>
      <h1 style="color: #f0f6fc; margin: 8px 0 6px 0; font-size: 28px; font-weight: 800; line-height: 1.25;">⚡ [0.943 SOTA] RSNA Knee: Ultra-Fast 2xT4 Parallel Inference & 4-Way CoAtNet Consensus</h1>
    </div>
    <div style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; border-radius: 10px; padding: 10px 18px;">
      <span style="color: #38bdf8; font-weight: 800; font-size: 16px;">🏆 Public LB: 0.943</span>
    </div>
  </div>
  <p style="color: #8b949e; margin-top: 16px; font-size: 14.5px; line-height: 1.65;">
    Next-generation high-speed inference pipeline engineered for the RSNA Knee Abnormality Detection Challenge. Combines the verified <b>0.943 SOTA recipe (Mattia Angeli v34 / Jiwei Liu v9)</b> with high-throughput <b>Dual Tesla T4 GPU parallelization</b>, achieving sub-30 minute inference on the entire hidden test set without sacrificing a single fraction of AUC.
  </p>
  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 18px;">
    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px;">
      <div style="color: #38bdf8; font-weight: 600; font-size: 12px; text-transform: uppercase;">4-Way CoAtNet Consensus</div>
      <div style="color: #c9d1d9; font-weight: 500; font-size: 13px; margin-top: 4px;">Residual + D4 + Global96 + Repair-v1 (25% each)</div>
    </div>
    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px;">
      <div style="color: #38bdf8; font-weight: 600; font-size: 12px; text-transform: uppercase;">Dual-GPU Acceleration</div>
      <div style="color: #c9d1d9; font-weight: 500; font-size: 13px; margin-top: 4px;">2x Tesla T4 Concurrent Reader Pairs (< 30 min)</div>
    </div>
    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px;">
      <div style="color: #38bdf8; font-weight: 600; font-size: 12px; text-transform: uppercase;">DINOv2 Shared Prefix</div>
      <div style="color: #c9d1d9; font-weight: 500; font-size: 13px; margin-top: 4px;">6 Frozen Transformer Blocks Cached & Reused</div>
    </div>
    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px;">
      <div style="color: #38bdf8; font-weight: 600; font-size: 12px; text-transform: uppercase;">Medical Pretraining</div>
      <div style="color: #c9d1d9; font-weight: 500; font-size: 13px; margin-top: 4px;">RadImageNet Backbone + A5 Slice Attention</div>
    </div>
  </div>
  <div style="margin-top: 16px; font-size: 12px; color: #6e7681; border-top: 1px solid #21262d; padding-top: 12px;">
    <b>Attribution & Credits:</b> Mattia Angeli · Jiwei Liu · Dread Development · Pilkwang Kim · Sofia Anjenje · Antoine G. · prvsiyan · Marwan Mahmoud
  </div>
</div>

---

### 📊 Benchmark Evolution & Leaderboard Progress

| Pipeline Version | Public LB | CoAtNet Consensus Architecture | GPU Runtime (2x T4) | Key Breakthrough |
|:---|:---:|:---|:---:|:---|
| **Raptor Baseline** | `0.924` | 1 CoAtNet single-pass | ~9 hours | First public CoAtNet pipeline |
| **DINOsaur Blend** | `0.941` | DINOv2 + Raptor MaxSpan | ~4.5 hours | Multimodal representation gain |
| **Speedy Raptors v7** | `0.942` | Residual Top-3 + D4 DepthZone (50/50) | ~1.5 hours | Dual GPU overlap & prefix sharing |
| **⚡ Current Version (v9 / v34)** | **`0.943`** | **4-Way Consensus: Residual + D4 + Global96 + Repair-v1 (25% each)** | **< 30 mins** | **Rank-of-mean-probabilities + full parallel pairs** |

---

### 🧠 Model Architecture Pipeline

```mermaid
graph LR
    DICOM["MRI DICOM Studies"] --> DINO_PREFIX["Shared Frozen DINOv2 Prefix<br/>(6 Blocks Cached)"]
    DINO_PREFIX --> DINO_TAILS["20 Member Tails<br/>(Slot Attention)"]
    DICOM --> A5["A5 Attention Pooling<br/>(5 Folds)"]
    DICOM --> RAD["RadImageNet Encoder<br/>(E10/E13/E11 Heads)"]
    
    DINO_TAILS --> TRANS_BASE["Transformer Base Ensemble"]
    A5 --> TRANS_BASE
    RAD --> TRANS_BASE
    
    DICOM --> RAPTOR["4 Raptor Views<br/>(MaxSpan / Native384)"]
    
    subgraph COAT_CONSENSUS["4-Way CoAtNet Consensus (Parallel 2x T4)"]
        C1["Residual Gated Top-3 (25%)"]
        C2["D4 DepthZone SWA3 (25%)"]
        C3["Global96 Top-3 (25%)"]
        C4["Repair-v1 Top-3 (25%)"]
    end
    
    DICOM --> COAT_CONSENSUS
    COAT_CONSENSUS --> COAT_RANK["Mean Probabilities & Ranking"]
    
    RAPTOR --> BLEND["Final Ranked Blend<br/>(60% Raptor + 40% CoAt)"]
    COAT_RANK --> BLEND
    TRANS_BASE --> FINAL["Verified submission.csv"]
    BLEND --> FINAL
```

---

### 💡 Why this Notebook Deserves Your Upvote
1. **True SOTA Score (0.943):** Adopts the verified top-tier public consensus from Mattia Angeli & Jiwei Liu without heuristic hacks or test leakage.
2. **Blazing Fast 2xT4 Parallelism:** Finishes well within Kaggle submission limits with zero memory overflow risks (cuDNN bounded).
3. **Reproducible & Clean:** Strict artifact verification contracts at startup ensure no silent fallback or corrupted submission.
4. **Fork-Ready:** Ready to submit directly or use as a robust meta-learner anchor for your own private ensembles!

*(If you find this work valuable, please consider leaving an **Upvote 👍** to support further open-source competitive engineering!)*

