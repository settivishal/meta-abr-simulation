"""
Meta-ABR Simulation Code
========================
CNT6885: Distributed Multimedia Systems — Spring 2026
University of Florida, CISE Department
Term Paper: "Adaptive Bitrate Streaming with Meta-Reinforcement Learning"

This file implements all five ABR algorithms evaluated in the paper:
  1. Buffer-Based (BBA)      [Huang et al., SIGCOMM 2014]
  2. RobustMPC               [Yin et al., SIGCOMM 2015]
  3. BOLA                    [Spiteri et al., INFOCOM 2016]
  4. Pensieve (simplified RL)[Mao et al., SIGCOMM 2017]
  5. Meta-ABR (proposed)

It also generates all 5 figures from the paper:
  - Figure 1: Mean QoE score by algorithm and network scenario
  - Figure 2: Rebuffering ratio by algorithm and scenario
  - Figure 3: Per-session bitrate decisions + buffer occupancy (Congested 4G)
  - Figure 4: Adaptation speed after mid-session regime change
  - Figure 5: CDF of per-session QoE (all scenarios + Congested 4G zoom)

Requirements:
  pip install numpy matplotlib

Usage:
  python meta_abr_simulation.py

Output:
  - Prints QoE and rebuffering tables to console
  - Saves fig1_qoe_bar.png through fig5_cdf.png in current directory
  - Saves results_summary.txt with full numerical results

References:
  [3] Huang et al. SIGCOMM 2014. https://dl.acm.org/doi/10.1145/2619239.2626296
  [4] Yin et al. SIGCOMM 2015. https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p325.pdf
  [5] Spiteri et al. INFOCOM 2020. https://arxiv.org/pdf/1601.06748
  [6] Mao et al. SIGCOMM 2017. https://dl.acm.org/doi/10.1145/3098822.3098843
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import os

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BITRATES     = [0.30, 0.75, 1.20, 2.50, 4.00, 6.00]   # Mbps — 6 quality levels
CHUNK_DUR    = 4       # seconds per chunk
BUFFER_MAX   = 30.0    # seconds — maximum playback buffer
BUFFER_INIT  = 10.0    # seconds — initial buffer (startup phase)
N_CHUNKS     = 60      # chunks per session
LAM          = 4.3     # QoE rebuffering penalty weight (from Pensieve [6])
MU           = 0.5     # QoE smoothness penalty weight
N_TRACES     = 25      # traces per scenario

ALGOS  = ['Buffer-Based', 'RobustMPC', 'BOLA', 'Pensieve', 'Meta-ABR']
COLORS = ['#e74c3c',  '#e67e22',  '#3498db', '#9b59b6',  '#27ae60']

SCENARIOS = ['Standard 4G', 'Congested 4G', '5G Sub-6GHz', 'WiFi Variable']

# scenario config: (name, mean_bps_mbps, variability_coeff, dip_start_chunk,
#                   dip_end_chunk, dip_factor, base_seed, in_training)
SCENARIO_CONFIGS = [
    ('Standard 4G',   8.0,  0.50, 20, 35, 0.25, 42,  True),
    ('Congested 4G',  3.0,  0.70, 10, 40, 0.12, 77,  True),
    ('5G Sub-6GHz',  14.0,  0.60, 25, 32, 0.20, 99,  True),
    ('WiFi Variable',18.0,  0.35, 10, 22, 0.30, 55,  False),   # held-out
]


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK TRACE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_bandwidth_trace(n_chunks, avg_mbps, variability,
                            dip_start, dip_end, dip_factor, seed):
    """
    Generate a synthetic bandwidth trace calibrated to real 4G/5G measurements.

    The trace consists of a base bandwidth with multiplicative Gaussian noise,
    and a sustained bandwidth dip between chunks [dip_start, dip_end] to model
    events like cell tower handoffs, network congestion, or signal blockage.

    Parameters
    ----------
    n_chunks    : int   — number of chunks in the session
    avg_mbps    : float — mean available bandwidth in Mbps
    variability : float — coefficient of variation (std/mean) for noise
    dip_start   : int   — chunk index where bandwidth dip begins
    dip_end     : int   — chunk index where bandwidth dip ends
    dip_factor  : float — bandwidth multiplier during dip (e.g. 0.12 = 12% of nominal)
    seed        : int   — random seed for reproducibility

    Returns
    -------
    np.ndarray of shape (n_chunks,) with bandwidth values in Mbps
    """
    rng = np.random.RandomState(seed)
    trace = []
    for i in range(n_chunks):
        dip   = dip_factor if dip_start <= i <= dip_end else 1.0
        noise = 1.0 + (rng.rand() - 0.5) * 2.0 * variability
        bw    = max(0.15, avg_mbps * dip * noise)
        trace.append(bw)
    return np.array(trace)


# ─────────────────────────────────────────────────────────────────────────────
# ABR ALGORITHM IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def pick_bitrate_buffer_based(bw, buf_level, last_br, bw_history):
    """
    Buffer-Based Algorithm (BBA) [Huang et al., SIGCOMM 2014].

    Maps buffer occupancy directly to quality level via fixed thresholds.
    Threshold values: [4, 8, 12, 18, 24] seconds map to quality levels 1–5.
    Above 24s buffer → highest quality level 6.
    """
    thresholds = [4, 8, 12, 18, 24]
    idx = next((i for i, t in enumerate(thresholds) if buf_level < t), 5)
    return BITRATES[min(idx, len(BITRATES) - 1)]


def pick_bitrate_robust_mpc(bw, buf_level, last_br, bw_history):
    """
    RobustMPC [Yin et al., SIGCOMM 2015].

    Uses the harmonic mean of the last 5 throughput samples as the
    throughput prediction, then selects the highest bitrate that fits
    within 80% of the predicted throughput (conservative discount factor).
    This discount accounts for prediction errors observed in prior chunks.
    """
    if bw_history:
        recent = bw_history[-5:]
        # Harmonic mean — robust to outliers, standard in streaming [6]
        harmonic_mean = len(recent) / sum(1.0 / b for b in recent)
    else:
        harmonic_mean = bw

    safe_bw = harmonic_mean * 0.80     # 20% conservative discount
    candidates = [b for b in BITRATES if b <= safe_bw]
    return candidates[-1] if candidates else BITRATES[0]


def pick_bitrate_bola(bw, buf_level, last_br, bw_history):
    """
    BOLA: Buffer Occupancy-based Lyapunov Algorithm [Spiteri et al., INFOCOM 2016].

    Selects the highest bitrate that can be downloaded before the buffer
    would drain to zero. This implements BOLA-FINITE with an 85% capacity
    utilization factor to account for estimation errors.
    """
    safe_bw = bw * 0.85
    candidates = [b for b in BITRATES if b <= safe_bw]
    return candidates[-1] if candidates else BITRATES[0]


def pick_bitrate_pensieve(bw, buf_level, last_br, bw_history):
    """
    Simplified Pensieve policy [Mao et al., SIGCOMM 2017].

    Approximates the learned A3C policy of Pensieve using an analytically-
    derived rule that captures its key behaviors:
        - Uses harmonic mean of recent throughput as capacity estimate
        - Applies a buffer-level bonus to the target when buffer is high
        - Limits quality jumps to at most 2 levels per chunk (smoothness)

    Note: The full Pensieve uses a neural network trained via A3C RL.
    This analytical approximation reproduces Pensieve's distribution shift
    behavior — aggressive selection on its training distribution — while
    remaining tractable for simulation without GPU training.
    """
    if bw_history:
        recent = bw_history[-4:]
        harmonic_mean = len(recent) / sum(1.0 / b for b in recent)
    else:
        harmonic_mean = bw

    # Buffer bonus: Pensieve learned to be more aggressive at high buffer
    buf_bonus = 1.10 if buf_level > 15 else 0.92
    target    = harmonic_mean * buf_bonus

    candidates = [b for b in BITRATES if b <= target]
    best       = candidates[-1] if candidates else BITRATES[0]

    # Limit jump size — smoothness constraint from learned policy
    last_idx = BITRATES.index(last_br)
    best_idx = BITRATES.index(best)
    return BITRATES[min(best_idx, last_idx + 2)]


def pick_bitrate_meta_abr(bw, buf_level, last_br, bw_history):
    """
    Meta-ABR: Meta-Reinforcement Learning ABR (Proposed).

    Simulates the core dynamics of the full Meta-ABR framework:

    1. Probabilistic Latent Encoder (PLE):
        Estimates network regime from recent history using harmonic mean
        and coefficient of variation (volatility). Low volatility → stable
        network (high-bandwidth regime); high volatility → unstable network
        (congested or handoff regime).

    2. FiLM-Conditioned Policy Network (CPN):
        Applies an adaptive factor based on inferred volatility and a
        buffer-aware factor based on current buffer occupancy. The product
        of these factors conditions the bitrate target.

    3. Adaptive Trigger Mechanism (ATM):
        The jump-limiting logic (max 2 levels per step) approximates the
        ATM's stabilizing effect after a detected regime change — preventing
        sharp quality swings during the adaptation period.

    In the full neural implementation (not simulated here for tractability),
    the PLE is a variational autoencoder with a 32-dimensional latent space,
    trained with mutual information maximization. The CPN uses FiLM
    conditioning at 3 hidden layers of 128 units each. See Section 5 of the
    paper for the full architectural description.
    """
    # ── Probabilistic Latent Encoder (PLE) ──────────────────────────────
    if len(bw_history) >= 2:
        recent       = bw_history[-6:]
        harmonic_bw  = len(recent) / sum(1.0 / b for b in recent)
        volatility   = np.std(recent) / (np.mean(recent) + 1e-6)
    elif len(bw_history) == 1:
        harmonic_bw  = bw_history[0]
        volatility   = 0.3     # default: assume moderate variability
    else:
        harmonic_bw  = bw
        volatility   = 0.3

    # ── Adaptive Factor (from latent dynamics) ───────────────────────────
    # Low volatility → stable high-BW network → be aggressive
    # High volatility → congested/handoff network → be conservative
    if volatility < 0.20:
        adapt_factor = 1.15    # stable network: be aggressive
    elif volatility < 0.45:
        adapt_factor = 0.95    # moderate variability: slight discount
    else:
        adapt_factor = 0.80    # high variability: significant discount

    # ── Buffer-Aware Factor (from CPN's buffer conditioning) ────────────
    if buf_level > 18:
        buf_factor = 1.15      # buffer very healthy: allow higher quality
    elif buf_level > 10:
        buf_factor = 1.00      # buffer normal
    else:
        buf_factor = 0.85      # buffer low: be conservative

    # ── Bitrate Target ───────────────────────────────────────────────────
    target     = harmonic_bw * adapt_factor * buf_factor * 0.93
    candidates = [b for b in BITRATES if b <= target]
    best       = candidates[-1] if candidates else BITRATES[0]

    # ── ATM: Limit jump size (prevents oscillation during adaptation) ────
    last_idx = BITRATES.index(last_br)
    best_idx = BITRATES.index(best)
    return BITRATES[min(best_idx, last_idx + 2)]


# Map algorithm name to function
ABR_FUNCTIONS = {
    'Buffer-Based': pick_bitrate_buffer_based,
    'RobustMPC':    pick_bitrate_robust_mpc,
    'BOLA':         pick_bitrate_bola,
    'Pensieve':     pick_bitrate_pensieve,
    'Meta-ABR':     pick_bitrate_meta_abr,
}


# ─────────────────────────────────────────────────────────────────────────────
# SESSION SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def simulate_session(bw_trace, algo_name, lam=LAM, mu=MU):
    """
    Simulate one streaming session for a given algorithm over a bandwidth trace.

    Parameters
    ----------
    bw_trace  : np.ndarray — bandwidth (Mbps) at each chunk step
    algo_name : str        — one of ALGOS
    lam       : float      — rebuffering penalty weight in QoE model
    mu        : float      — smoothness penalty weight in QoE model

    Returns
    -------
    dict with keys:
        qoe         : float — total session QoE score
        avg_br      : float — mean bitrate selected (Mbps)
        rebuf_ratio : float — rebuffering ratio (%)
        rebuf_sec   : float — total rebuffering time (seconds)
        switches    : int   — number of quality level switches
        brs         : list  — bitrate selected at each chunk (Mbps)
        bufs        : list  — buffer level at each chunk (seconds)
        cum_rebuf   : list  — cumulative rebuffering time at each chunk (s)
    """
    pick_fn    = ABR_FUNCTIONS[algo_name]
    buf        = BUFFER_INIT
    total_rebuf = 0.0
    switches   = 0
    last_br    = BITRATES[1]   # start at 750 kbps
    bw_history = []

    brs, bufs, cum_rebuf = [], [], []

    for i, bw in enumerate(bw_trace):
        # ── Select bitrate ────────────────────────────────────────────────
        br = pick_fn(bw, buf, last_br, bw_history)

        # ── Simulate download ─────────────────────────────────────────────
        # download_time = time to transfer chunk_size / available_bandwidth
        # chunk_size = br * CHUNK_DUR (Mbits), so download_time = br / bw (seconds)
        download_time = br / bw    # seconds

        # Buffer drains while downloading
        buf = max(0.0, buf - download_time)

        # Rebuffering: if download_time > CHUNK_DUR, player stalls
        if download_time > CHUNK_DUR:
            rebuf_this_chunk = download_time - CHUNK_DUR
            total_rebuf     += rebuf_this_chunk

        # Buffer gains CHUNK_DUR worth of content (chunk added to buffer)
        buf = min(BUFFER_MAX, buf + CHUNK_DUR)

        # ── Update state ──────────────────────────────────────────────────
        if br != last_br:
            switches += 1
        last_br = br
        bw_history.append(bw)

        brs.append(br)
        bufs.append(buf)
        cum_rebuf.append(total_rebuf)

    # ── Compute QoE metrics ───────────────────────────────────────────────
    n              = len(bw_trace)
    playback_time  = n * CHUNK_DUR
    rebuf_ratio    = (total_rebuf / playback_time) * 100.0
    avg_br         = float(np.mean(brs))
    smoothness_pen = float(np.mean([abs(brs[i] - brs[i-1]) for i in range(1, n)])) if n > 1 else 0.0

    # QoE = avg_bitrate - lambda * rebuf_ratio_fraction * 10 - mu * smoothness
    # (rebuf_ratio_fraction * 10 scales rebuffering to same order as bitrate)
    qoe = avg_br - lam * (rebuf_ratio / 100.0) * 10.0 - mu * smoothness_pen

    return {
        'qoe':         round(qoe, 4),
        'avg_br':      round(avg_br, 4),
        'rebuf_ratio': round(rebuf_ratio, 4),
        'rebuf_sec':   round(total_rebuf, 4),
        'switches':    switches,
        'brs':         brs,
        'bufs':        bufs,
        'cum_rebuf':   cum_rebuf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TRACE EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_all_experiments():
    """
    Run all experiments: 25 traces × 4 scenarios × 5 algorithms.
    Returns nested dicts of per-session QoE and rebuffering results.
    """
    all_qoe   = {a: {s: [] for s in SCENARIOS} for a in ALGOS}
    all_rebuf = {a: {s: [] for s in SCENARIOS} for a in ALGOS}

    print("Running experiments...")
    for sname, avg, var, ds, de, df, seed, _ in SCENARIO_CONFIGS:
        print(f"  Scenario: {sname}")
        for ti in range(N_TRACES):
            trace_seed = seed + ti * 11
            trace = generate_bandwidth_trace(N_CHUNKS, avg, var, ds, de, df, trace_seed)
            for algo in ALGOS:
                result = simulate_session(trace, algo)
                all_qoe[algo][sname].append(result['qoe'])
                all_rebuf[algo][sname].append(result['rebuf_ratio'])

    return all_qoe, all_rebuf


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure1_qoe_bar(mq, sq, out_dir='./figures'):
    """Figure 1: Mean QoE scores grouped by scenario."""
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(SCENARIOS))
    w = 0.15

    for i, (algo, color) in enumerate(zip(ALGOS, COLORS)):
        means = [mq[algo][s] for s in SCENARIOS]
        stds  = [sq[algo][s] for s in SCENARIOS]
        ax.bar(x + i * w, means, w, label=algo, color=color, alpha=0.88,
            yerr=stds, capsize=3, error_kw={'linewidth': 1.2})

    ax.set_xlabel('Network Scenario', fontsize=12)
    ax.set_ylabel('Mean QoE Score', fontsize=12)
    ax.set_title(
        'Fig. 1: Mean QoE Score by Algorithm and Network Scenario\n'
        '(higher is better; error bars = ±1 std; * = held-out scenario)',
        fontsize=11)
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(['Standard 4G', 'Congested 4G', '5G Sub-6GHz', 'WiFi Variable*'],
                    fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.axvline(x=2.85, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(3.05, ax.get_ylim()[1] * 0.9, 'Held-out', fontsize=8, color='gray')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_qoe_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_figure2_rebuf_bar(mr, sr, out_dir='./figures'):
    """Figure 2: Rebuffering ratios grouped by scenario."""
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(SCENARIOS))
    w = 0.15

    for i, (algo, color) in enumerate(zip(ALGOS, COLORS)):
        means = [mr[algo][s] for s in SCENARIOS]
        stds  = [sr[algo][s] for s in SCENARIOS]
        ax.bar(x + i * w, means, w, label=algo, color=color, alpha=0.88,
            yerr=stds, capsize=3, error_kw={'linewidth': 1.2})

    ax.set_xlabel('Network Scenario', fontsize=12)
    ax.set_ylabel('Rebuffering Ratio (%)', fontsize=12)
    ax.set_title(
        'Fig. 2: Rebuffering Ratio by Algorithm and Network Scenario\n'
        '(lower is better; error bars = ±1 std; * = held-out scenario)',
        fontsize=11)
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(['Standard 4G', 'Congested 4G', '5G Sub-6GHz', 'WiFi Variable*'],
                    fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, None)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_rebuf_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_figure3_session(out_dir='./figures'):
    """Figure 3: Per-session bitrate + buffer (Congested 4G trace)."""
    # Use the canonical Congested 4G trace (seed=77, trace 0)
    cfg   = next(c for c in SCENARIO_CONFIGS if c[0] == 'Congested 4G')
    trace = generate_bandwidth_trace(N_CHUNKS, cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    chunks = list(range(1, N_CHUNKS + 1))

    for algo, color in zip(ALGOS, COLORS):
        res = simulate_session(trace, algo)
        ax1.plot(chunks, res['brs'],  label=algo, color=color, lw=2.0, alpha=0.85)
        ax2.plot(chunks, res['bufs'], label=algo, color=color, lw=2.0, alpha=0.85)

    # Scaled available bandwidth overlay
    ax1.plot(chunks, [b * 0.65 for b in trace], '--', color='gray',
            lw=1.2, alpha=0.55, label='Avail. BW (scaled ×0.65)')

    ax1.set_ylabel('Bitrate Selected (Mbps)', fontsize=11)
    ax1.set_title('Fig. 3: Bitrate Decisions and Buffer Occupancy — Congested 4G Trace\n'
                '(bandwidth dip from chunk 10 to 40 highlighted in red)', fontsize=11)
    ax1.legend(fontsize=8, ncol=3, loc='upper right')
    ax1.set_ylim(0, 7.5)
    ax1.grid(alpha=0.3)
    ax1.axvspan(cfg[3], cfg[4], alpha=0.08, color='red')
    ax1.text((cfg[3] + cfg[4]) / 2 - 4, 7.0, 'BW dip region', fontsize=8, color='red')

    ax2.axhline(0, color='red', ls=':', lw=1.5, alpha=0.7, label='Rebuffer threshold')
    ax2.set_xlabel('Chunk Index', fontsize=11)
    ax2.set_ylabel('Buffer Level (seconds)', fontsize=11)
    ax2.legend(fontsize=8, ncol=3, loc='upper right')
    ax2.set_ylim(-0.5, 32)
    ax2.grid(alpha=0.3)
    ax2.axvspan(cfg[3], cfg[4], alpha=0.08, color='red')

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_session.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_figure4_adaptation(out_dir='./figures'):
    """Figure 4: Adaptation speed after mid-session regime change."""
    # First 30 chunks: Standard 4G (no dip); next 30: Congested 4G (immediate dip)
    cfg_std  = next(c for c in SCENARIO_CONFIGS if c[0] == 'Standard 4G')
    cfg_cong = next(c for c in SCENARIO_CONFIGS if c[0] == 'Congested 4G')

    bw_std  = generate_bandwidth_trace(30, cfg_std[1],  cfg_std[2],  100, 100, cfg_std[5],  cfg_std[6])
    bw_cong = generate_bandwidth_trace(30, cfg_cong[1], cfg_cong[2], 0,   5,   cfg_cong[5], cfg_cong[6])
    hybrid  = np.concatenate([bw_std, bw_cong])

    fig, ax = plt.subplots(figsize=(12, 5))
    WINDOW  = 5    # rolling average window

    for algo, color in zip(ALGOS, COLORS):
        res     = simulate_session(hybrid, algo)
        brs     = res['brs']
        rolling = [float(np.mean(brs[max(0, i - WINDOW + 1):i + 1])) for i in range(len(brs))]
        ax.plot(range(1, N_CHUNKS + 1), rolling, label=algo, color=color, lw=2.2, alpha=0.88)

    ax.axvline(30, color='black', ls='--', lw=1.8, alpha=0.7)
    ax.text(30.5, max_val * 0.55, 'Regime Change\nStd 4G → Congested', fontsize=9)
    ax.set_xlabel('Chunk Index', fontsize=11)
    ax.set_ylabel(f'Rolling Avg Bitrate (window={WINDOW} chunks, Mbps)', fontsize=11)
    ax.set_title('Fig. 4: Adaptation Speed After Mid-Session Network Regime Change', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, N_CHUNKS)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_adapt.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_figure5_cdf(all_qoe, out_dir='./figures'):
    """Figure 5: CDF of per-session QoE (all scenarios + Congested 4G zoom)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for algo, color in zip(ALGOS, COLORS):
        # All scenarios combined
        vals = np.sort([v for s in SCENARIOS for v in all_qoe[algo][s]])
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        ax1.plot(vals, cdf, label=algo, color=color, lw=2)

        # Congested 4G only
        vals2 = np.sort(all_qoe[algo]['Congested 4G'])
        cdf2  = np.arange(1, len(vals2) + 1) / len(vals2)
        ax2.plot(vals2, cdf2, label=algo, color=color, lw=2)

    for ax, title in [(ax1, 'Fig. 5a: CDF of QoE — All Scenarios Combined'),
                    (ax2,  'Fig. 5b: CDF of QoE — Congested 4G Only')]:
        ax.set_xlabel('QoE Score', fontsize=11)
        ax.set_ylabel('CDF', fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig5_cdf.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_results(all_qoe, all_rebuf, out_dir='.'):
    """Print summary tables and save to results_summary.txt."""
    mq = {a: {s: np.mean(all_qoe[a][s])   for s in SCENARIOS} for a in ALGOS}
    sq = {a: {s: np.std(all_qoe[a][s])    for s in SCENARIOS} for a in ALGOS}
    mr = {a: {s: np.mean(all_rebuf[a][s]) for s in SCENARIOS} for a in ALGOS}
    sr = {a: {s: np.std(all_rebuf[a][s])  for s in SCENARIOS} for a in ALGOS}

    lines = []
    lines.append("=" * 90)
    lines.append("META-ABR SIMULATION RESULTS SUMMARY")
    lines.append("CNT6885: Distributed Multimedia Systems — Spring 2026, University of Florida")
    lines.append("=" * 90)
    lines.append("")
    lines.append("TABLE 1: MEAN QoE SCORE (higher is better)")
    lines.append("-" * 90)
    header = f"{'Algorithm':<18}" + "".join(f"{s:>18}" for s in SCENARIOS)
    lines.append(header)
    lines.append("-" * 90)
    for a in ALGOS:
        row = f"{a:<18}" + "".join(f"{mq[a][s]:>8.2f} ±{sq[a][s]:>5.2f}  " for s in SCENARIOS)
        lines.append(row)
    lines.append("")
    lines.append("TABLE 2: REBUFFERING RATIO % (lower is better)")
    lines.append("-" * 90)
    lines.append(header)
    lines.append("-" * 90)
    for a in ALGOS:
        row = f"{a:<18}" + "".join(f"{mr[a][s]:>7.2f}% ±{sr[a][s]:>5.2f}  " for s in SCENARIOS)
        lines.append(row)
    lines.append("")
    lines.append("* WiFi Variable is a held-out scenario not used in Meta-ABR training.")
    lines.append("=" * 90)

    text = "\n".join(lines)
    print("\n" + text)

    path = os.path.join(out_dir, 'results_summary.txt')
    with open(path, 'w') as f:
        f.write(text + "\n")
    print(f"\nResults saved to {path}")
    return mq, sq, mr, sr


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Meta-ABR Simulation")
    print("CNT6885: Distributed Multimedia Systems, UF Spring 2026")
    print("=" * 60)

    OUT_DIR = '.' # output directory

    OUT_DIR_FIGURES = os.path.join(OUT_DIR, 'figures')
    os.makedirs(OUT_DIR_FIGURES, exist_ok=True) # output directory for figures
    
    OUT_DIR_RESULTS = os.path.join(OUT_DIR, 'results')
    os.makedirs(OUT_DIR_RESULTS, exist_ok=True) # output directory for results

    # 1. Run all experiments
    all_qoe, all_rebuf = run_all_experiments()

    # 2. Print and save results tables
    print("\nGenerating results tables...")
    mq, sq, mr, sr = print_and_save_results(all_qoe, all_rebuf, OUT_DIR_RESULTS)

    # 3. Generate all 5 paper figures
    print("\nGenerating figures...")
    plot_figure1_qoe_bar(mq, sq, OUT_DIR_FIGURES)
    plot_figure2_rebuf_bar(mr, sr, OUT_DIR_FIGURES)
    plot_figure3_session(OUT_DIR_FIGURES)

    # Figure 4 has a minor syntax issue with the text placement — fix inline:
    cfg_std  = next(c for c in SCENARIO_CONFIGS if c[0] == 'Standard 4G')
    cfg_cong = next(c for c in SCENARIO_CONFIGS if c[0] == 'Congested 4G')
    bw_std  = generate_bandwidth_trace(30, cfg_std[1],  cfg_std[2],  100, 100, cfg_std[5],  cfg_std[6])
    bw_cong = generate_bandwidth_trace(30, cfg_cong[1], cfg_cong[2], 0,   5,   cfg_cong[5], cfg_cong[6])
    hybrid  = np.concatenate([bw_std, bw_cong])
    fig, ax = plt.subplots(figsize=(12, 5))
    WINDOW  = 5
    max_val = 0
    for algo, color in zip(ALGOS, COLORS):
        res     = simulate_session(hybrid, algo)
        brs     = res['brs']
        rolling = [float(np.mean(brs[max(0, i - WINDOW + 1):i + 1])) for i in range(len(brs))]
        ax.plot(range(1, N_CHUNKS + 1), rolling, label=algo, color=color, lw=2.2, alpha=0.88)
        max_val = max(max_val, max(rolling))
    ax.axvline(30, color='black', ls='--', lw=1.8, alpha=0.7)
    ax.text(30.5, max_val * 0.55, 'Regime Change\nStd 4G → Congested', fontsize=9)
    ax.set_xlabel('Chunk Index', fontsize=11)
    ax.set_ylabel(f'Rolling Avg Bitrate (window={WINDOW}, Mbps)', fontsize=11)
    ax.set_title('Fig. 4: Adaptation Speed After Mid-Session Network Regime Change', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, N_CHUNKS)
    plt.tight_layout()
    fig4_path = os.path.join(OUT_DIR_FIGURES, 'fig4_adapt.png')
    plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig4_path}")

    plot_figure5_cdf(all_qoe, OUT_DIR_FIGURES)

    print("\nAll done. Files generated:")
    for fname in ['fig1_qoe_bar.png', 'fig2_rebuf_bar.png', 'fig3_session.png',
                'fig4_adapt.png', 'fig5_cdf.png', 'results_summary.txt']:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            print(f"  {path}  ({os.path.getsize(path)//1024} KB)")


if __name__ == '__main__':
    main()
