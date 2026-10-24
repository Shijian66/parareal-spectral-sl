"""Reproduce Figures 3, 4 (and optionally 6) of the paper.

Usage
-----
    python scripts/plot_figures.py            # Figures 3 and 4
    python scripts/plot_figures.py --fig6     # also Figure 6 (variable speed)
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
try:                                     # use the installed package if present
    import propagator as sp
except ImportError:                      # otherwise fall back to the source tree
    sys.path.insert(0, str(ROOT / "src"))
    import propagator as sp  # noqa: E402

# styling shared by all figures (serif, thin lines, no top/right frame)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})

BLUE, RED, GREEN, GREY = "#1f4fd8", "#d62728", "#2ca02c", "0.55"
TOL = 1e-6            # tolerance used in the paper
EXACT_FLOW = 1e-14    # round-off floor of a perfectly isometric propagator


def _decorate(ax, dT, subtitle):
    ax.axhline(TOL, color=GREY, lw=0.7, ls=(0, (1, 2)))
    ax.axhline(EXACT_FLOW, color="0.15", lw=0.9, ls=(0, (4, 2, 1, 2)),
               label="spectral-shift limit (exact flow)")
    ax.set_yscale("log")
    ax.set_ylim(3e-16, 5e-1)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Relative error")
    ax.set_title(r"$\Delta T = %.3f$%s" % (dT, subtitle), fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              fontsize=6.8, handlelength=1.8, columnspacing=1.0)
    ax.text(ax.get_xlim()[1], TOL * 1.6, r"$10^{-6}$", ha="right", va="bottom",
            fontsize=8, color=GREY)


def figure3(grid, dT=0.02, dt=1e-3):
    """Fig. 3: s = 2 (integer shift), implicit Euler vs spectral semi-Lagrangian."""
    U0 = grid.initial_condition()
    err_ie = grid.parareal(U0, dT, dt, grid.coarse_implicit_euler, max_iter=12, tol=0.0)
    err_sl = grid.parareal(U0, dT, dt, grid.coarse_improved, max_iter=12, tol=0.0)

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ie = _truncate(err_ie)
    sl = _truncate_at_min(err_sl)
    ax.semilogy(np.arange(1, len(ie) + 1), ie, "o-",
                color=BLUE, ms=3.4, lw=1.0, label="implicit Euler")
    ax.semilogy(np.arange(1, len(sl) + 1), sl, "s-",
                color=RED, ms=3.4, lw=1.0, label="spectral semi-Lagrangian")
    _decorate(ax, dT, r"  ($s = 2$, integer shift)")
    fig.tight_layout(pad=0.3)
    fig.savefig("Figure3.pdf", dpi=300)
    fig.savefig("Figure3.png", dpi=300)
    plt.close(fig)
    print("Figure 3: implicit Euler %d iterations, semi-Lagrangian %d iterations"
          % (_count_to(err_ie, TOL), _count_to(err_sl, TOL)))
    return err_ie, err_sl


def figure4(grid, dT=0.015, dt=1e-3):
    """Fig. 4: s = 1.5 (fractional shift), plus the cubic-interpolation variant."""
    U0 = grid.initial_condition()
    err_ie = grid.parareal(U0, dT, dt, grid.coarse_implicit_euler, max_iter=12, tol=0.0)
    err_sl = grid.parareal(U0, dT, dt, grid.coarse_improved, max_iter=12, tol=0.0)
    err_cu = grid.parareal(U0, dT, dt,
                           lambda U, d: grid.coarse_improved(U, d, use_cubic=True),
                           max_iter=12, tol=0.0)

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.semilogy(np.arange(1, len(_truncate(err_ie)) + 1), _truncate(err_ie), "o-",
                color=BLUE, ms=3.4, lw=1.0, label="implicit Euler")
    ax.semilogy(np.arange(1, len(_truncate(err_sl)) + 1), _truncate(err_sl), "s-",
                color=RED, ms=3.4, lw=1.0, label="spectral semi-Lagrangian")
    ax.semilogy(np.arange(1, len(_truncate(err_cu)) + 1), _truncate(err_cu), "D-",
                color=GREEN, ms=3.0, lw=0.9, label="cubic interpolation")
    _decorate(ax, dT, r"  ($s = 1.5$, fractional shift)")
    fig.tight_layout(pad=0.3)
    fig.savefig("Figure4.pdf", dpi=300)
    fig.savefig("Figure4.png", dpi=300)
    plt.close(fig)
    print("Figure 4: implicit Euler %d, linear %d, cubic %d iterations"
          % (_count_to(err_ie, TOL), _count_to(err_sl, TOL), _count_to(err_cu, TOL)))
    return err_ie, err_sl, err_cu


def figure6(dT=0.02, dt=1e-3):
    """Fig. 6 (optional): variable speed in the travel-time variable.

    A 40% variation of the wave speed, equation posed on the travel-time grid,
    so that the coarse propagators of ``TravelTimeGrid`` apply unchanged.
    The exponential integrator is included because the paper compares against
    it; it needs one eigendecomposition of the operator.
    """
    tt = sp.TravelTimeGrid(N=100, c0=1.0, amplitude=0.4)
    u0 = (np.sin(2 * np.pi * tt.tau / tt.L)
          + 0.5 * np.sin(6 * np.pi * tt.tau / tt.L))
    U0 = np.concatenate([u0, np.zeros(tt.N)])

    err_ie = tt.parareal(U0, dT, tt.coarse_implicit_euler, max_iter=40, tol=0.0)
    err_sl = tt.parareal(U0, dT, tt.coarse_improved, max_iter=40, tol=0.0)
    err_ex = tt.parareal(U0, dT, tt.exact_flow, max_iter=40, tol=0.0)

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    k_show = 8                                  # window used in the paper
    ax.semilogy(np.arange(1, k_show + 1), err_ie[:k_show], "o-", color=BLUE,
                ms=3.2, lw=1.0, label="implicit Euler")
    sl_show = _truncate_at_min(err_sl, extra=0)[:k_show]
    ax.semilogy(np.arange(1, len(sl_show) + 1), sl_show, "s-", color=RED,
                ms=3.2, lw=1.0, label="spectral semi-Lagrangian")
    ax.semilogy(np.arange(1, k_show + 1), err_ex[:k_show], "^-", color=GREEN,
                ms=3.2, lw=1.0, label="exponential integrator")
    _decorate(ax, dT, r"  ($c$ varies by 40%, travel-time grid)")
    fig.tight_layout(pad=0.3)
    fig.savefig("Figure6.pdf", dpi=300)
    fig.savefig("Figure6.png", dpi=300)
    plt.close(fig)
    print("Figure 6: implicit Euler %d, semi-Lagrangian %d, exponential %d iterations"
          % (_count_to(err_ie, TOL), _count_to(err_sl, TOL), _count_to(err_ex, TOL)))
    return err_ie, err_sl, err_ex


def _count_to(errors, tol):
    """First k for which the error falls below tol (len(errors) if never)."""
    for k, err in enumerate(errors, start=1):
        if err < tol:
            return k
    return len(errors)


def _truncate_at_min(errors, extra=1):
    """Cut a history just after its minimum: beyond that round-off dominates."""
    e = np.asarray(errors)
    return e[:min(len(e), int(np.argmin(e)) + 1 + extra)]


def _truncate(errors, floor=3e-14, extra=1):
    """Cut a history one point after it first reaches the round-off floor."""
    hit = np.nonzero(np.asarray(errors) <= floor)[0]
    if hit.size == 0:
        return np.asarray(errors)
    return np.asarray(errors)[:min(len(errors), hit[0] + 1 + extra)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig6", action="store_true",
                        help="also produce Figure 6 (variable speed)")
    args = parser.parse_args()

    grid = sp.Grid(N=100, L=1.0, c=1.0)
    figure3(grid)
    figure4(grid)
    if args.fig6:
        figure6()


if __name__ == "__main__":
    main()
