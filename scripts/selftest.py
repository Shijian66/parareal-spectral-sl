"""Self-contained check that does not need pytest.

    python scripts/selftest.py          # fast checks, a few seconds
    python scripts/selftest.py --full   # also the iteration counts of the paper

Use this if pytest is not installed yet, or to find out whether a failure comes
from the code or from the test runner.
"""

import argparse
import pathlib
import sys
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
try:
    import propagator as sp
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))
    import propagator as sp

PASSED = []
FAILED = []


def check(name):
    """Decorator: run the function and record pass/fail."""
    def wrap(fn):
        try:
            fn()
        except Exception as exc:                  # noqa: BLE001 - report everything
            FAILED.append((name, exc))
            print("FAIL  %s\n      %s" % (name, exc))
            traceback.print_exc(limit=2)
        else:
            PASSED.append(name)
            print("ok    %s" % name)
        return fn
    return wrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="also check the iteration counts (about a minute)")
    args = parser.parse_args()

    print("propagator imported from:", sp.__file__)
    grid = sp.Grid(N=100, L=1.0, c=1.0)

    @check("operator shapes")
    def _():
        n = grid.N
        assert grid.A.shape == (2 * n, 2 * n)
        assert np.array_equal(grid.A[:n, n:], np.eye(n))
        assert np.array_equal(grid.A[n:, :n], grid.D2)

    @check("spectral integral of a constant equals 2 c dT")
    def _():
        out = grid.spectral_integral_v(np.ones(grid.N), 0.015)
        assert np.allclose(out, 2.0 * grid.c * 0.015, atol=1e-12)

    @check("spectral integral matches 2 sin(k c dT)/k")
    def _():
        k = 2.0 * np.pi / grid.L
        v = np.sin(k * grid.x)
        exact = 2.0 * np.sin(k * grid.c * 0.015) / k * v
        assert np.allclose(grid.spectral_integral_v(v, 0.015), exact, atol=1e-13)

    @check("interpolants are exact on the grid")
    def _():
        values = np.sin(2 * np.pi * grid.x / grid.L)
        pos = np.arange(grid.N, dtype=float)
        assert np.allclose(grid.interp(pos, values), values, atol=1e-14)
        assert np.allclose(grid.interp_cubic(pos, values), values, atol=1e-14)

    @check("integer shift reproduces the exact flow")
    def _():
        u0 = np.sin(2 * np.pi * grid.x / grid.L)
        U0 = np.concatenate([u0, np.zeros(grid.N)])
        out = grid.coarse_improved(U0, 0.02)
        exact = np.cos(2 * np.pi * grid.c * 0.02 / grid.L) * u0
        assert np.linalg.norm(out[:grid.N] - exact) / np.linalg.norm(exact) < 1e-12

    if args.full:
        @check("Figure 3 counts: implicit Euler 5, semi-Lagrangian 1")
        def _():
            U0 = grid.initial_condition()
            ie = grid.parareal(U0, 0.02, 1e-3, grid.coarse_implicit_euler, max_iter=8)
            sl = grid.parareal(U0, 0.02, 1e-3, grid.coarse_improved, max_iter=8)
            assert _count_to(ie) == 5, ie
            assert _count_to(sl) == 1, sl

        @check("Figure 4 counts: 5 / 3 / 1 (linear, cubic)")
        def _():
            U0 = grid.initial_condition()
            ie = grid.parareal(U0, 0.015, 1e-3, grid.coarse_implicit_euler, max_iter=8)
            sl = grid.parareal(U0, 0.015, 1e-3, grid.coarse_improved, max_iter=8)
            cu = grid.parareal(
                U0, 0.015, 1e-3,
                lambda U, d: grid.coarse_improved(U, d, use_cubic=True), max_iter=8)
            assert (_count_to(ie), _count_to(sl), _count_to(cu)) == (5, 3, 1)

    print()
    print("%d passed, %d failed" % (len(PASSED), len(FAILED)))
    return 1 if FAILED else 0


def _count_to(errors, tol=1e-6):
    for k, err in enumerate(errors, start=1):
        if err < tol:
            return k
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
