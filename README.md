# A Spectral Semi-Lagrangian Coarse Propagator for Parareal

[![CI](https://github.com/Shijian66/parareal-spectral-sl/actions/workflows/ci.yml/badge.svg)](https://github.com/Shijian66/parareal-spectral-sl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

Reference implementation and figure-generation code for the manuscript

> Shijian Wang, *A Spectral Semi-Lagrangian Coarse Propagator for the Parareal
> Solution of the One-Dimensional Wave Equation*.

The repository reproduces Figures 3, 4 and 6 of the paper and the iteration
counts reported there: five to one at `ΔT = 0.02`, five to three at
`ΔT = 0.015`, and thirty-one to three for the variable-speed problem.

## Idea in one paragraph

Parareal converges at a rate set by its coarse propagator. For the wave
equation the standard choice, implicit Euler, is strongly dissipative: at the
grid Nyquist wavenumber it attenuates a mode by order `10^-40` over a unit time
interval, so the coarse prediction carries no information about the fast part of
the solution. This code replaces implicit Euler by a propagator built from
d'Alembert's formula. It advances along the two characteristics `x ± cΔT` and
evaluates the characteristic interval integral

```
(S v)(x) = ∫_{x-cΔT}^{x+cΔT} v(s) ds
```

**exactly in Fourier space**, where its symbol is the real function

```
Ŝ(k) = 2 sin(k c ΔT) / k ,      Ŝ(0) = 2 c ΔT .
```

A pair of FFTs applies it at `O(N log N)` cost, so no quadrature error is
introduced no matter how many cells the interval spans. The only remaining
approximation is the interpolation of the two characteristic feet, which is
second order for linear interpolation and fourth order for the cubic variant
(Remark 4 in the paper).

## Repository layout

```
.
├── .github/workflows/ci.yml   # tests on Linux/macOS/Windows, Python 3.9-3.12
├── CITATION.cff               # how to cite the code
├── LICENSE                    # MIT
├── README.md
├── pyproject.toml             # packaging and pytest configuration
├── requirements.txt
├── src/
│   └── propagator.py          # operators, fine solver, coarse propagators, Parareal
├── scripts/
│   └── plot_figures.py        # Figures 3 and 4 (and Figure 6 with --fig6)
└── tests/
    ├── conftest.py
    └── test_propagator.py     # 12 tests, including the paper's iteration counts
```

## Installation

```bash
git clone https://github.com/Shijian66/parareal-spectral-sl.git
cd parareal-spectral-sl

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pip install -e .                   # optional: makes `import propagator` global
```

Python 3.9 or newer is required. The only runtime dependencies are NumPy and
Matplotlib.

## Reproducing the figures

From the repository root:

```bash
python scripts/plot_figures.py
```

This writes `Figure3.pdf` and `Figure4.pdf` (vector, 300 dpi) together with PNG
copies for quick inspection:

* **Figure 3** — `ΔT = 0.02` (`s = 2`, integer shift): implicit Euler versus the
  spectral semi-Lagrangian propagator.
* **Figure 4** — `ΔT = 0.015` (`s = 1.5`, fractional shift): the same two curves
  plus the cubic-interpolation variant (`use_cubic=True`).

Both carry a dash-dotted reference line at `10^-14`, the round-off floor of a
perfectly isometric propagator, and a dotted line at the tolerance `10^-6`.

The optional variable-speed figure is produced with

```bash
python scripts/plot_figures.py --fig6
```

which adds `Figure6.pdf`: the same comparison in a medium whose wave speed
varies by 40%, posed in the travel-time variable.

## Expected output

```
$ python scripts/plot_figures.py
Figure 3: implicit Euler 5 iterations, semi-Lagrangian 1 iterations
Figure 4: implicit Euler 5, linear 3, cubic 1 iterations
```

The iteration count is the first `k` for which the relative error falls below
`10^-6`. The curves themselves are correct to the last digit shown: the
semi-Lagrangian iteration reaches `2 × 10^-14` at `ΔT = 0.015`, that is, the
exact flow to round-off.

## Tests

```bash
pip install pytest
pytest -q
```

Twelve tests check the building blocks against closed-form values (symbol of the
characteristic integral, order of the interpolants, exactness at integer shift)
and then assert the iteration counts quoted in the paper. The whole suite runs in
about a minute.

## Using the library

```python
import propagator as p

grid = p.Grid(N=100, L=1.0, c=1.0)          # periodic, constant wave speed
U0 = grid.initial_condition()               # u0 = sin(2 pi x / L), v0 = 0

errors = grid.parareal(U0, dT=0.015, dt=1e-3,
                       coarse=grid.coarse_improved)         # linear interpolation
errors_cubic = grid.parareal(U0, dT=0.015, dt=1e-3,
                             coarse=lambda U, d:
                                 grid.coarse_improved(U, d, use_cubic=True))
```

| method | purpose |
|---|---|
| `Grid.step_rk4`, `Grid.fine_solver` | fourth-order Runge–Kutta fine propagator |
| `Grid.spectral_integral_v` | exact characteristic integral via FFT |
| `Grid.interp`, `Grid.interp_cubic` | linear / four-point Lagrange interpolation of the feet |
| `Grid.coarse_implicit_euler` | the standard coarse propagator |
| `Grid.coarse_improved(..., use_cubic=False)` | the spectral semi-Lagrangian propagator |
| `Grid.parareal` | Parareal driver, returns the relative error per iteration |
| `TravelTimeGrid` | variable speed in the travel-time variable (Section 5.5) |

Module-level wrappers with the same names operate on a default
`Grid(N=100, L=1.0, c=1.0)`.

## Numerical settings

| symbol | value | meaning |
|---|---|---|
| `N` | 100 | grid points, `Δx = 0.01` |
| `L`, `c` | 1.0, 1.0 | interval length and wave speed |
| `dt` | `10^-3` | fine step, `c dt/Δx = 0.1 < 1` (CFL) |
| `ΔT` | `0.02`, `0.015` | coarse step, `s = cΔT/Δx = 2` and `1.5` |
| `T` | 1.0 | final time |

The fine propagator is RK4 applied to the first-order system `U_t = A U` with
`U = (u, v)`, `A = [[0, I], [c² D₂, 0]]`, and centred periodic differences `D₁`,
`D₂`.

## Troubleshooting

**Pylance says `propagator` cannot be resolved.** The module lives in `src/`,
which is not on the default import path. Either install the package
(`pip install -e .`) or open the folder in VS Code with the bundled
`.vscode/settings.json`, which adds `./src` to `python.analysis.extraPaths`.
Check the interpreter with

```bash
python -c "import propagator; print(propagator.__file__)"
```

**Pylance says `pytest` cannot be resolved, or `pytest` is not recognised.**
It is not installed in the interpreter VS Code is using:

```bash
python -m pip install pytest
python -m pytest -q            # note: -m, so the same interpreter is used
```

**`pytest` runs but collects 0 tests.** Test files must be named `test_*.py` or
`*_test.py`; a file called `pararealtest.py` is skipped by default. Keep the
suite at `tests/test_propagator.py`, or pass the file explicitly
(`python -m pytest pararealtest.py`).

**Nothing works and you want to know whether the code or the tooling is at
fault.** Run the runner-free check:

```bash
python scripts/selftest.py          # a few seconds
python scripts/selftest.py --full   # also the iteration counts
```

It imports `propagator` directly and prints the path it came from, so a failure
there is a code or path problem, not a pytest problem.

## Citation

If you use this code, please cite the paper above; a machine-readable entry is
provided in [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
