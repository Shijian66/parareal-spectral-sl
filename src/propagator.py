"""Core algorithms for a spectral semi-Lagrangian Parareal coarse propagator.

The module implements, for the one-dimensional wave equation written as the
first-order system

        u_t = v,          v_t = c^2 u_xx ,

a coarse propagator that is built from d'Alembert's formula instead of from a
generic ODE integrator.  The characteristic interval integral

        (S v)(x) = int_{x-c dT}^{x+c dT} v(s) ds

is evaluated exactly in Fourier space, where its symbol is the real function
S_hat(k) = 2 sin(k c dT) / k, with the limit S_hat(0) = 2 c dT.  The only
remaining approximation is the interpolation of the two characteristic feet,
which may be linear (second order) or cubic (fourth order).

Public API
----------
Grid                     periodic constant-speed problem, N grid points
TravelTimeGrid           variable speed handled in the travel-time variable
Grid.coarse_improved     spectral semi-Lagrangian coarse step
Grid.coarse_implicit_euler  the standard implicit-Euler coarse step
Grid.parareal            Parareal driver, returns the error history

Module-level functions (step_rk4, fine_solver, spectral_integral_v, interp,
interp_cubic, coarse_improved, coarse_implicit_euler, parareal) are thin
wrappers around a default Grid(N=100, L=1.0, c=1.0), so that the smallest
possible script is

        import propagator as p
        p.parareal(p.initial_condition(), 0.015, 1e-3, p.coarse_improved)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Grid", "TravelTimeGrid",
    "step_rk4", "fine_solver", "spectral_integral_v", "interp", "interp_cubic",
    "coarse_improved", "coarse_implicit_euler", "parareal",
    "initial_condition",
]

FINE_STEP = 1e-3


# ---------------------------------------------------------------------------
# periodic, constant wave speed
# ---------------------------------------------------------------------------
class Grid:
    """Periodic grid, centred differences, and the two coarse propagators."""

    def __init__(self, N: int = 100, L: float = 1.0, c: float = 1.0):
        self.N = int(N)
        self.L = float(L)
        self.c = float(c)
        self.dx = self.L / self.N
        self.x = np.arange(self.N) * self.dx
        self.k = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.dx)

        self.D1 = np.zeros((self.N, self.N))       # centred du/dx, periodic
        self.D2 = np.zeros((self.N, self.N))       # centred d2u/dx2, periodic
        for i in range(self.N):
            self.D1[i, (i - 1) % self.N] = -1.0 / (2.0 * self.dx)
            self.D1[i, (i + 1) % self.N] = +1.0 / (2.0 * self.dx)
            self.D2[i, (i - 1) % self.N] = 1.0 / self.dx ** 2
            self.D2[i, i] = -2.0 / self.dx ** 2
            self.D2[i, (i + 1) % self.N] = 1.0 / self.dx ** 2

        # first-order system U_t = A U with U = (u, v)
        self.A = np.zeros((2 * self.N, 2 * self.N))
        self.A[:self.N, self.N:] = np.eye(self.N)
        self.A[self.N:, :self.N] = self.c ** 2 * self.D2

    # -- fine propagator ---------------------------------------------------
    def step_rk4(self, U: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.A @ U
        k2 = self.A @ (U + 0.5 * dt * k1)
        k3 = self.A @ (U + 0.5 * dt * k2)
        k4 = self.A @ (U + dt * k3)
        return U + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def fine_solver(self, U: np.ndarray, dT: float, dt: float = FINE_STEP) -> np.ndarray:
        """m = dT/dt sub-steps of RK4; the accurate reference propagator."""
        U = np.copy(U)
        for _ in range(int(round(dT / dt))):
            U = self.step_rk4(U, dt)
        return U

    # -- spectral characteristic integral ----------------------------------
    def spectral_integral_v(self, v: np.ndarray, dT: float) -> np.ndarray:
        """int_{x-c dT}^{x+c dT} v(s) ds, exact in Fourier space."""
        factor = np.zeros_like(self.k)
        nz = np.abs(self.k) > 1e-12
        factor[nz] = 2.0 * np.sin(self.k[nz] * self.c * dT) / self.k[nz]
        factor[~nz] = 2.0 * self.c * dT                    # limit k -> 0
        return np.fft.ifft(factor * np.fft.fft(v)).real

    # -- interpolation of the characteristic feet --------------------------
    def interp(self, xp: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Periodic linear interpolation at fractional grid positions xp."""
        i0 = np.floor(xp).astype(int) % self.N
        i1 = (i0 + 1) % self.N
        w = xp - np.floor(xp)
        return (1.0 - w) * values[i0] + w * values[i1]

    def interp_cubic(self, xp: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Periodic four-point Lagrange interpolation, O(h^4), local stencil."""
        i = np.floor(xp).astype(int)
        w = xp - i
        f_m1 = values[(i - 1) % self.N]
        f_0 = values[i % self.N]
        f_1 = values[(i + 1) % self.N]
        f_2 = values[(i + 2) % self.N]
        l_m1 = -w * (w - 1.0) * (w - 2.0) / 6.0
        l_0 = (w + 1.0) * (w - 1.0) * (w - 2.0) / 2.0
        l_1 = -(w + 1.0) * w * (w - 2.0) / 2.0
        l_2 = (w + 1.0) * w * (w - 1.0) / 6.0
        return l_m1 * f_m1 + l_0 * f_0 + l_1 * f_1 + l_2 * f_2

    # -- coarse propagators ------------------------------------------------
    def coarse_implicit_euler(self, U: np.ndarray, dT: float) -> np.ndarray:
        """(I - dT A) U^{n+1} = U^n; the standard choice."""
        return np.linalg.solve(np.eye(2 * self.N) - dT * self.A, U)

    def coarse_improved(self, U: np.ndarray, dT: float,
                        use_cubic: bool = False) -> np.ndarray:
        """One spectral semi-Lagrangian step (d'Alembert + exact integral).

        Parameters
        ----------
        use_cubic : bool
            Cubic interpolation of the characteristic feet (Remark 4) instead
            of the default linear interpolation.
        """
        N = self.N
        u, v = U[:N], U[N:]
        s = self.c * dT / self.dx
        x = np.arange(N, dtype=float)
        xl, xr = (x - s) % N, (x + s) % N            # characteristic feet
        J = self.interp_cubic if use_cubic else self.interp

        u_l, u_r = J(xl, u), J(xr, u)
        v_l, v_r = J(xl, v), J(xr, v)
        ux = self.D1 @ u
        i_v = self.spectral_integral_v(v, dT)         # O(N log N), no quadrature

        u_new = 0.5 * (u_l + u_r) + 0.5 * i_v / self.c
        v_new = 0.5 * (v_l + v_r) + 0.5 * self.c * (J(xr, ux) - J(xl, ux))
        return np.concatenate([u_new, v_new])

    # -- Parareal ----------------------------------------------------------
    def parareal(self, U0: np.ndarray, dT: float, dt: float, coarse,
                 max_iter: int = 15, tol: float = 1e-6,
                 T_final: float = 1.0) -> np.ndarray:
        """Run Parareal and return the relative error after each iteration."""
        N = self.N
        n_steps = int(round(T_final / dT))

        U = np.zeros((2 * N, n_steps + 1))            # coarse prediction
        U[:, 0] = U0
        for n in range(n_steps):
            U[:, n + 1] = coarse(U[:, n], dT)

        ref = np.zeros((2 * N, n_steps + 1))          # fine reference
        ref[:, 0] = U0
        for n in range(n_steps):
            ref[:, n + 1] = self.fine_solver(ref[:, n], dT, dt)

        errors = []
        U_fine = np.copy(U)
        for _ in range(max_iter):
            U_new = np.zeros((2 * N, n_steps + 1))
            U_new[:, 0] = U0
            for n in range(n_steps):
                U_new[:, n + 1] = (coarse(U_new[:, n], dT)
                                   + self.fine_solver(U_fine[:, n], dT, dt)
                                   - coarse(U_fine[:, n], dT))
            err = np.linalg.norm(U_new - ref) / np.linalg.norm(ref)
            errors.append(float(err))
            U_fine = U_new
            if err < tol:
                break
        return np.array(errors)

    def initial_condition(self, modes=(1,), amplitudes=(1.0,)) -> np.ndarray:
        """u0 = sum a_m sin(2 pi m x / L), v0 = 0."""
        u0 = np.zeros(self.N)
        for m, a in zip(modes, amplitudes):
            u0 += a * np.sin(2.0 * np.pi * m * self.x / self.L)
        return np.concatenate([u0, np.zeros(self.N)])


# ---------------------------------------------------------------------------
# variable wave speed, posed in the travel-time variable
# ---------------------------------------------------------------------------
class TravelTimeGrid:
    """Variable speed through tau(x) = int dx/c(x).

    With that change of variable the equation becomes

        u_tt = u_tautau - (c_tau/c) u_tau ,

    a constant-speed wave equation with a lower-order impedance term.  The
    coarse propagator is the Strang composition of half a step of the impedance
    term (exact, since c_tau/c is diagonal), a full d'Alembert step with c = 1,
    and a second half step.  The test problem is posed directly on a uniform
    tau grid, so no interpolation between grids is needed.
    """

    def __init__(self, N: int = 100, L: float = 1.0, c0: float = 1.0,
                 amplitude: float = 0.4):
        self.N = int(N)
        self.L = float(L)
        self.dtau = self.L / self.N
        self.tau = np.arange(self.N) * self.dtau
        self.k = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.dtau)

        self.c_tau = c0 + amplitude * np.sin(2.0 * np.pi * self.tau / self.L)
        self.dc = amplitude * 2.0 * np.pi / self.L * np.cos(2.0 * np.pi * self.tau / self.L)
        self.impedance = self.dc / self.c_tau          # c_tau / c

        self.D1 = np.zeros((self.N, self.N))
        self.D2 = np.zeros((self.N, self.N))
        for i in range(self.N):
            self.D1[i, (i - 1) % self.N] = -1.0 / (2.0 * self.dtau)
            self.D1[i, (i + 1) % self.N] = +1.0 / (2.0 * self.dtau)
            self.D2[i, (i - 1) % self.N] = 1.0 / self.dtau ** 2
            self.D2[i, i] = -2.0 / self.dtau ** 2
            self.D2[i, (i + 1) % self.N] = 1.0 / self.dtau ** 2

        self.A = np.zeros((2 * self.N, 2 * self.N))
        self.A[:self.N, self.N:] = np.eye(self.N)
        self.A[self.N:, :self.N] = self.D2 - self.D1 * self.impedance[:, None]

    # fine propagator (same RK4 as above, but on this operator)
    def step_rk4(self, U, dt):
        k1 = self.A @ U
        k2 = self.A @ (U + 0.5 * dt * k1)
        k3 = self.A @ (U + 0.5 * dt * k2)
        k4 = self.A @ (U + dt * k3)
        return U + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def fine_solver(self, U, dT, dt=FINE_STEP):
        U = np.copy(U)
        for _ in range(int(round(dT / dt))):
            U = self.step_rk4(U, dt)
        return U

    def _interp(self, xp, values):
        i0 = np.floor(xp).astype(int) % self.N
        i1 = (i0 + 1) % self.N
        w = xp - np.floor(xp)
        return (1.0 - w) * values[i0] + w * values[i1]

    def _integral(self, v, dT):
        factor = np.zeros_like(self.k)
        nz = np.abs(self.k) > 1e-12
        factor[nz] = 2.0 * np.sin(self.k[nz] * dT) / self.k[nz]
        factor[~nz] = 2.0 * dT
        return np.fft.ifft(factor * np.fft.fft(v)).real

    def _dalembert(self, U, dT):
        """Exact constant-speed (c = 1 in tau) d'Alembert step."""
        N = self.N
        u, v = U[:N], U[N:]
        s = dT / self.dtau
        x = np.arange(N, dtype=float)
        xl, xr = (x - s) % N, (x + s) % N
        u_new = 0.5 * (self._interp(xl, u) + self._interp(xr, u)) + 0.5 * self._integral(v, dT)
        ux = self.D1 @ u
        v_new = (0.5 * (self._interp(xl, v) + self._interp(xr, v))
                 + 0.5 * (self._interp(xr, ux) - self._interp(xl, ux)))
        return np.concatenate([u_new, v_new])

    def _impedance_half(self, U, dT):
        N = self.N
        u, v = U[:N], U[N:]
        h = 0.5 * dT
        return np.concatenate([u, v - h * self.impedance * (self.D1 @ u)])

    def coarse_improved(self, U, dT):
        """Strang split: impedance/2, d'Alembert, impedance/2."""
        U = self._impedance_half(U, dT)
        U = self._dalembert(U, dT)
        return self._impedance_half(U, dT)

    def coarse_implicit_euler(self, U, dT):
        return np.linalg.solve(np.eye(2 * self.N) - dT * self.A, U)

    def exact_flow(self, U, dT, _cache={}):
        """exp(dT A) through one eigendecomposition (cached)."""
        if "eig" not in _cache:
            w, V = np.linalg.eig(self.A)
            _cache["eig"] = (w, V, np.linalg.inv(V))
        w, V, Vi = _cache["eig"]
        return (V @ (np.exp(w * dT) * (Vi @ U))).real

    def parareal(self, U0, dT, coarse, max_iter=40, tol=1e-6, T_final=1.0):
        N = self.N
        n_steps = int(round(T_final / dT))
        U = np.zeros((2 * N, n_steps + 1))
        U[:, 0] = U0
        for n in range(n_steps):
            U[:, n + 1] = coarse(U[:, n], dT)
        ref = np.zeros((2 * N, n_steps + 1))
        ref[:, 0] = U0
        for n in range(n_steps):
            ref[:, n + 1] = self.fine_solver(ref[:, n], dT)
        errors = []
        U_fine = np.copy(U)
        for _ in range(max_iter):
            U_new = np.zeros((2 * N, n_steps + 1))
            U_new[:, 0] = U0
            for n in range(n_steps):
                U_new[:, n + 1] = (coarse(U_new[:, n], dT)
                                   + self.fine_solver(U_fine[:, n], dT)
                                   - coarse(U_fine[:, n], dT))
            err = np.linalg.norm(U_new - ref) / np.linalg.norm(ref)
            errors.append(float(err))
            U_fine = U_new
            if err < tol:
                break
        return np.array(errors)


# ---------------------------------------------------------------------------
# convenience wrappers around a default periodic grid
# ---------------------------------------------------------------------------
_DEFAULT = Grid()


def step_rk4(U, dt):
    return _DEFAULT.step_rk4(U, dt)


def fine_solver(U, dT, dt=FINE_STEP):
    return _DEFAULT.fine_solver(U, dT, dt)


def spectral_integral_v(v, dT, c=None):
    return _DEFAULT.spectral_integral_v(v, dT)


def interp(xp, values):
    return _DEFAULT.interp(xp, values)


def interp_cubic(xp, values):
    return _DEFAULT.interp_cubic(xp, values)


def coarse_implicit_euler(U, dT):
    return _DEFAULT.coarse_implicit_euler(U, dT)


def coarse_improved(U, dT, use_cubic: bool = False):
    return _DEFAULT.coarse_improved(U, dT, use_cubic=use_cubic)


def parareal(U0, dT, dt, coarse, max_iter=15, tol=1e-6, T_final=1.0):
    return _DEFAULT.parareal(U0, dT, dt, coarse, max_iter=max_iter, tol=tol,
                             T_final=T_final)


def initial_condition(modes=(1,), amplitudes=(1.0,)):
    return _DEFAULT.initial_condition(modes, amplitudes)
