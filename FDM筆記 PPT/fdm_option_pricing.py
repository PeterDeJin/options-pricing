"""
Finite Difference Methods for Option Pricing
Based on: Financial Engineering Lecture Series, May 2026

Bugs fixed vs. original slides:
  [FIX 1] explicit_fdm_call: removed unused dead-code variable `idx`
  [FIX 2] cn_fdm_call / fdm_greeks: save old BCs before updating so the
           CN explicit half uses V^n boundary values, not V^{n+1}
  [FIX 3] fdm_greeks: j==Nt-2 → j==Nt-1; original spanned 2*dtau so
           theta was off by a factor of 2
  [FIX 4] american_put_implicit: lower BC must be V[0]=K (immediate
           exercise is always optimal at S=0); original used the
           European-put discounting BC V[0]*(1-r*dtau)
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────
# Black-Scholes analytical price  (benchmark)
# ─────────────────────────────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ─────────────────────────────────────────────────────────────────────
# Thomas algorithm  (tridiagonal solver)
# Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
# ─────────────────────────────────────────────────────────────────────
def thomas_solve(a, b, c, d):
    n   = len(b)
    c_  = np.zeros(n)
    d_  = np.zeros(n)
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n):
        m     = b[i] - a[i] * c_[i - 1]
        c_[i] = c[i] / m
        d_[i] = (d[i] - a[i] * d_[i - 1]) / m
    x = np.zeros(n)
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x


# ─────────────────────────────────────────────────────────────────────
# Explicit FDM  –  European Call
# Stability: sigma^2 * n_max^2 * dtau < 1  (CFL condition)
# ─────────────────────────────────────────────────────────────────────
def explicit_fdm_call(S0, K, T, r, sigma, NS=200, Nt=2000):
    S_max = 3 * K
    dS    = S_max / NS
    dtau  = T / Nt
    S     = np.linspace(0, S_max, NS + 1)
    V     = np.maximum(S - K, 0)          # payoff at tau=0

    n_vec = np.arange(1, NS)
    assert np.all(1 - (sigma**2 * n_vec**2 + r) * dtau > 0), "CFL violated!"

    for j in range(Nt):
        n    = n_vec
        am   = 0.5 * (sigma**2 * n**2 - r * n) * dtau
        a0   = 1.0 - (sigma**2 * n**2 + r) * dtau
        ap   = 0.5 * (sigma**2 * n**2 + r * n) * dtau
        Vnew = am * V[n - 1] + a0 * V[n] + ap * V[n + 1]
        V[0]    = V[0] * (1 - r * dtau)       # BC: S=0  (call → 0, discounted)
        V[NS]   = 2 * V[NS - 1] - V[NS - 2]   # BC: S=Smax  (linear extrapolation)
        V[1:NS] = Vnew

    # [FIX 1] original had `idx = int(round(S0/dS))` here – never used, removed
    return np.interp(S0, S, V)


# ─────────────────────────────────────────────────────────────────────
# Implicit FDM  –  European Call  (unconditionally stable)
# ─────────────────────────────────────────────────────────────────────
def implicit_fdm_call(S0, K, T, r, sigma, NS=200, Nt=500):
    S_max = 3 * K
    dS    = S_max / NS
    dtau  = T / Nt
    S     = np.linspace(0, S_max, NS + 1)
    V     = np.maximum(S - K, 0)
    n     = np.arange(1, NS)

    A = 0.5 * (r * n - sigma**2 * n**2) * dtau        # sub-diagonal
    B = (sigma**2 * n**2 + r) * dtau                  # diagonal offset
    C = -0.5 * (sigma**2 * n**2 + r * n) * dtau       # super-diagonal

    a_td = A.copy()
    b_td = 1 + B     # b = 1 + (sigma^2*n^2 + r)*dtau; r already inside B
    c_td = C.copy()

    for j in range(Nt):
        V[0]  = V[0] * (1 - r * dtau)
        V[NS] = 2 * V[NS - 1] - V[NS - 2]
        rhs      = V[1:NS].copy()
        rhs[0]  -= A[0]  * V[0]      # absorb lower BC
        rhs[-1] -= C[-1] * V[NS]     # absorb upper BC
        V[1:NS] = thomas_solve(a_td, b_td, c_td, rhs)

    return np.interp(S0, S, V)


# ─────────────────────────────────────────────────────────────────────
# Crank-Nicolson FDM  –  European Call  (O(dtau^2, dS^2))
# ─────────────────────────────────────────────────────────────────────
def cn_fdm_call(S0, K, T, r, sigma, NS=200, Nt=500):
    S_max = 3 * K
    dS    = S_max / NS
    dtau  = T / Nt
    S     = np.linspace(0, S_max, NS + 1)
    V     = np.maximum(S - K, 0)
    n     = np.arange(1, NS)

    # Half-step coefficients (CN = 0.5*implicit + 0.5*explicit)
    al = 0.25 * (r * n - sigma**2 * n**2) * dtau
    be = 0.5  * (sigma**2 * n**2 + r) * dtau
    ga = -0.25 * (sigma**2 * n**2 + r * n) * dtau

    for j in range(Nt):
        # [FIX 2] Save old BCs before updating.
        # The CN explicit half needs V^n boundary values; the slide mistakenly
        # used V^{n+1} for both halves (updating BCs before computing the RHS).
        V0_old  = V[0]
        VNS_old = V[NS]

        # Advance BCs to new time level
        V[0]  = V[0] * (1 - r * dtau)
        V[NS] = 2 * V[NS - 1] - V[NS - 2]

        # Explicit (old-level) RHS  –  interior nodes still hold V^n
        rhs = (-al * np.r_[V0_old,  V[1:NS - 1]]
               + (1 - be) * V[1:NS]
               - ga * np.r_[V[2:NS], VNS_old])

        # Absorb new BCs into the implicit LHS
        rhs[0]  -= al[0]  * V[0]
        rhs[-1] -= ga[-1] * V[NS]

        V[1:NS] = thomas_solve(al, 1 + be, ga, rhs)

    return np.interp(S0, S, V)


# ─────────────────────────────────────────────────────────────────────
# American Put  –  Implicit + Early-Exercise Projection
# ─────────────────────────────────────────────────────────────────────
def american_put_implicit(S0, K, T, r, sigma, NS=200, Nt=500):
    S_max  = 3 * K
    dS     = S_max / NS
    dtau   = T / Nt
    S      = np.linspace(0, S_max, NS + 1)
    payoff = np.maximum(K - S, 0)
    V      = payoff.copy()
    n      = np.arange(1, NS)

    A = 0.5 * (r * n - sigma**2 * n**2) * dtau
    B = (sigma**2 * n**2 + r) * dtau
    C = -0.5 * (sigma**2 * n**2 + r * n) * dtau
    b_td = 1 + B

    for j in range(Nt):
        # [FIX 4] American put at S=0: immediate exercise always optimal,
        # so V(0,tau)=K for all tau.  The slide used V[0]*(1-r*dtau) which
        # is the European put BC (= K*e^{-r*tau}), not the American one.
        V[0]  = payoff[0]   # = K  (constant, no discounting)
        V[NS] = 0.0         # put is worthless deep out-of-the-money

        rhs      = V[1:NS].copy()
        rhs[0]  -= A[0]  * V[0]
        rhs[-1] -= C[-1] * V[NS]

        V[1:NS] = thomas_solve(A, b_td, C, rhs)
        V[1:NS] = np.maximum(V[1:NS], payoff[1:NS])   # early-exercise projection

    return np.interp(S0, S, V)


# ─────────────────────────────────────────────────────────────────────
# Greeks  –  Delta, Gamma, Theta  (computed on the CN grid)
# ─────────────────────────────────────────────────────────────────────
def fdm_greeks(S0, K, T, r, sigma, NS=400, Nt=1000):
    S_max = 3 * K
    dS    = S_max / NS
    dtau  = T / Nt
    S     = np.linspace(0, S_max, NS + 1)
    V     = np.maximum(S - K, 0)
    n     = np.arange(1, NS)

    al = 0.25 * (r * n - sigma**2 * n**2) * dtau
    be = 0.5  * (sigma**2 * n**2 + r) * dtau
    ga = -0.25 * (sigma**2 * n**2 + r * n) * dtau

    V_prev = None
    for j in range(Nt):
        # [FIX 3] Must save V at the START of the LAST step (j==Nt-1).
        # The slide used j==Nt-2, which saved V two steps before the end,
        # making theta = -(V_T - V_{T-2*dtau})/dtau → factor-of-2 error.
        if j == Nt - 1:
            V_prev = V.copy()

        # [FIX 2 applied here too] Save old BCs for correct CN explicit half
        V0_old  = V[0]
        VNS_old = V[NS]
        V[0]  = V[0] * (1 - r * dtau)
        V[NS] = 2 * V[NS - 1] - V[NS - 2]

        rhs = (-al * np.r_[V0_old,  V[1:NS - 1]]
               + (1 - be) * V[1:NS]
               - ga * np.r_[V[2:NS], VNS_old])
        rhs[0]  -= al[0]  * V[0]
        rhs[-1] -= ga[-1] * V[NS]
        V[1:NS] = thomas_solve(al, 1 + be, ga, rhs)

    idx   = int(round(S0 / dS))
    delta = (V[idx + 1] - V[idx - 1]) / (2 * dS)
    gamma = (V[idx + 1] - 2 * V[idx] + V[idx - 1]) / dS**2
    # Theta = dV/dt = -dV/dtau; now V_prev is exactly 1 step behind V
    theta = -(V[idx] - V_prev[idx]) / dtau

    return {'Delta': round(delta, 4), 'Gamma': round(gamma, 4), 'Theta': round(theta, 4)}


# ─────────────────────────────────────────────────────────────────────
# Common grid parameters
# ─────────────────────────────────────────────────────────────────────
S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
NS, Nt = 200, 500
S_max  = 3 * K
dS     = S_max / NS
dtau   = T / Nt
S_grid = np.linspace(0, S_max, NS + 1)


# ─────────────────────────────────────────────────────────────────────
# Benchmark: all methods vs Black-Scholes formula
# ─────────────────────────────────────────────────────────────────────
exact = bs_call(100, 100, 1, 0.05, 0.20)   # ≈ 10.4506

results = {
    'BS Formula (exact)' : exact,
    'Explicit FDM'       : explicit_fdm_call(100, 100, 1, .05, .20, 200, 2000),
    'Implicit FDM'       : implicit_fdm_call(100, 100, 1, .05, .20, 200, 500),
    'Crank-Nicolson'     : cn_fdm_call(100, 100, 1, .05, .20, 200, 500),
    'American Put (Impl)': american_put_implicit(100, 100, 1, .05, .20, 200, 500),
}

print(f"{'Method':<25} {'Price':>10} {'Error':>10}")
print('-' * 47)
for name, price in results.items():
    err = abs(price - exact) if 'American' not in name else 'N/A'
    print(f"{name:<25} {price:>10.4f} {str(err):>10}")


# ─────────────────────────────────────────────────────────────────────
# CFL Stability Violation Demo
# ─────────────────────────────────────────────────────────────────────
print("\n=== CFL VIOLATION DEMO ===")
for Nt_test in [50, 100, 500, 2000]:
    try:
        price = explicit_fdm_call(100, 100, 1, .05, .20, NS=200, Nt=Nt_test)
        err   = abs(price - exact)
        flag  = "*** UNSTABLE ***" if price < 0 or price > 200 else ""
        print(f"Nt={Nt_test:5d}  price={price:10.4f}  err={err:.4f}  {flag}")
    except Exception as e:
        print(f"Nt={Nt_test:5d}  CRASHED: {e}")


# ─────────────────────────────────────────────────────────────────────
# Convergence Study: error vs grid size
# ─────────────────────────────────────────────────────────────────────
Ns_list = [50, 100, 200, 400]
errors  = {"Explicit": [], "Implicit": [], "CN": []}

for NS_c in Ns_list:
    Nt_exp = max(2000, int(NS_c**2 * 0.3))   # respect CFL for explicit
    errors["Explicit"].append(
        abs(explicit_fdm_call(100, 100, 1, .05, .2, NS_c, Nt_exp) - exact))
    errors["Implicit"].append(
        abs(implicit_fdm_call(100, 100, 1, .05, .2, NS_c, 500) - exact))
    errors["CN"].append(
        abs(cn_fdm_call(100, 100, 1, .05, .2, NS_c, 500) - exact))

fig, ax = plt.subplots(figsize=(7, 4))
for name, err in errors.items():
    ax.loglog(Ns_list, err, 'o-', label=name)
ax.set_xlabel("N_S (spatial nodes)")
ax.set_ylabel("Absolute error")
ax.legend()
ax.set_title("Convergence: FDM vs BS formula")
plt.tight_layout()
plt.savefig("convergence.pdf")
plt.show()
