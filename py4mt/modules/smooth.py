"""
Author: Volker Rath (DIAS)
Copilot (v2025-12-07)

Module: spline_selector_bspline.py
Purpose: Spline parameter selection using RGCV/MGCV and likelihood-based
         criteria (REML, GML, UBRE), with exact B-spline basis,
         integrated squared second derivative penalty, fit/plot routines,
         and bootstrap confidence bands.
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

class SplineSelectorBSpline:
    def __init__(self, x, y, knots, degree=3):
        self.x = x
        self.y = y
        self.knots = knots
        self.degree = degree
        self.X = self._build_basis()
        self.penalty_matrix = self._build_penalty_matrix()

    # ------------------------------------------------------------
    # Build B-spline basis matrix
    # ------------------------------------------------------------
    def _build_basis(self):
        n_basis = len(self.knots) - self.degree - 1
        X = np.zeros((len(self.x), n_basis))
        for i in range(n_basis):
            coeff = np.zeros(n_basis)
            coeff[i] = 1.0
            spline = BSpline(self.knots, coeff, self.degree)
            X[:, i] = spline(self.x)
        return X

    # ------------------------------------------------------------
    # Penalty matrix: ∫ (B_i'' * B_j'') dx
    # ------------------------------------------------------------
    def _build_penalty_matrix(self):
        n_basis = self.X.shape[1]
        P = np.zeros((n_basis, n_basis))
        grid = np.linspace(min(self.x), max(self.x), 400)
        for i in range(n_basis):
            coeff_i = np.zeros(n_basis); coeff_i[i] = 1.0
            Bi = BSpline(self.knots, coeff_i, self.degree)
            Bi_dd = Bi.derivative(2)(grid)
            for j in range(n_basis):
                coeff_j = np.zeros(n_basis); coeff_j[j] = 1.0
                Bj = BSpline(self.knots, coeff_j, self.degree)
                Bj_dd = Bj.derivative(2)(grid)
                P[i, j] = np.trapezoid(Bi_dd * Bj_dd, grid)
        return P

    # ------------------------------------------------------------
    # Hat matrix
    # ------------------------------------------------------------
    def hat_matrix(self, lam):
        XtX = self.X.T @ self.X
        A = XtX + lam * self.penalty_matrix
        A_inv = la.inv(A)
        return self.X @ A_inv @ self.X.T

    # ------------------------------------------------------------
    # RGCV / MGCV
    # ------------------------------------------------------------
    def rgcv(self, lam, alpha=1.0):
        H = self.hat_matrix(lam)
        yhat = H @ self.y
        resid = self.y - yhat
        rss = np.sum(resid**2)
        trH = np.trace(H)
        denom = (1 - alpha * trH / len(self.y))**2
        return rss / denom

    # ------------------------------------------------------------
    # Likelihood-based criteria
    # ------------------------------------------------------------
    def likelihood(self, lam, method="REML"):
        H = self.hat_matrix(lam)
        yhat = H @ self.y
        resid = self.y - yhat
        n = len(self.y)
        sigma2 = np.sum(resid**2) / (n - np.trace(H))

        if method == "REML":
            logdet = np.log(np.linalg.det(np.eye(n) - H))
            return n * np.log(sigma2) + logdet
        elif method == "GML":
            logdet = np.log(np.linalg.det(np.eye(n) - H))
            return n * np.log(sigma2) + 2 * logdet
        else:
            raise ValueError("Unknown method")

    # ------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------
    def select_lambda(self, method="RGCV", lam_grid=None):
        if lam_grid is None:
            lam_grid = np.logspace(-3, 3, 50)

        scores = {}
        for lam in lam_grid:
            if method in ["RGCV", "MGCV"]:
                alpha = 1.0 if method == "RGCV" else 1.4
                score = self.rgcv(lam, alpha=alpha)
            else:
                score = self.likelihood(lam, method=method)
            scores[lam] = score

        lam_opt = min(scores, key=scores.get)
        return lam_opt, scores

    # ------------------------------------------------------------
    # Fit spline for given λ
    # ------------------------------------------------------------
    def fit(self, lam):
        XtX = self.X.T @ self.X
        A = XtX + lam * self.penalty_matrix
        coeffs = la.solve(A, self.X.T @ self.y)
        return coeffs

    def predict(self, coeffs, x_new):
        n_basis = self.X.shape[1]
        X_new = np.zeros((len(x_new), n_basis))
        for i in range(n_basis):
            coeff = np.zeros(n_basis); coeff[i] = 1.0
            spline = BSpline(self.knots, coeff, self.degree)
            X_new[:, i] = spline(x_new)
        return X_new @ coeffs

    # ------------------------------------------------------------
    # Bootstrap confidence bands
    # ------------------------------------------------------------
    def bootstrap_bands(self, lam, n_boot=200, alpha=0.05):
        coeffs = self.fit(lam)
        yhat = self.predict(coeffs, self.x)
        resid = self.y - yhat

        x_new = np.linspace(min(self.x), max(self.x), 200)
        fits = []
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            resampled = rng.choice(resid, size=len(resid), replace=True)
            y_boot = yhat + resampled
            coeffs_boot = la.solve(self.X.T @ self.X + lam*self.penalty_matrix,
                                   self.X.T @ y_boot)
            fits.append(self.predict(coeffs_boot, x_new))
        fits = np.array(fits)
        lower = np.percentile(fits, 100*alpha/2, axis=0)
        upper = np.percentile(fits, 100*(1-alpha/2), axis=0)
        return x_new, lower, upper

    # ------------------------------------------------------------
    # Benchmark all methods with fitted curves + bands
    # ------------------------------------------------------------
    def benchmark_all(self, lam_grid=None, plot_fit=True, with_bands=True):
        methods = ["RGCV", "MGCV", "REML", "GML"]
        if lam_grid is None:
            lam_grid = np.logspace(-6, 3, 50)

        results = {}
        # Criterion plots
        fig, axes = plt.subplots(2, 3, figsize=(12,8))
        axes = axes.flatten()
        for idx, method in enumerate(methods):
            lam_opt, scores = self.select_lambda(method=method, lam_grid=lam_grid)
            coeffs = self.fit(lam_opt)
            results[method] = (lam_opt, scores, coeffs)

            ax = axes[idx]
            ax.semilogx(list(scores.keys()), list(scores.values()), marker="o")
            ax.axvline(lam_opt, color="red", linestyle="--", label=f"opt λ={lam_opt:.3f}")
            ax.set_title(method)
            ax.set_xlabel("λ")
            ax.set_ylabel("criterion")
            ax.legend()
            ax.grid(True, which="both", ls="--")

        if len(methods) < len(axes):
            fig.delaxes(axes[-1])
        plt.tight_layout()
        plt.show()

        # Fit plots
        if plot_fit:
            x_new = np.linspace(min(self.x), max(self.x), 200)
            plt.figure(figsize=(10,6))
            plt.scatter(self.x, self.y, color="gray", alpha=0.6, label="data")
            for method, (lam_opt, _, coeffs) in results.items():
                y_fit = self.predict(coeffs, x_new)
                plt.plot(x_new, y_fit, label=f"{method} (λ={lam_opt:.3f})")
                if with_bands:
                    x_band, lower, upper = self.bootstrap_bands(lam_opt)
                    plt.fill_between(x_band, lower, upper, alpha=0.2)
            plt.legend()
            plt.title("Spline fits with bootstrap confidence bands")
            plt.show()

        return results

    def normal_bands(self, lam, alpha=0.05):
        H = self.hat_matrix(lam)
        yhat = H @ self.y
        resid = self.y - yhat
        sigma2 = np.sum(resid**2) / (len(self.y) - np.trace(H))

        # Standard error of fitted values
        se = np.sqrt(np.diag(H @ H.T) * sigma2)

        z = 1.96  # for 95% CI
        lower = yhat - z * se
        upper = yhat + z * se
        return self.x, lower, upper

    def hat_matrix_weighted(self, lam, W):
        # W is diagonal weights (n,), typically 1/Var(y_i)
        WX = (W[:, None] * self.X)
        XtWX = self.X.T @ (W[:, None] * self.X)
        A = XtWX + lam * self.penalty_matrix
        A_inv = la.inv(A)
        return WX @ A_inv @ self.X.T

# 1) Generate synthetic data
n = 150
x = np.linspace(0, 1, n)
true_fn = lambda t: np.sin(2*np.pi*t) + 0.5*np.cos(4*np.pi*t)
y = true_fn(x) + 0.15*np.random.randn(n)

# 2) Build a cubic B-spline knot vector
degree = 3
n_internal_knots = 15
internal = np.linspace(x.min(), x.max(), n_internal_knots)
# Open uniform knot vector: pad with boundary knots repeated 'degree' times
knots = np.concatenate(([x.min()]*degree, internal, [x.max()]*degree))

# 3) Initialize selector
selector = SplineSelectorBSpline(x, y, knots, degree=degree)

# 4) Run benchmark: criteria curves, fitted splines, and bootstrap bands
# - lam_grid controls the search range; adjust for your scale if needed
results = selector.benchmark_all(lam_grid=np.logspace(-5, 3, 60),
                                 plot_fit=True,
                                 with_bands=True)

# 5) Report optimal λ per method, and reconstruct fits for downstream use
print("\nOptimal λ per method:")
fitted_curves = {}
x_eval = np.linspace(x.min(), x.max(), 300)

for method, (lam_opt, scores, coeffs) in results.items():
    print(f"  {method}: λ* = {lam_opt:.6f}")
    y_eval = selector.predict(coeffs, x_eval)
    fitted_curves[method] = {
        "lambda": lam_opt,
        "coeffs": coeffs,
        "x": x_eval,
        "y": y_eval,
        "scores": scores,
    }

# 6) Example: access MGCV fit and its bootstrap band for custom plotting/analysis
lam_mgcv = fitted_curves["MGCV"]["lambda"]
x_band, lower_band, upper_band = selector.bootstrap_bands(lam_mgcv, n_boot=300, alpha=0.05)

# 7) (Optional) Simple provenance dictionary
provenance = {
    "n": n,
    "degree": degree,
    "n_internal_knots": n_internal_knots,
    "lam_grid_bounds": (-4, 4),
    "methods": list(results.keys()),
}

print("\nProvenance summary:", provenance)
