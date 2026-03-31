"""
===========================================================
Hawkes Process Layer — Priority 4 Redundancy Check
Traders @ SMU Quant Finance Club
===========================================================

Workflow:
1) Build VPIN kernel (K_VPIN)
2) Build Hawkes proxy kernel (K_HAWKES_PROXY) — NO MLE
3) Compute Centered Kernel Alignment (CKA)
4) Decision rule:
      CKA > 0.75 → redundant → STOP
      CKA < 0.5  → independent → allow full Hawkes implementation
      else       → ambiguous

IMPORTANT:
We DO NOT implement full Hawkes MLE unless CKA < 0.5
===========================================================
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize


# ==========================================================
# -------------------- UTILITY FUNCTIONS -------------------
# ==========================================================

def rbf_kernel(X, sigma=None):
    """
    Compute RBF (Gaussian) kernel matrix.
    """
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    if sigma is None:
        # median heuristic
        sigma = np.median(pairwise_sq_dists)
        sigma = np.sqrt(sigma + 1e-8)
    K = np.exp(-pairwise_sq_dists / (2 * sigma**2))
    return K


def center_gram(K):
    """
    Center Gram matrix using H = I - 11^T/n
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def compute_cka(K, L):
    """
    Linear CKA between two Gram matrices
    """
    Kc = center_gram(K)
    Lc = center_gram(L)
    numerator = np.trace(Kc @ Lc)
    denominator = np.sqrt(np.trace(Kc @ Kc) * np.trace(Lc @ Lc))
    return numerator / (denominator + 1e-12)


# ==========================================================
# ------------------ VPIN KERNEL BUILDER -------------------
# ==========================================================

def build_vpin_kernel(vpin_features, kernel_type="rbf"):
    """
    vpin_features: (n_samples x d_features)
    """
    if kernel_type == "linear":
        return vpin_features @ vpin_features.T
    elif kernel_type == "rbf":
        return rbf_kernel(vpin_features)
    else:
        raise ValueError("Unsupported kernel type.")


# ==========================================================
# ------------ HAWKES PROXY FEATURE BUILDER ----------------
# ==========================================================

def build_hawkes_proxy_features(event_timestamps, window_size, total_time):
    """
    Build cheap Hawkes proxy features WITHOUT MLE.

    Instead of fitting Hawkes, we approximate excitation by:
        1) Event counts per window
        2) Inter-arrival mean
        3) Short-term burst ratio

    event_timestamps: array of trade times
    window_size: bin size
    total_time: total time horizon

    Returns: (n_windows x 3 feature matrix)
    """

    bins = np.arange(0, total_time + window_size, window_size)
    counts, _ = np.histogram(event_timestamps, bins=bins)

    n_windows = len(counts)
    features = []

    for i in range(n_windows):
        window_start = bins[i]
        window_end = bins[i + 1]
        mask = (event_timestamps >= window_start) & (event_timestamps < window_end)
        events = event_timestamps[mask]

        count = counts[i]

        if len(events) > 1:
            inter_arrivals = np.diff(events)
            mean_ia = np.mean(inter_arrivals)
            burst_ratio = count / (mean_ia + 1e-8)
        else:
            mean_ia = 0
            burst_ratio = 0

        features.append([count, mean_ia, burst_ratio])

    return np.array(features)


def build_hawkes_proxy_kernel(proxy_features, kernel_type="rbf"):
    if kernel_type == "linear":
        return proxy_features @ proxy_features.T
    elif kernel_type == "rbf":
        return rbf_kernel(proxy_features)
    else:
        raise ValueError("Unsupported kernel type.")


# ==========================================================
# ----------------- FULL HAWKES MLE MODEL ------------------
# (ONLY USED IF CKA < 0.5)
# ==========================================================

class HawkesMLE:
    """
    Univariate exponential Hawkes process:
        λ(t) = μ + Σ α exp(-β(t - ti))
    """

    def __init__(self, timestamps):
        self.timestamps = np.array(timestamps)

    def log_likelihood(self, params):
        mu, alpha, beta = params
        if mu <= 0 or alpha < 0 or beta <= 0:
            return np.inf

        t = self.timestamps
        n = len(t)
        intensity = np.zeros(n)

        for i in range(n):
            history = t[:i]
            if len(history) > 0:
                intensity[i] = mu + np.sum(alpha * np.exp(-beta * (t[i] - history)))
            else:
                intensity[i] = mu

        log_part = np.sum(np.log(intensity + 1e-12))

        integral = mu * t[-1]
        for ti in t:
            integral += (alpha / beta) * (1 - np.exp(-beta * (t[-1] - ti)))

        return -(log_part - integral)

    def fit(self):
        init = np.array([0.1, 0.5, 1.0])
        bounds = [(1e-5, None), (0, None), (1e-5, None)]
        result = minimize(self.log_likelihood, init, bounds=bounds)
        return result.x


# ==========================================================
# ------------------- MAIN PIPELINE ------------------------
# ==========================================================

def hawkes_layer_pipeline(vpin_features,
                          event_timestamps,
                          window_size,
                          total_time):

    print("Building VPIN Kernel...")
    K_vpin = build_vpin_kernel(vpin_features)

    print("Building Hawkes Proxy Features...")
    hawkes_proxy_features = build_hawkes_proxy_features(
        event_timestamps,
        window_size,
        total_time
    )

    print("Building Hawkes Proxy Kernel...")
    K_hawkes_proxy = build_hawkes_proxy_kernel(hawkes_proxy_features)

    print("Computing CKA...")
    cka_value = compute_cka(K_vpin, K_hawkes_proxy)
    print(f"CKA(K_VPIN, K_HAWKES_PROXY) = {cka_value:.4f}")

    if cka_value > 0.75:
        print("\nResult: REDUNDANT. Do NOT implement Hawkes layer.")
        return {"cka": cka_value, "decision": "redundant"}

    elif cka_value < 0.5:
        print("\nResult: Independent signal detected.")
        print("Proceeding with FULL Hawkes MLE estimation...")

        hawkes_model = HawkesMLE(event_timestamps)
        mu, alpha, beta = hawkes_model.fit()

        print(f"Estimated Parameters: μ={mu:.4f}, α={alpha:.4f}, β={beta:.4f}")

        return {
            "cka": cka_value,
            "decision": "independent",
            "hawkes_params": (mu, alpha, beta)
        }

    else:
        print("\nResult: Ambiguous (0.5 ≤ CKA ≤ 0.75).")
        print("Recommend further feature refinement before committing.")
        return {"cka": cka_value, "decision": "ambiguous"}


# ==========================================================
# ------------------- EXAMPLE USAGE ------------------------
# ==========================================================

if __name__ == "__main__":

    # Simulated example (replace with historical data)
    np.random.seed(42)

    # Simulated VPIN features
    vpin_features = np.random.randn(200, 5)

    # Simulated event timestamps
    event_timestamps = np.cumsum(np.random.exponential(scale=1.0, size=1000))
    total_time = event_timestamps[-1]

    result = hawkes_layer_pipeline(
        vpin_features=vpin_features,
        event_timestamps=event_timestamps,
        window_size=5.0,
        total_time=total_time
    )

    print("\nFinal Decision:", result)
