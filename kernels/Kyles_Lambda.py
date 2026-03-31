"""
====================================================================
Kyle's Lambda Layer — Priority Verification Framework
Traders @ SMU Quant Finance Club
====================================================================

STRICT PRIORITY RULE:
Before integrating full Kyle’s Lambda layer into RKHS stack,
run Centered Kernel Alignment (CKA) against K_VPIN.

Decision Rule:
    CKA > 0.75  → Redundant → STOP
    CKA < 0.50  → Independent → Allow Integration
    0.50–0.75   → Ambiguous → Investigate further

====================================================================
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.special import kv, gamma
from sklearn.linear_model import LinearRegression


# ==============================================================
# ---------------------- KERNEL UTILITIES ----------------------
# ==============================================================

def center_gram(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def compute_cka(K, L):
    Kc = center_gram(K)
    Lc = center_gram(L)
    num = np.trace(Kc @ Lc)
    den = np.sqrt(np.trace(Kc @ Kc) * np.trace(Lc @ Lc))
    return num / (den + 1e-12)


def matern_kernel(X, nu=1.5, rho=1.0):
    """
    Matérn kernel with smoothness nu (default 1.5)
    """
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    d = pairwise_dists + 1e-12

    if nu == 0.5:
        K = np.exp(-d / rho)
    else:
        factor = (2 ** (1 - nu)) / gamma(nu)
        scaled = np.sqrt(2 * nu) * d / rho
        K = factor * (scaled ** nu) * kv(nu, scaled)
        K[d == 0] = 1.0

    return K


# ==============================================================
# ---------------------- DOLLAR BAR CREATION -------------------
# ==============================================================

def create_dollar_bars(df, dollar_threshold):
    """
    df must contain:
        - price
        - size
        - signed_size (signed order flow)
    """
    cumulative_dollar = 0
    bars = []
    bar = []

    for _, row in df.iterrows():
        dollar_value = row['price'] * abs(row['size'])
        cumulative_dollar += dollar_value
        bar.append(row)

        if cumulative_dollar >= dollar_threshold:
            bars.append(pd.DataFrame(bar))
            bar = []
            cumulative_dollar = 0

    return bars


# ==============================================================
# ------------------- ROLLING LAMBDA ESTIMATION ----------------
# ==============================================================

def rolling_lambda(price_series, signed_flow, window):
    """
    Rolling OLS: ΔP = λX + ε
    """
    lambdas = np.full(len(price_series), np.nan)

    for i in range(window, len(price_series)):
        y = price_series[i-window:i].diff().dropna().values.reshape(-1, 1)
        x = signed_flow[i-window:i].values.reshape(-1, 1)

        if len(y) == len(x):
            model = LinearRegression().fit(x, y)
            lambdas[i] = model.coef_[0][0]

    return lambdas


# ==============================================================
# ------------------ LOOK-AHEAD SAFE RESIDUAL ------------------
# ==============================================================

def compute_residual(price, signed_flow, lambda_series, window=20):
    """
    Look-ahead free residual construction
    """
    residual = np.full(len(price), np.nan)

    for t in range(1, len(price)):
        if not np.isnan(lambda_series[t-1]):
            expected_impact = lambda_series[t-1] * signed_flow.iloc[t]
            delta_p = price.iloc[t] - price.iloc[t-1]
            residual[t] = delta_p - expected_impact

    # Out-of-sample rolling z-score
    rolling_mean = pd.Series(residual).rolling(window).mean()
    rolling_std = pd.Series(residual).rolling(window).std()

    z_residual = (residual - rolling_mean) / (rolling_std + 1e-12)

    return z_residual.values


# ==============================================================
# ----------------- BUILD KYLE FEATURE VECTOR ------------------
# ==============================================================

def build_kyle_features(price_series, signed_flow):
    """
    4D feature vector:
        λ_1
        λ_5
        λ_20
        residual_z
    """
    lambda_1 = rolling_lambda(price_series, signed_flow, window=1)
    lambda_5 = rolling_lambda(price_series, signed_flow, window=5)
    lambda_20 = rolling_lambda(price_series, signed_flow, window=20)

    residual_z = compute_residual(price_series, signed_flow, lambda_20)

    features = np.column_stack([
        lambda_1,
        lambda_5,
        lambda_20,
        residual_z
    ])

    # Drop NaNs
    features = features[~np.isnan(features).any(axis=1)]

    return features


# ==============================================================
# ------------------ VPIN KERNEL (REFERENCE) -------------------
# ==============================================================

def build_vpin_kernel(vpin_features):
    pairwise_sq_dists = squareform(pdist(vpin_features, 'sqeuclidean'))
    sigma = np.sqrt(np.median(pairwise_sq_dists) + 1e-12)
    return np.exp(-pairwise_sq_dists / (2 * sigma**2))


# ==============================================================
# -------------------- MAIN LAYER PIPELINE ---------------------
# ==============================================================

def kyle_lambda_layer_pipeline(df,
                                vpin_features,
                                dollar_threshold):

    print("Creating Dollar Bars...")
    bars = create_dollar_bars(df, dollar_threshold)

    prices = []
    signed_flows = []

    for bar in bars:
        prices.append(bar['price'].iloc[-1])
        signed_flows.append(bar['signed_size'].sum())

    price_series = pd.Series(prices)
    signed_flow_series = pd.Series(signed_flows)

    print("Building Kyle Feature Vector...")
    kyle_features = build_kyle_features(price_series, signed_flow_series)

    print("Building Matérn Kernel...")
    K_kyle = matern_kernel(kyle_features, nu=1.5, rho=1.0)

    print("Building VPIN Kernel...")
    K_vpin = build_vpin_kernel(vpin_features[:len(K_kyle)])

    print("Running CKA Redundancy Check...")
    cka_value = compute_cka(K_kyle, K_vpin)
    print(f"CKA(K_Kyle, K_VPIN) = {cka_value:.4f}")

    if cka_value > 0.75:
        print("\nResult: REDUNDANT. Do NOT integrate Kyle’s Lambda layer.")
        return {"cka": cka_value, "decision": "redundant"}

    elif cka_value < 0.5:
        print("\nResult: Independent signal confirmed.")
        print("Kyle’s Lambda layer APPROVED for RKHS integration.")

        return {
            "cka": cka_value,
            "decision": "independent",
            "kernel": K_kyle
        }

    else:
        print("\nResult: Ambiguous. Further refinement required.")
        return {"cka": cka_value, "decision": "ambiguous"}


# ==============================================================
# ---------------------- RKHS COMPOSITE ------------------------
# ==============================================================

def build_global_kernel(K_lob, K_vpin, K_kyle):
    """
    Tensor product (Hadamard for computational tractability)
    """
    return K_lob * K_vpin * K_kyle


# ==============================================================
# ------------------------- EXAMPLE ----------------------------
# ==============================================================

if __name__ == "__main__":

    np.random.seed(42)

    # Simulated trade-level dataset
    df = pd.DataFrame({
        "price": np.cumsum(np.random.randn(10000)) + 100,
        "size": np.random.randint(1, 10, 10000),
        "signed_size": np.random.choice([-1, 1], 10000) * np.random.randint(1, 10, 10000)
    })

    # Simulated VPIN features
    vpin_features = np.random.randn(500, 4)

    result = kyle_lambda_layer_pipeline(
        df=df,
        vpin_features=vpin_features,
        dollar_threshold=5000
    )

    print("\nFinal Decision:", result)
