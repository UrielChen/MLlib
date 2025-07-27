import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List, Union


def phil_test_main(save_path, total_pred_flat, total_targets_flat, n_asset, t_time=-1):
    total_pred = total_pred_flat.reshape(t_time, n_asset, order='C')
    total_targets = total_targets_flat.reshape(t_time, n_asset, order='C')
    print("MSE:", np.mean((total_pred - total_targets) ** 2))
    oos_r2 = panel_oos_r2_general(total_targets, total_pred, "uncentered", "both")
    if isinstance(oos_r2, pd.Series):
        oos_r2 = pd.DataFrame(oos_r2)
        oos_r2.to_csv(Path(save_path, 'oss_r2.csv'))

    decile_portfolios = construct_decile_portfolios(total_targets, total_pred, K=5, weighting_scheme='inverse_vol')
    metrics_selected = calculate_performance_metrics(
        returns=pd.concat([decile_portfolios['decile_returns'], decile_portfolios['long_short_return']], axis=1),
        cumulative_returns=pd.concat([decile_portfolios['decile_cum_returns'], decile_portfolios['long_short_cum_return']], axis=1),
        portfolios_to_evaluate="all"
    )
    print("\n=== Performance of Portfolios ===")
    print(metrics_selected)
    return



def panel_oos_r2_general(y_true, y_pred, r2_type='mean', mode='total'):
    """
    Function: panel_oos_r2_general

    Description:
    ------------
    Computes panel out-of-sample R² either using:
    1. Uncentered R² (benchmark is 0)
    2. Mean-based R² (benchmark is expanding historical mean)

    Supports two output modes:
    - 'total'  : a single scalar R² for the whole panel
    - 'partial': per-entity R²s as a pandas Series

    Inputs:
    -------
    - y_true   : DataFrame or ndarray, shape = (T, N), true target values
    - y_pred   : DataFrame or ndarray, shape = (T, N), predicted values
    - r2_type  : str, either 'uncentered' or 'mean'
    - mode     : str, either 'total' or 'partial'

    Outputs:
    --------
    - float if mode == 'total'
    - pd.Series if mode == 'partial'

    Assumptions:
    ------------
    - Index alignment is assumed if using DataFrame
    - NaNs are automatically handled (excluded)
    """
    # Step 0: Validate r2_type
    valid_r2_types = {'uncentered', 'mean'}
    if r2_type not in valid_r2_types:
        raise ValueError(f"Invalid r2_type: {r2_type}. Choose from {valid_r2_types}.")

    # Step 1: Validate mode
    valid_modes = {'total', 'partial', 'both'}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Choose from {valid_modes}.")

    # Step 2: Convert ndarray inputs to DataFrame
    if isinstance(y_true, np.ndarray):
        y_true = pd.DataFrame(y_true)  # shape: (T, N)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred)  # shape: (T, N)

    # Step 3: Ensure shapes match
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"

    # Step 4: Initialize depending on mode
    r2_dict = {}  # to store R² per entity
    rss_total = 0.0  # total residual sum of squares
    tss_total = 0.0  # total sum of squares

    # Step 5: Loop over entities (columns)
    for col in y_true.columns:
        yt = y_true[col]  # shape: (T,)
        yp = y_pred[col]  # shape: (T,)

        # Valid points where both yt and yp are non-NaN
        valid_idx = yt.notna() & yp.notna()
        if valid_idx.sum() < 2:
            continue  # skip if not enough data

        if r2_type == 'uncentered':
            yt_valid = yt[valid_idx]
            yp_valid = yp[valid_idx]
            rss = ((yt_valid - yp_valid) ** 2).sum()
            tss = (yt_valid ** 2).sum()

        elif r2_type == 'mean':
            expanding_mean = yt.expanding(min_periods=2).mean().shift(1)
            valid_mean_idx = expanding_mean.notna()
            final_idx = valid_idx & valid_mean_idx

            if final_idx.sum() < 2:
                continue  # not enough usable data

            yt_final = yt[final_idx]
            yp_final = yp[final_idx]
            mean_final = expanding_mean[final_idx]

            rss = ((yt_final - yp_final) ** 2).sum()
            tss = ((yt_final - mean_final) ** 2).sum()
        else:
            raise ValueError
        # Store or accumulate depending on mode
        if mode == 'partial':
            r2 = 1 - rss / tss
            r2_dict[col] = r2

        elif mode == 'total':
            rss_total += rss
            tss_total += tss
        elif mode == 'both':
            r2 = 1 - rss / tss
            r2_dict[col] = r2
            rss_total += rss
            tss_total += tss
        else:
            raise ValueError

    # Step 6: Return result based on mode
    if mode == 'partial':
        r2_series = pd.Series(r2_dict, name='partial_R2')
        print("Partial (per-entity) OOS R² calculated:")
        print(r2_series)
        return r2_series

    elif mode == 'total':
        r2_total = 1 - rss_total / tss_total
        print(f"Total OOS Panel R² ({r2_type}): {r2_total:.4f}")
        return r2_total
    elif mode == "both":
        r2_series = pd.Series(r2_dict, name='partial_R2')
        r2_total = 1 - rss_total / tss_total
        # print("Partial (per-entity) OOS R² calculated:")
        # print(r2_series)
        print(f"Total OOS Panel R² ({r2_type}): {r2_total:.4f}")
        return r2_series
    else:
        raise ValueError


def construct_decile_portfolios(y_true: Union[pd.DataFrame, np.ndarray], y_pred: Union[pd.DataFrame, np.ndarray],
                                K=10, weighting_scheme='equal', rolling_vol_window=21) -> Dict[str, pd.DataFrame]:
    """
    Constructs decile-based portfolios and a long-short portfolio.

    Parameters:
    - y_true: pd.DataFrame or np.ndarray of true returns, shape (T, N)
    - y_pred: pd.DataFrame or np.ndarray of predicted returns, shape (T, N)
    - K: number of deciles to split the cross-section into
    - weighting_scheme: 'equal' or 'inverse_vol'
    - rolling_vol_window: window size for rolling volatility (used if weighting_scheme == 'inverse_vol')

    Returns:
    - results: dict with:
        'decile_returns': pd.DataFrame of shape (T, K)
        'long_short_return': pd.Series of shape (T,)
    """

    # Convert to DataFrames if inputs are arrays
    if isinstance(y_true, np.ndarray):
        y_true = pd.DataFrame(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred)

    # Ensure alignment
    assert y_true.shape == y_pred.shape, "Shape mismatch between y_true and y_pred"
    assert (y_true.index == y_pred.index).all(), "Time index mismatch"
    assert (y_true.columns == y_pred.columns).all(), "Asset columns mismatch"

    T, N = y_true.shape
    decile_returns = pd.DataFrame(index=y_true.index, columns=[f'D{d + 1}' for d in range(K)])
    long_short_return = pd.Series(index=y_true.index, dtype=float, name='long_short')

    # Pre-compute rolling volatility
    if weighting_scheme == 'inverse_vol':
        rolling_vol = y_true.rolling(window=rolling_vol_window).std()

    for t in y_true.index:
        pred_t = y_pred.loc[t, :]
        true_t = y_true.loc[t, :]

        # Drop NaNs
        valid = pred_t.notna() & true_t.notna()
        if valid.sum() < K:
            continue  # not enough assets to split into K deciles

        pred_t_valid = pred_t[valid]
        true_t_valid = true_t[valid]

        # Assign deciles (0 to K-1)
        try:
            deciles = pd.qcut(pred_t_valid, K, labels=False, duplicates='drop')
        except ValueError:
            continue  # skip if qcut fails

        for d in range(K):
            assets_in_decile = deciles[deciles == d].index
            if len(assets_in_decile) == 0:
                decile_returns.loc[t, f'D{d + 1}'] = np.nan
                continue

            if weighting_scheme == 'equal':
                weight = 1.0 / len(assets_in_decile)
                returns = true_t_valid[assets_in_decile]
                decile_return = (returns * weight).sum()

            elif weighting_scheme == 'inverse_vol':
                vol_t = rolling_vol.loc[t, assets_in_decile]
                valid_vol = vol_t > 0
                if valid_vol.sum() == 0:
                    decile_returns.loc[t, f'D{d + 1}'] = np.nan
                    continue
                inv_vol = 1.0 / vol_t[valid_vol]
                weights = inv_vol / inv_vol.sum()
                returns = true_t_valid[weights.index]
                decile_return = (returns * weights).sum()

            else:
                raise NotImplementedError("Supported weighting schemes: 'equal', 'inverse_vol'.")

            decile_returns.loc[t, f'D{d + 1}'] = decile_return

        # Long-short: top decile - bottom decile
        top_return = decile_returns.loc[t, f'D{K}']
        bottom_return = decile_returns.loc[t, 'D1']
        if pd.notna(top_return) and pd.notna(bottom_return):
            long_short_return.loc[t] = top_return - bottom_return
        else:
            long_short_return.loc[t] = np.nan

    decile_cum_returns = (1 + decile_returns).cumprod(axis=0)
    long_short_cum_returns = (1 + long_short_return).cumprod(axis=0)

    return {
        'decile_returns': decile_returns.astype(float),
        'decile_cum_returns': decile_cum_returns.astype(float),
        'long_short_return': long_short_return,
        'long_short_cum_return': long_short_cum_returns
    }



def calculate_performance_metrics(returns: pd.DataFrame,
                                  cumulative_returns: pd.DataFrame,
                                  portfolios_to_evaluate: Union[List[str], str] = 'all',
                                  risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Calculate performance metrics for selected portfolios.

    Parameters:
    - returns: pd.DataFrame of portfolio returns (T x N)
    - cumulative_returns: pd.DataFrame of cumulative returns (T x N)
    - portfolios_to_evaluate: list of portfolio names to evaluate or 'all' for all columns
    - risk_free_rate: float, annualized risk-free rate (default is 0)

    Returns:
    - metrics_df: pd.DataFrame of performance metrics (N portfolios x metrics)
    """

    # Validate and select portfolios
    if portfolios_to_evaluate == 'all':
        selected_cols = returns.columns.intersection(cumulative_returns.columns)
    else:
        if isinstance(portfolios_to_evaluate, str):
            portfolios_to_evaluate = [portfolios_to_evaluate]
        missing = set(portfolios_to_evaluate) - set(returns.columns) - set(cumulative_returns.columns)
        if missing:
            raise ValueError(f"Portfolio(s) not found in returns or cumulative returns: {missing}")
        selected_cols = portfolios_to_evaluate

    metrics = {}

    for col in selected_cols:
        ret = returns[col].dropna()
        cum_ret = cumulative_returns[col].dropna()

        if ret.empty or cum_ret.empty:
            continue

        # Annualized return
        ann_return = (cum_ret.iloc[-1]) ** (12 / len(ret)) - 1

        # Annualized volatility
        ann_vol = ret.std() * np.sqrt(12)

        # Sharpe ratio
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else np.nan

        # Sortino ratio
        downside_std = np.sqrt((ret[ret < 0] ** 2).mean()) * np.sqrt(12)
        sortino = (ann_return - risk_free_rate) / downside_std if downside_std != 0 else np.nan

        # Maximum drawdown
        running_max = cum_ret.cummax()
        drawdown = (cum_ret / running_max - 1).min()

        # Calmar ratio
        calmar = ann_return / abs(drawdown) if drawdown != 0 else np.nan

        metrics[col] = {
            "Ann.Return": ann_return,
            "Ann.Volatility": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDD": drawdown,
            "Calmar": calmar,
        }

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "Portfolio"
    return metrics_df


