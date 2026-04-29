import pandas as pd
import numpy as np
from typing import Optional, Callable, List

from src.feature_design import FeatureRegistry

def feature_audit(df: pd.DataFrame,
                  target: Optional[pd.Series] = None,
                  registry: Optional['FeatureRegistry'] = None,
                  flag_range: float = 500,
                  flag_skew: float = 10,
                  flag_null: float = 0.15,
                  tag_registry: bool = False,
                  run_correlation: bool = False,
                  correlation_threshold: float = 0.95,
                  run_lgb: bool = False,
                  lgb_n_estimators: int = 300,
                  lgb_top_n: Optional[int] = None,
                  filter_tags: Optional[List[str]] = None) -> pd.DataFrame:

    import warnings
    from scipy.stats import pointbiserialr

    num_cols    = df.select_dtypes(include=np.number).columns.tolist()
    binary_cols = [c for c in num_cols if df[c].dropna().isin([0, 1]).all()]
    cont_cols   = [c for c in num_cols if c not in binary_cols]

    pos = df[target == 1] if target is not None else None
    neg = df[target == 0] if target is not None else None

    rows = []

    for col in num_cols:
        s        = df[col].dropna()
        null_pct = df[col].isnull().mean()
        if len(s) == 0:
            continue

        is_binary = col in binary_cols
        row = {
            'feature'  : col,
            'type'     : 'binary' if is_binary else 'continuous',
            'null_pct' : round(null_pct * 100, 1),
            'mean'     : round(s.mean(), 4),
            'std'      : round(s.std(),  4),
            'min'      : round(s.min(),  3),
            'p99'      : round(s.quantile(.99), 3),
            'max'      : round(s.max(),  3),
            'skew'     : round(s.skew(), 2),
            'range'    : round(s.max() - s.min(), 1),
            'zero_pct' : round((df[col] == 0).mean() * 100, 1),
        }

        if target is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    r, p = pointbiserialr(df[col].fillna(s.median()), target)
                    row['target_corr'] = round(r, 4)
                    row['p_value']     = round(p, 4)
                except Exception:
                    row['target_corr'] = None
                    row['p_value']     = None

            std = df[col].std()
            if std > 0 and pos is not None and neg is not None:
                row['effect_size'] = round(
                    abs(pos[col].mean() - neg[col].mean()) / std, 4
                )
            else:
                row['effect_size'] = None

            if is_binary:
                tr_pos = target[df[col] == 1].mean()
                tr_neg = target[df[col] == 0].mean()
                row['target_rate_pos'] = round(tr_pos * 100, 1)
                row['target_rate_neg'] = round(tr_neg * 100, 1)
                row['lift']            = round(tr_pos / target.mean(), 3) if target.mean() > 0 else None
            else:
                row['target_rate_pos'] = None
                row['target_rate_neg'] = None
                row['lift']            = None

        flags = []
        if row['range']     > flag_range: flags.append('EXTREME_RANGE')
        if abs(row['skew']) > flag_skew:  flags.append('HIGH_SKEW')
        if null_pct         > flag_null:  flags.append('HIGH_NULL')
        if row['std']       < 0.001:      flags.append('NEAR_CONSTANT')
        if is_binary:
            prev = s.mean()
            if prev < 0.02:               flags.append('LOW_PREVALENCE')
            if prev > 0.98:               flags.append('HIGH_PREVALENCE')
        if target is not None:
            if row.get('p_value') is not None and row['p_value'] > 0.05:
                flags.append('NOT_SIG')

        row['flags'] = flags
        rows.append(row)

    # ── CORRELATION PASS ──────────────────────────────────────────────────────
    if run_correlation:
        print("  Running correlation filter...")
        num_df      = df.select_dtypes(include=np.number)
        corr_matrix = num_df.corr().abs()
        upper       = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        variances   = num_df.var()
        corr_drop   = set()
        for col in upper.columns:
            for other in upper.index[upper[col] > correlation_threshold].tolist():
                if col in corr_drop or other in corr_drop:
                    continue
                corr_drop.add(col if variances[col] < variances[other] else other)

        for row in rows:
            if row['feature'] in corr_drop:
                if 'CORRELATED' not in row['flags']:
                    row['flags'].append('CORRELATED')
        print(f"  Correlation pass: {len(corr_drop)} features tagged CORRELATED")

    # ── LGB PASS ──────────────────────────────────────────────────────────────
    if run_lgb and target is not None:
            import lightgbm as lgb
            print("  Running quick LGB importance pass...")
            X_lgb = df.select_dtypes(include=np.number).fillna(
                df.select_dtypes(include=np.number).median()
                ).drop(columns=corr_drop)
            
            quick_lgb = lgb.LGBMClassifier(
                n_estimators  = lgb_n_estimators,   # was 100
                max_depth     = 4,     # was 2
                num_leaves     = 15,   # was 2
                learning_rate = 0.05,
                class_weight  = 'balanced',
                random_state  = 42,
                verbosity     = -1,
                n_jobs        = -1,
            )
            
            quick_lgb.fit(X_lgb, target)

            lgb_imp = pd.Series(
                quick_lgb.feature_importances_, index=X_lgb.columns
                ).sort_values(ascending=False)

            top_imp     = set(lgb_imp.head(lgb_top_n).index) if lgb_top_n else set(lgb_imp.index)
            zero_imp    = set(lgb_imp[lgb_imp == 0].index)
            not_top_imp = set(lgb_imp.index) - top_imp  # everyone outside top N regardless of score

            for row in rows:
                feat = row['feature']
                if feat in zero_imp:
                    if 'ZERO_LGB_IMPORTANCE' not in row['flags']:
                        row['flags'].append('ZERO_LGB_IMPORTANCE')
                if feat in not_top_imp:                          # note: if, not elif
                    if 'DROPPED_LGB_TOP_N' not in row['flags']:
                        row['flags'].append('DROPPED_LGB_TOP_N')
                elif feat in top_imp:
                    if 'LGB_TOP_N' not in row['flags']:
                        row['flags'].append('LGB_TOP_N')

            print(f"  LGB pass: {len(top_imp)} in top {lgb_top_n}, "
                f"{len(zero_imp)} zero importance, "
                f"{len(not_top_imp)} dropped (includes zero)")
            

    if run_correlation and run_lgb and lgb_top_n and target is not None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        top_cols = [
            row['feature'] for row in rows
            if 'LGB_TOP_N' in row['flags']
            and row['feature'] in df.columns
        ]

        if 0 < len(top_cols) < 50:
            corr = df[top_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig, ax = plt.subplots(
                figsize=(max(10, len(top_cols) * 0.4),
                         max(8,  len(top_cols) * 0.4))
            )
            sns.heatmap(
                corr, mask=mask, cmap='RdYlGn_r', center=0,
                vmin=-1, vmax=1, annot=len(top_cols) <= 30,
                fmt='.2f', linewidths=0.3, ax=ax,
                cbar_kws={'shrink': 0.7}
            )
            ax.set_title(
                f'Feature Correlation — top {len(top_cols)} by LGB importance'
            )
            plt.tight_layout()
            plt.show()
        elif len(top_cols) >= 50:
            print(f"  Skipping correlation plot — {len(top_cols)} features > 50")

    # ── TAG FILTER ────────────────────────────────────────────────────────────
    # If filter_tags provided, restrict rows to features that have ALL
    # of the specified tags in either their registry tags or their audit flags.
    # Useful for inspecting a specific cohort without re-running the full audit.
    if filter_tags:
        filtered_rows = []
        for row in rows:
            audit_flags  = {f.lower() for f in row['flags']}
            reg_tags     = set(registry.get(row['feature']).tags) \
                           if registry and row['feature'] in registry else set()
            all_tags     = audit_flags | reg_tags
            if all(t.lower() in all_tags for t in filter_tags):
                filtered_rows.append(row)
        rows = filtered_rows
        print(f"  filter_tags={filter_tags}: {len(rows)} features match")


    sort_col = 'effect_size' if target is not None else 'range'
    audit_df = (
        pd.DataFrame(rows)
        .sort_values(sort_col, ascending=False, na_position='last')
        .reset_index(drop=True)
    )

    if tag_registry and registry is not None:
        tagged = 0
        for _, row in audit_df.iterrows():
            name  = row['feature']
            flags = row['flags']
            if not flags or name not in registry:
                continue
            spec = registry.get(name)
            for flag in flags:
                tag = flag.lower()
                if tag not in spec.tags:
                    spec.tags.append(tag)
            tagged += 1
        print(f"  Registry tags updated for {tagged} features")

    flagged = audit_df[audit_df['flags'].apply(len) > 0]
    print(f"  Audited: {len(audit_df)}  |  Flagged: {len(flagged)}  "
          f"|  Binary: {len(binary_cols)}  |  Continuous: {len(cont_cols)}")

    return audit_df


def display_audit(audit_df: pd.DataFrame,
                  top_n: int = 30) -> None:
    """
    Print continuous and binary sections separately, then a flag summary.

    Parameters
    ----------
    audit_df : output of feature_audit()
    top_n    : rows to show per section
    """

    cont   = audit_df[audit_df['type'] == 'continuous'].copy()
    binary = audit_df[audit_df['type'] == 'binary'].copy()

    # ── CONTINUOUS ────────────────────────────────────────────────────────────
    cont_cols = ['feature', 'null_pct', 'zero_pct', 'std', 'skew']
    if 'target_corr' in cont.columns:
        cont_cols += ['target_corr', 'p_value', 'effect_size']
    cont_cols += ['flags']

    cont_display = cont[cont_cols].head(top_n).copy()
    cont_display['flags'] = cont_display['flags'].apply(
        lambda f: ', '.join(f) if f else ''
    )

    print("=" * 120)
    print(f"  CONTINUOUS FEATURES — top {min(top_n, len(cont))} by effect size")
    print("=" * 120)
    print(cont_display.to_string(index=False))

    # ── BINARY ────────────────────────────────────────────────────────────────
    bin_cols = ['feature', 'null_pct', 'zero_pct', 'mean']
    if 'target_rate_pos' in binary.columns:
        bin_cols += ['target_rate_pos', 'target_rate_neg', 'lift', 'effect_size']
    bin_cols += ['flags']

    binary_display = binary[bin_cols].head(top_n).copy()
    binary_display.rename(columns={'mean': 'prevalence'}, inplace=True)
    binary_display['flags'] = binary_display['flags'].apply(
        lambda f: ', '.join(f) if f else ''
    )

    print("=" * 120)
    print(f"  BINARY FEATURES — top {min(top_n, len(binary))} by effect size")
    print("=" * 120)
    print(binary_display.to_string(index=False))

    # ── FLAG SUMMARY ──────────────────────────────────────────────────────────
    from collections import Counter

    all_flags = [
        flag
        for flags in audit_df['flags']
        for flag in flags
    ]
    flag_counts = Counter(all_flags)

    print("=" * 60)
    print("  FLAG SUMMARY")
    print("=" * 60)
    if flag_counts:
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"  {flag:<25} {count:>5} features")
    else:
        print("  No flags raised.")
    print("=" * 60)


def population_stability_index(reference: pd.Series,
                                current: pd.Series,
                                n_bins: int = 10) -> float:
    """
    PSI between a reference distribution (training) and current (new data).
    PSI < 0.1  : stable
    PSI 0.1-0.2: moderate shift — investigate
    PSI > 0.2  : significant drift — consider retraining
    """
    ref_clean = reference.dropna()
    cur_clean = current.dropna()

    bins      = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    bins[0]   = -np.inf
    bins[-1]  =  np.inf

    ref_counts = np.histogram(ref_clean, bins=bins)[0]
    cur_counts = np.histogram(cur_clean, bins=bins)[0]

    ref_pct = np.where(ref_counts == 0, 0.0001, ref_counts / len(ref_clean))
    cur_pct = np.where(cur_counts == 0, 0.0001, cur_counts / len(cur_clean))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 4)


def feature_select(model_df: pd.DataFrame,
                    target_var: str,
                    registry: 'FeatureRegistry',
                    DROP_TAGS = [
                                'high_null',
                                'near_constant',
                                'low_prevalence',
                                'high_prevalence',
                                'correlated',
                                'zero_lgb_importance',
                                'dropped_lgb_top_n',
                                ]) -> list:
    """
    Tag-driven feature selection. Reads drop decisions from registry tags
    set by feature_audit(). Run feature_audit first with:
        run_correlation=True, run_lgb=True, lgb_top_n=N, tag_registry=True

    Drop order follows ordering of DROP_TAGS

    Returns
    -------
    List of selected feature names, tagged 'selected' in registry.
    """
    

    all_features = [c for c in model_df.columns if c != target_var]
    remaining    = set(all_features)

    print("="*100)
    print(f"FEATURE SELECTION")
    print("="*100)
    print(f"  Starting: {len(remaining)} features")

    for tag in DROP_TAGS:
        to_drop = {s.name for s in registry.by_tag(tag)} & remaining
        remaining -= to_drop
        print(f"  {tag:<30} dropped {len(to_drop):>5}  →  kept {len(remaining):>5}")

    top_features = [f for f in all_features if f in remaining]

    for feat in top_features:
        if feat in registry:
            spec = registry.get(feat)
            if 'selected' not in spec.tags:
                spec.tags.append('selected')

    return top_features