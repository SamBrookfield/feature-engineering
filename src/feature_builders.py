"""
feature_builders.py
===================
Generalised feature-building utilities for tabular ML projects.

Designed to pair with feature_design.py (FeatureRegistry, FeatureSpec,
FeatureSource, build_from_registry). Provides:

  - sanitise_columns      : make column names safe for any ML library
  - cast_bools            : bool → int8 to prevent object-dtype after get_dummies
  - downcast_dtypes       : halve memory usage by shrinking numeric dtypes
  - encode_categoricals   : label-encode binary cats, one-hot the rest
  - drop_high_null_cols   : remove columns above a null-rate threshold
  - build_features        : full pipeline — registry compute → encode → null-handle
  - aggregate_to_grain    : aggregate a child table to a parent grain with prefixing
  - assemble              : left-join a list of aggregated tables onto a spine

Typical usage
-------------
    from src.feature_builders import sanitise_columns, build_features, downcast_dtypes

    df_clean    = sanitise_columns(raw_df)
    df_features = build_features(
        df          = df_clean,
        registry    = REGISTRY,
        source      = FeatureSource.APPLICATION,   # or pass None to skip registry
        null_thresh = 0.50,
        fill_nulls  = False,
        sampling_ids = None,
        id_col      = 'SK_ID_CURR',
    )
    df_features = downcast_dtypes(df_features)

Two-step aggregation (child table with no direct link to spine):
    child_agg = aggregate_to_grain(
        df        = child_df,
        group_col = 'SK_ID_BUREAU',
        agg_dict  = {'MONTHS_BALANCE': ['count', 'mean']},
        prefix    = 'BB_',
    )
    # then join child_agg → bureau → spine manually, or nest calls

Single-step aggregation (child table has spine ID directly):
    bureau_agg = aggregate_to_grain(
        df        = bureau_df,
        group_col = 'SK_ID_CURR',
        agg_dict  = {c: ['mean', 'max', 'min'] for c in num_cols},
        prefix    = 'BUREAU_',
    )
    final = assemble(
        spine        = app_features,
        supplements  = [(bureau_agg, 'bureau')],
        join_key     = 'SK_ID_CURR',
    )
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional — only required if caller passes a registry / source
try:
    from src.feature_design import FeatureRegistry, FeatureSource, build_from_registry
    _REGISTRY_AVAILABLE = True
except ImportError:
    _REGISTRY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# DTYPE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def sanitise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace any character that is not alphanumeric or underscore with '_'.
    Returns a copy — does not mutate the input.

    Safe for LightGBM, XGBoost, sklearn, and most SQL-derived column names.

    Example
    -------
    >>> df.columns = ['a b', 'c[0]', 'd-e']
    >>> sanitise_columns(df).columns.tolist()
    ['a_b', 'c_0_', 'd_e']
    """
    out = df.copy()
    out.columns = [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in out.columns]
    return out


def cast_bools(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all bool columns to int8.
    Prevents object dtype after pd.get_dummies when bools are mixed in.
    Returns a copy.
    """
    out = df.copy()
    bool_cols = out.select_dtypes(include='bool').columns
    out[bool_cols] = out[bool_cols].astype(np.int8)
    return out


def downcast_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest type that preserves values:
      - float64  → float32
      - int64/32 → smallest int that fits (int8 if binary 0/1)
      - bool     → int8

    Typically halves memory usage on wide feature DataFrames.

    Parameters
    ----------
    df      : input DataFrame
    verbose : print before/after memory usage

    Returns
    -------
    Downcasted copy of df.
    """
    mem_before = df.memory_usage(deep=True).sum() / 1024 ** 2
    out = df.copy()

    for col in out.columns:
        dtype = out[col].dtype

        if dtype == bool:
            out[col] = out[col].astype(np.int8)

        elif dtype == np.float64:
            out[col] = pd.to_numeric(out[col], downcast='float')

        elif dtype in (np.int64, np.int32):
            if out[col].dropna().isin([0, 1]).all():
                out[col] = out[col].astype(np.int8)
            else:
                out[col] = pd.to_numeric(out[col], downcast='integer')

    if verbose:
        mem_after = out.memory_usage(deep=True).sum() / 1024 ** 2
        print("=" * 80)
        print("DOWNCAST DTYPES")
        print(f"  Before : {mem_before:.1f} MB")
        print(f"  After  : {mem_after:.1f} MB")
        print(f"  Saving : {(1 - mem_after / mem_before) * 100:.1f}%")

    return out


# ══════════════════════════════════════════════════════════════════════════════
# ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def encode_categoricals(
    df: pd.DataFrame,
    max_binary_card: int = 2,
    dummy_na: bool = False,
) -> pd.DataFrame:
    """
    Encode object-dtype columns:
      - Binary (nunique <= max_binary_card) : label-encode in place
      - All others                          : one-hot via pd.get_dummies

    Parameters
    ----------
    df              : input DataFrame
    max_binary_card : columns with this many or fewer unique values are
                      label-encoded rather than one-hotted (default 2)
    dummy_na        : whether to add a NaN indicator column in get_dummies

    Returns
    -------
    Encoded copy of df.
    """
    from sklearn.preprocessing import LabelEncoder

    out = df.copy()
    le = LabelEncoder()

    cat_cols = [c for c in out.columns if out[c].dtype == 'object']
    binary_cols = [
        c for c in cat_cols
        if out[c].nunique(dropna=False) <= max_binary_card
    ]
    for col in binary_cols:
        out[col] = le.fit_transform(out[col].astype(str))

    # Remaining categoricals → one-hot
    out = pd.get_dummies(out, dummy_na=dummy_na)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# NULL HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def drop_high_null_cols(
    df: pd.DataFrame,
    threshold: float = 0.50,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Drop columns where the fraction of nulls exceeds `threshold`.

    Parameters
    ----------
    df        : input DataFrame
    threshold : null rate above which a column is dropped (0–1, default 0.5)
    verbose   : print count of dropped columns

    Returns
    -------
    Filtered copy of df.
    """
    to_drop = [c for c in df.columns if df[c].isnull().mean() > threshold]
    if verbose and to_drop:
        print(f"  Dropping {len(to_drop)} columns "
              f"(>{threshold * 100:.0f}% null)")
    return df.drop(columns=to_drop)


# ══════════════════════════════════════════════════════════════════════════════
# SENTINEL REPLACEMENT
# ══════════════════════════════════════════════════════════════════════════════

def replace_sentinels(
    df: pd.DataFrame,
    columns: List[str],
    sentinel: Any = 365243,
    replace_with: Any = np.nan,
) -> pd.DataFrame:
    """
    Replace a known sentinel value in specific columns with a substitute.

    Useful for datasets that encode "missing" or "not applicable" as a magic
    number (e.g. 365243 for DAYS columns in Home Credit data).

    Parameters
    ----------
    df           : input DataFrame
    columns      : list of column names to apply replacement to
    sentinel     : the value to replace (default 365243)
    replace_with : replacement value (default np.nan)

    Returns
    -------
    Copy of df with replacements applied.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].replace(sentinel, replace_with)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_features(
    df: pd.DataFrame,
    registry=None,
    source=None,
    id_col: Optional[str] = None,
    sampling_ids=None,
    sentinel_cols: Optional[List[str]] = None,
    sentinel_value: Any = 365243,
    null_thresh: float = 0.50,
    fill_nulls: bool = False,
    encode: bool = True,
    max_binary_card: int = 2,
    dummy_na: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    General-purpose feature pipeline for a single-grain DataFrame.

    Steps
    -----
    1. Optionally subset rows to sampling_ids
    2. Replace sentinel values in specified columns
    3. Registry-driven feature computation (if registry + source provided)
    4. Encode categoricals (label-encode binary, one-hot the rest)
    5. Cast bools to int8, sanitise column names
    6. Drop columns above null_thresh, optionally fill remaining nulls

    Parameters
    ----------
    df             : raw input DataFrame at the target grain
    registry       : FeatureRegistry instance (optional)
    source         : FeatureSource enum for registry lookup (optional)
    id_col         : name of the ID column used for sampling (e.g. 'SK_ID_CURR')
    sampling_ids   : iterable of IDs to subset to; None = use all rows
    sentinel_cols  : column names where sentinel replacement should be applied
    sentinel_value : the sentinel magic number to replace with NaN (default 365243)
    null_thresh    : drop columns with null rate above this (0–1, default 0.50)
    fill_nulls     : if True, fill remaining nulls with column median
    encode         : if True, encode categorical columns (default True)
    max_binary_card: label-encode categoricals with ≤ this many unique values
    dummy_na       : pass through to pd.get_dummies
    verbose        : print progress and shape info

    Returns
    -------
    Processed DataFrame at the same grain.

    Notes
    -----
    - Columns are sanitised (special chars → '_') before return.
    - Caller is responsible for any aggregation if this is a child table.
    - To skip registry, pass registry=None or source=None.
    """
    out = df.copy()

    # ── 1. SAMPLING ───────────────────────────────────────────────────────────
    if sampling_ids is not None and id_col is not None:
        out = out[out[id_col].isin(sampling_ids)].copy()
        if verbose:
            print(f"  Sampled to {len(out):,} rows")
    elif sampling_ids is not None and id_col is None:
        warnings.warn(
            "sampling_ids provided but id_col is None — sampling skipped."
        )

    # ── 2. SENTINEL REPLACEMENT ───────────────────────────────────────────────
    if sentinel_cols:
        out = replace_sentinels(out, sentinel_cols, sentinel_value, np.nan)

    # ── 3. REGISTRY-DRIVEN FEATURES ───────────────────────────────────────────
    if registry is not None and source is not None and _REGISTRY_AVAILABLE:
        out = build_from_registry(out, source, registry, verbose=verbose)
    elif registry is not None and source is None:
        warnings.warn(
            "registry provided but source is None — registry step skipped."
        )

    # ── 4. ENCODE CATEGORICALS ────────────────────────────────────────────────
    if encode:
        out = encode_categoricals(out, max_binary_card=max_binary_card,
                                  dummy_na=dummy_na)

    # ── 5. DTYPE CLEANUP ──────────────────────────────────────────────────────
    out = cast_bools(out)
    out = sanitise_columns(out)

    # ── 6. NULL HANDLING ──────────────────────────────────────────────────────
    out = drop_high_null_cols(out, threshold=null_thresh, verbose=verbose)

    if fill_nulls:
        out = out.fillna(out.median(numeric_only=True))

    if verbose:
        print("=" * 80)
        print(f"  build_features output shape : {out.shape}")
        print(f"  Nulls remaining             : {out.isnull().sum().sum()}")

    return out


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_to_grain(
    df: pd.DataFrame,
    group_col: str,
    agg_dict: Dict[str, Union[str, List[str]]],
    prefix: str = '',
    drop_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Aggregate a child-grain DataFrame to a parent grain, with column prefixing.

    Parameters
    ----------
    df        : input DataFrame at the child grain
    group_col : column to group by (becomes the index / join key)
    agg_dict  : {column: agg_func_or_list} passed directly to groupby.agg()
                e.g. {'AMOUNT': ['mean', 'max'], 'COUNT': 'sum'}
    prefix    : string prepended to all output column names (e.g. 'BUREAU_')
    drop_cols : columns to drop from df before aggregating (e.g. surrogate keys)
    verbose   : print output shape and null count

    Returns
    -------
    Aggregated DataFrame with flattened, prefixed column names.
    group_col is reset to a regular column (not the index).

    Example
    -------
    bureau_agg = aggregate_to_grain(
        df        = bureau_df,
        group_col = 'SK_ID_CURR',
        agg_dict  = {c: ['mean', 'max', 'min'] for c in num_cols},
        prefix    = 'BUREAU_',
        drop_cols = ['SK_ID_BUREAU'],
    )
    """
    out = df.copy()

    if drop_cols:
        out = out.drop(columns=[c for c in drop_cols if c in out.columns],
                       errors='ignore')

    agg = out.groupby(group_col).agg(agg_dict)

    # Flatten MultiIndex columns produced by list aggregations
    if isinstance(agg.columns, pd.MultiIndex):
        agg.columns = [
            prefix + '_'.join(str(p) for p in col).upper()
            for col in agg.columns
        ]
    else:
        agg.columns = [prefix + str(c).upper() for c in agg.columns]

    agg = sanitise_columns(agg.reset_index())

    if verbose:
        print("=" * 80)
        print(f"  aggregate_to_grain → {agg.shape}  "
              f"(group={group_col}, prefix='{prefix}')")
        print(f"  Nulls : {agg.isnull().sum().sum()}")

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def assemble(
    spine: pd.DataFrame,
    supplements: List[Tuple[pd.DataFrame, str]],
    join_key: str = 'id',
    fill_nulls: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Left-join a list of aggregated DataFrames onto a spine DataFrame.

    Parameters
    ----------
    spine        : base DataFrame (one row per entity, e.g. per applicant)
    supplements  : list of (df, label) tuples — df must share join_key with spine
    join_key     : column name to join on (default 'id')
    fill_nulls   : if True, fill NaNs introduced by each join with column median
    verbose      : print shape after each join

    Returns
    -------
    Spine with all supplements left-joined on, deduplicated columns removed.

    Notes
    -----
    - Duplicate columns (excluding join_key) are warned about and dropped from
      the supplement before merging, preserving the spine version.
    - After merging, any remaining duplicate column names (e.g. from edge cases)
      are deduplicated by keeping the first occurrence.
    """
    out = spine.copy()
    spine_cols = set(out.columns)

    if verbose:
        print("=" * 80)
        print(f"  assemble: spine {out.shape}, {len(supplements)} supplements")

    for df, label in supplements:

        # Collision guard — drop columns already present in out (except join key)
        overlap = [c for c in df.columns if c in set(out.columns) and c != join_key]
        if overlap:
            warnings.warn(
                f"[{label}] dropping {len(overlap)} duplicate column(s): {overlap}"
            )
            df = df.drop(columns=overlap)

        before = out.shape[1]
        out = out.merge(df, on=join_key, how='left')

        # Deduplicate any remaining name clashes (keep first occurrence)
        out = out.loc[:, ~out.columns.duplicated(keep='first')]

        added = out.shape[1] - before
        new_cols = [c for c in out.columns if c not in spine_cols and c != join_key]

        if fill_nulls and new_cols:
            medians = out[new_cols].median(numeric_only=True)
            out[new_cols] = out[new_cols].fillna(medians)

        spine_cols = set(out.columns)

        if verbose:
            print(f"  {label:<30} +{added:>4} columns → {out.shape}")

    if verbose:
        print("=" * 80)
        print(f"  Final shape     : {out.shape}")
        print(f"  Nulls remaining : {out.isnull().sum().sum()}")

    return out