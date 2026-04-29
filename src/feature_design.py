import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class FeatureFamily(Enum):
    CREDIT_RISK  = "credit_risk"
    REPAYMENT    = "repayment"
    INCOME_BURDEN= "income_burden"
    BEHAVIOURAL  = "behavioural"
    DEMOGRAPHIC  = "demographic"
    CROSS_SOURCE = "cross_source"
    ENGINEERED   = "engineered"

class FeatureSource(Enum):
    APPLICATION  = "application_train"
    BUREAU       = "bureau"
    BUREAU_BAL   = "bureau_balance"
    PREV_APP     = "previous_application"
    INSTALLMENTS = "installments_payments"
    CREDIT_CARD  = "credit_card_balance"
    POS_CASH     = "pos_cash_balance"
    DERIVED      = "derived"

# ══════════════════════════════════════════════════════════════════════════════
# DIRECTION ENUM — replaces free-string direction field
# ══════════════════════════════════════════════════════════════════════════════

class Direction(Enum):
    POSITIVE  = "positive"    # higher value → higher risk / target
    NEGATIVE  = "negative"    # higher value → lower risk / target
    NONLINEAR = "nonlinear"   # relationship is non-monotonic
    UNKNOWN   = "unknown"     # not yet characterised

    @property
    def label(self) -> str:
        """Human-readable label for LLM prompts and reports."""
        return {
            Direction.POSITIVE:  "higher values increase risk",
            Direction.NEGATIVE:  "higher values decrease risk",
            Direction.NONLINEAR: "relationship is non-linear",
            Direction.UNKNOWN:   "direction not characterised",
        }[self]

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SPEC
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureSpec:
    """
    Declarative definition of one feature.
    family and source are plain strings — valid values are defined by the
    RegistryConfig passed to the FeatureRegistry, not hardcoded here.
    """
    name:        str
    family:      str                        # e.g. "credit_risk", "engagement"
    source:      str                        # e.g. "bureau", "clickstream"
    description: str
    formula:     Optional[str]              = None
    compute:     Optional[Callable]         = None
    actionable:  bool                       = True
    direction:   Direction                  = Direction.UNKNOWN
    tags:        List[str]                  = field(default_factory=list)

# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY CONFIG — project-level vocabulary, injected at instantiation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegistryConfig:
    """
    Defines the valid vocabulary for a specific project's registry.
    Pass an instance to FeatureRegistry() to enforce project-level constraints.

    Parameters
    ----------
    families : set of valid family strings, e.g. {"credit_risk", "repayment"}
    sources  : set of valid source strings, e.g. {"application_train", "bureau"}
    strict   : if True, raise on invalid family/source; if False, warn and allow

    Example — credit project
    ------------------------
    CREDIT_CONFIG = RegistryConfig(
        families = {
            "credit_risk", "repayment", "income_burden",
            "behavioural", "demographic", "cross_source", "engineered",
        },
        sources = {
            "application_train", "bureau", "bureau_balance",
            "previous_application", "installments_payments",
            "credit_card_balance", "pos_cash_balance", "derived",
        },
    )

    Example — churn project
    -----------------------
    CHURN_CONFIG = RegistryConfig(
        families = {"engagement", "product_usage", "support", "billing", "demographic"},
        sources  = {"crm", "clickstream", "support_tickets", "billing_system", "derived"},
    )
    """
    families: set = field(default_factory=set)
    sources:  set = field(default_factory=set)
    strict:   bool = False   # True = raise, False = warn

    def validate_family(self, family: str) -> None:
        if self.families and family not in self.families:
            msg = f"Family '{family}' not in config families: {self.families}"
            if self.strict:
                raise ValueError(msg)
            else:
                import warnings
                warnings.warn(msg)

    def validate_source(self, source: str) -> None:
        if self.sources and source not in self.sources:
            msg = f"Source '{source}' not in config sources: {self.sources}"
            if self.strict:
                raise ValueError(msg)
            else:
                import warnings
                warnings.warn(msg)

# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class FeatureRegistry:
    """
    Central catalogue of all features in the project.

    Key methods
    -----------
    register(spec)              : add or overwrite one FeatureSpec
    register_many(specs)        : add a list of FeatureSpecs
    bulk_register(columns, map) : register a list of column names by prefix match
    get(name)                   : retrieve one FeatureSpec by name
    by_source(source)           : all specs for a given FeatureSource
    by_family(family)           : all specs for a given FeatureFamily
    by_tag(tag)                 : all specs with a given tag
    actionable_features()       : names of features the user can change
    to_dataframe()              : export full catalogue as DataFrame
    llm_name_map()              : {col_name: plain_english} for LLM prompts
    """

    def __init__(self, config: Optional[RegistryConfig] = None):
        self._specs:  dict[str, FeatureSpec] = {}
        self._config: RegistryConfig         = config or RegistryConfig()

    # REGISTRATION 

    def register(self, spec: FeatureSpec) -> None:
        self._config.validate_family(spec.family)
        self._config.validate_source(spec.source)
        self._specs[spec.name] = spec

    def register_many(self, specs: List[FeatureSpec]) -> None:
        for s in specs:
            self.register(s)

    def bulk_register(self,
                      columns: List[str],
                      prefix_map: dict,
                      tag: str = 'bulk_registered') -> None:
        """
        Register a list of columns by matching against a prefix map.

        prefix_map format:
            { 'PREFIX_': (FeatureFamily, FeatureSource, description, actionable, direction) }

        More specific (longer) prefixes take priority — sorted automatically.
        Columns that don't match any prefix are skipped with a warning.
        """
        sorted_prefixes  = sorted(prefix_map.keys(), key=len, reverse=True)
        unregistered     = []

        for col in columns:
            matched = False
            for prefix in sorted_prefixes:
                if col.startswith(prefix):
                    family, source, desc, actionable, direction = prefix_map[prefix]
                    # only register if not already present — don't overwrite
                    # explicit registrations with bulk ones
                    if col not in self._specs:
                        self.register(FeatureSpec(
                            name        = col,
                            family      = family,
                            source      = source,
                            description = desc,
                            formula     = None,
                            compute     = None,
                            actionable  = actionable,
                            direction   = direction,
                            tags        = [prefix.lower().rstrip('_'), tag]
                        ))
                    matched = True
                    break
            if not matched:
                unregistered.append(col)

        if unregistered:
            print("="*100)
            # In bulk_register, replace the unregistered print block with:
            truly_unregistered = [c for c in unregistered if c not in self._specs]
            already_registered = [c for c in unregistered if c in self._specs]

            if already_registered:
                print(f"bulk_register: {len(already_registered)} columns already explicitly registered — skipped (correct)")
            if truly_unregistered:
                print(f"bulk_register: {len(truly_unregistered)} columns matched no prefix and are unregistered:")
                print(f"{truly_unregistered[:10]}" + (' ...' if len(truly_unregistered) > 10 else ''))

    # UPDATE
    
    def update(self, name: str, **kwargs) -> None:
        """
        Update fields on an existing FeatureSpec by name.
        Accepts any FeatureSpec field as a keyword argument.
        Tags are merged (not replaced) — pass tags=['new_tag'] to add without losing existing.

        Example
        -------
        REGISTRY.update('EXT_SOURCE_MIN',
            description = 'Lowest of the three external credit scores across the window',
            direction   = 'lower_is_riskier',
            tags        = ['final_selection']
        )
        """
        if name not in self._specs:
            raise KeyError(f"Feature '{name}' not in registry — register it first")

        spec = self._specs[name]

        for field, value in kwargs.items():
            if field == 'tags':
                # merge tags rather than replace
                for tag in value:
                    if tag not in spec.tags:
                        spec.tags.append(tag)
            elif hasattr(spec, field):
                setattr(spec, field, value)
            else:
                raise ValueError(f"FeatureSpec has no field '{field}'")

    def bulk_update(self, updates: list) -> None:
        """
        Batch update. Each item is a dict with 'name' plus any FeatureSpec fields.

        Example
        -------
        REGISTRY.update_many([
            {'name': 'EXT_SOURCE_MIN', 'description': '...', 'tags': ['final_selection']},
            {'name': 'CREDIT_GOODS_RATIO', 'description': '...', 'tags': ['final_selection']},
        ])
        """
        for item in updates:
            name = item.pop('name')
            self.update(name, **item)
            item['name'] = name  # restore so the list isn't mutated


    # QUERYING 

    def get(self, name: str) -> FeatureSpec:
        if name not in self._specs:
            raise KeyError(f"Feature '{name}' not in registry")
        return self._specs[name]

    def by_source(self, source: str) -> List[FeatureSpec]:
        return [s for s in self._specs.values() if s.source == source]

    def by_family(self, family: str) -> List[FeatureSpec]:
        return [s for s in self._specs.values() if s.family == family]

    def by_tag(self, tag: str) -> List[FeatureSpec]:
        return [s for s in self._specs.values() if tag in s.tags]

    def actionable_features(self) -> List[str]:
        return [s.name for s in self._specs.values() if s.actionable]

    def computable_features(self, source: FeatureSource) -> List[FeatureSpec]:
        return [s for s in self.by_source(source) if s.compute is not None]

    # EXPORT / VISUALISE

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'name':        s.name,
            'family':      s.family,             # plain string, no .value needed
            'source':      s.source,
            'description': s.description,
            'formula':     s.formula,
            'actionable':  s.actionable,
            'direction':   s.direction.value,    # Direction enum -> string
            'direction_label': s.direction.label,
            'has_compute': s.compute is not None,
            'tags':        ', '.join(s.tags),
        } for s in self._specs.values()])

    def llm_name_map(self, include_direction: bool = False) -> dict:
        """
        Returns {col_name: description} for all registered features.
        Optionally appends direction label to each description.
        """
        result = {}
        for name, spec in self._specs.items():
            desc = spec.description if spec.description else name.replace('_', ' ').title()
            if include_direction and spec.direction != Direction.UNKNOWN:
                desc = f"{desc} ({spec.direction.label})"
            result[name] = desc
        return result

    def summary(self) -> None:
        df = self.to_dataframe()
        print(f"{'='*100}")
        print(f"Registry summary — {len(df)} features")
        print(f"{'='*100}")
        print(f"  By source:")
        for src, grp in df.groupby('source'):
            print(f"    {src:<30} {len(grp):>5}")
        print(f"  By family:")
        for fam, grp in df.groupby('family'):
            print(f"    {fam:<30} {len(grp):>5}")
        print(f"  Computable (have compute fn): {df['has_compute'].sum()}")
        print(f"  Actionable:                   {df['actionable'].sum()}")
        print(f"  Tagged engineered:            {len(self.by_tag('engineered'))}")

    def inspect(self,
                filter_tags: Optional[List[str]] = None,
                search_pattern: Optional[str] = None,
                match_columns: Optional[List[str]] = None,
                show_candidates: bool = True) -> None:
        """
        Display a broad view of registry features, optionally filtered.

        Parameters
        ----------
        filter_tags     : only show features with ALL of these tags
        search_pattern  : regex pattern — show features whose name matches,
                        OR show computable features whose aggregated column
                        names in match_columns match the pattern
        match_columns   : list of column names to search against (e.g. model_df.columns)
                        used with search_pattern to find aggregated descendants
        show_candidates : if True and search_pattern + match_columns given,
                        show matched column names alongside registry entry
        """
        import re
        if isinstance(filter_tags, str):
            filter_tags = [filter_tags]

        specs = list(self._specs.values())

        # Filter by tags
        if filter_tags:
            specs = [
                s for s in specs
                if all(t in s.tags for t in filter_tags)
            ]

        # Filter by pattern on name
        if search_pattern and not match_columns:
            pattern = re.compile(search_pattern, re.IGNORECASE)
            specs   = [s for s in specs if pattern.search(s.name)]

        # Pattern match against external column list (aggregated descendants)
        col_matches = {}
        if search_pattern and match_columns:
            pattern     = re.compile(search_pattern, re.IGNORECASE)
            all_cols    = set(match_columns)
            # first filter specs by pattern on name OR by having matching descendants
            matched_specs = []
            for s in specs:
                name_match = pattern.search(s.name)
                desc_match = sorted([
                    c for c in all_cols
                    if re.search(re.escape(s.name), c, re.IGNORECASE)
                ])
                if name_match or desc_match:
                    matched_specs.append(s)
                    col_matches[s.name] = desc_match
            specs = matched_specs

        if not specs:
            print("  No features matched.")
            return

        print("=" * 140)
        print(f"  Registry inspect — {len(specs)} features"
            + (f" | tags={filter_tags}" if filter_tags else "")
            + (f" | pattern='{search_pattern}'" if search_pattern else ""))
        print("=" * 140)
        print(f"  {'Name':<45} {'Family':<18} {'Dir':<20} {'Actionable':<10}  Tags")
        print(f"  {'-'*135}")

        for spec in sorted(specs, key=lambda s: s.source):
            tags    = ', '.join(spec.tags) if spec.tags else '—'
            act     = 'Yes' if spec.actionable else 'No'
            dirn    = spec.direction or '—'
            print(f"  {spec.name:<45} {spec.family:<18} {dirn:<20} {act:<10}  {tags}")

            if spec.description:
                print(f"  {'':10}   {spec.description[:90]}")

            if show_candidates and spec.name in col_matches and col_matches[spec.name]:
                selected = [c for c in col_matches[spec.name]
                            if c in self._specs and 'selected' in self._specs[c].tags]
                cands = col_matches[spec.name][:5]
                suffix = ' ...' if len(col_matches[spec.name]) > 5 else ''
                sel_str = f"  selected: {selected}" if selected else "  none selected"
                print(f"  {'':45}   aggregated → {cands}{suffix}")
                print(f"  {'':45}   {sel_str}")

            print()

    def __len__(self):   return len(self._specs)

    def __contains__(self, name): return name in self._specs

    def __repr__(self):  return f"FeatureRegistry({len(self)} features)"

# ══════════════════════════════════════════════════════════════════════════════
# BATCH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_from_registry(df: pd.DataFrame,
                         source: FeatureSource,
                         registry: FeatureRegistry,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Compute all registry features for a given source that have a compute function.
    Batches all results into a single pd.concat — one DataFrame copy regardless
    of feature count, faster than repeated column assignment in a loop.

    Parameters
    ----------
    df       : input DataFrame for this source
    source   : FeatureSource enum — filters registry to matching specs
    registry : FeatureRegistry instance
    verbose  : print count of features added and any failures

    Returns
    -------
    df with new computed columns appended
    """
    specs  = registry.computable_features(source)
    errors = []
    new_cols = {}

    for spec in specs:
        try:
            result = spec.compute(df)
            if isinstance(result, pd.Series):
                new_cols[spec.name] = result.values
            else:
                new_cols[spec.name] = result
        except Exception as e:
            errors.append((spec.name, str(e)))

    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        out    = pd.concat([df, new_df], axis=1)
    else:
        out = df.copy()

    if verbose:
        print(f"  build_from_registry [{source}]: "
              f"+{len(new_cols)} features"
              + (f", {len(errors)} failed" if errors else ""))
    if errors:
        for name, err in errors:
            print(f"    FAILED {name}: {err}")

    return out



# ══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ══════════════════════════════════════════════════════════════════════════════
"""

# 1. Bulk register all columns from training_df 
# training_df is your assembled model_df before feature selection
# training_df = assemble(...)

all_cols = [c for c in training_df.columns if c != 'TARGET']
REGISTRY.bulk_register(all_cols, PREFIX_MAP)

# 2. Print registry summary 
REGISTRY.summary()

# 3. Use build_from_registry in your application builder 
# Pass the raw application_train DataFrame — registry computes all
# features that have a compute function registered for APPLICATION source
app_features = build_from_registry(
    df       = application_train_clean,
    source   = FeatureSource.APPLICATION,
    registry = REGISTRY,
)
# app_features now has all original columns + EXT_SOURCE_MEAN,
# ANNUITY_INCOME_RATIO, AGE_YEARS etc. appended

# 4. Query by family 
income_features = REGISTRY.by_family(FeatureFamily.INCOME_BURDEN)
print([f.name for f in income_features])

# 5. Query engineered features after selection 
top = REGISTRY.by_tag('engineered')
print(f"Top features: {top}")

# 6. Get actionable features — for LLM recommendation filtering 
actionable = REGISTRY.actionable_features()
# In the LLM explainer, only surface recommendations for actionable features

# 7. Get plain-English name map — feed into LLM prompt 
name_map = REGISTRY.llm_name_map()
# name_map['ANNUITY_INCOME_RATIO'] -> 'Monthly loan repayment as a fraction of declared annual income'
# name_map['CC_UTILISATION_MEAN']  -> 'Credit card utilisation rate'

# 8. Export full catalogue as DataFrame 
catalogue = REGISTRY.to_dataframe()
catalogue.to_csv('outputs/feature_catalogue.csv', index=False)

# 9. After feature selection — check which engineered features survived 
# selected_features comes from select_features()
survived_top = [f for f in REGISTRY.by_tag('engineered') if f in selected_features]
dropped_top  = [f for f in REGISTRY.by_tag('engineered') if f not in selected_features]
print(f"Top features that survived selection: {survived_top}")
print(f"Top features dropped:                 {dropped_top}")

# 10. Get direction for a specific feature — used in evaluation 
spec = REGISTRY.get('CC_UTILISATION_MEAN')
print(spec.direction)    # 'higher_is_riskier'
print(spec.actionable)   # True
print(spec.description)  # 'Credit card utilisation rate'


# View all final_selection features with descriptions
REGISTRY.inspect(filter_tags=['final_selection'])

# View all computable features and their aggregated descendants in model_df
REGISTRY.inspect(
    search_pattern  = '.',        # matches everything
    match_columns   = list(model_df.columns),
    show_candidates = True,
    filter_tags     = None
)

# Find everything related to EXT_SOURCE and its aggregated forms
REGISTRY.inspect(
    search_pattern  = 'EXT_SOURCE',
    match_columns   = list(model_df.columns),
)

# Check which final_selection features are actionable
REGISTRY.inspect(filter_tags=['final_selection', 'actionable'])
# or
actionable_final = [
    s.name for s in REGISTRY.by_tag('final_selection')
    if s.actionable
]

"""