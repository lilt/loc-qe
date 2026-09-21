"""Challenge-set generation: candidate sampling, negative-sample injection, subsampling.

Pure-pandas ports of the internal Sisyphus jobs
``SegmentHistorySampleCandidates``, ``CreateNegativeChecklistSamples``, and
``SubsampleChallengeSetJob``. All sampling is seeded; given the same input
DataFrame (including row order) the output is deterministic.

Input rows need at least the columns ``language_pair``, ``source`` (source
text), ``translation`` (MT output), ``reference``, and
``translation_reference_chrfpp`` (chrF++ between translation and reference).
"""

import logging
import random

import numpy as np
import pandas as pd

from loccheck.constants import ChecklistCategory, category_applies_to_language_pair
from loccheck.helpers import AbstractChecklistHelper, FrenchNBSPChecklistHelper

logger = logging.getLogger(__name__)


def _sample_per_group(
    df: pd.DataFrame, by: list[str], n: int, random_seed: int
) -> pd.DataFrame:
    """Seeded per-group row sampling, capping each group at *n* rows."""
    parts = [
        group.sample(n=min(len(group), n), replace=False, random_state=random_seed)
        for _, group in df.groupby(by, sort=True)
    ]
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts)


# Column dtypes for CSV round-trips, so text/score columns keep their types.
COLUMN_TYPES_MAP = {
    "source": str,
    "positive": str,
    "negative": str,
    "reference": str,
    "translation": str,
    "score": np.float64,
    "translation_reference_chrfpp": np.float64,
    "domain": str,
    "language_pair": str,
}


def sample_candidates(
    df: pd.DataFrame,
    categories: list[ChecklistCategory],
    language_pair: str | None = None,  # format xx-yy; None = auto-detect from data
    samples_per_category: int | None = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Filter rows into per-category candidate pools.

    Each returned row is a copy of an input row with an added ``category``
    column; a row can appear once per category it qualifies for.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    if len(df) == 0:
        logger.warning("Input has no data rows")
        return pd.DataFrame(columns=df.columns)

    # Drop rows where required text columns are NaN (e.g. empty translations
    # from WMT data that become NaN after CSV round-tripping).
    _text_cols = [c for c in ("source", "translation", "reference") if c in df.columns]
    df = df.dropna(subset=_text_cols)

    df = df[
        df["translation_reference_chrfpp"] < 100
    ]  # prefer having a reference that's different from the positive

    if language_pair is not None:
        lang_pairs = [language_pair]
    else:
        lang_pairs = sorted(df["language_pair"].unique().tolist())

    collected_samples = pd.DataFrame()
    for category in categories:
        try:
            helper = AbstractChecklistHelper.from_category(
                category, random_seed=random_seed
            )
        except ValueError as e:
            logger.warning(f"Failed to load helper for category {category}: {e}")
            continue

        applicable_lps = [
            pair
            for pair in lang_pairs
            if category_applies_to_language_pair(category, pair)
        ]
        if not applicable_lps:
            continue

        df_lp = df[df["language_pair"].isin(applicable_lps)]
        if df_lp.empty:
            continue

        filter_mask = df_lp.apply(helper.candidate_condition, axis=1)
        df_filtered = df_lp[filter_mask].copy()

        if samples_per_category is not None and len(df_filtered) > samples_per_category:
            df_filtered = df_filtered.sample(
                n=samples_per_category,
                replace=False,
                random_state=random_seed,
            )
        df_filtered["category"] = category.value
        collected_samples = pd.concat([collected_samples, df_filtered])

    # If no samples were collected, create empty DataFrame with same structure as input
    if len(collected_samples) == 0:
        collected_samples = pd.DataFrame(columns=df.columns)

    return collected_samples


def create_negative_samples(
    df: pd.DataFrame,
    categories: list[ChecklistCategory],
    language_pair: str | None = None,  # format xx-yy; None = auto-detect from data
    samples_per_category: int | None = 50,
    random_seed: int = 42,
    max_samples_per_source: int | None = None,
    french_nbsp_regular_space_perturb_rate: float = 0.5,
    french_nbsp_strict_filter: bool = False,
) -> pd.DataFrame:
    """Turn candidate rows (from :func:`sample_candidates`) into minimal pairs.

    Each output row has ``src``, ``mt`` (positive), ``neg``, and ``ref``
    columns plus the metadata of the originating candidate row.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    expected_columns = ["source", "positive", "negative", "reference"]
    if len(df) == 0:
        logger.warning("Input has no candidate rows")
        return pd.DataFrame(columns=expected_columns)

    if language_pair is not None:
        lang_pairs = [language_pair]
    else:
        lang_pairs = sorted(df["language_pair"].unique().tolist())

    logger.info("Input: %d candidates across %d lang pairs", len(df), len(lang_pairs))

    negative_samples = pd.DataFrame(columns=expected_columns)
    for category in categories:
        applicable_lps = [
            pair
            for pair in lang_pairs
            if category_applies_to_language_pair(category, pair)
        ]
        if not applicable_lps:
            logger.info(
                "category=%s: skipped, no applicable lang pairs (data has %s)",
                category.value,
                lang_pairs,
            )
            continue
        try:
            if category == ChecklistCategory.french_nbsp:
                helper = FrenchNBSPChecklistHelper(
                    random_seed=random_seed,
                    regular_space_perturb_rate=french_nbsp_regular_space_perturb_rate,
                    strict_filter=french_nbsp_strict_filter,
                )
            else:
                helper = AbstractChecklistHelper.from_category(
                    category, random_seed=random_seed
                )
        except ValueError as e:
            logger.warning(f"Failed to load helper for category {category}: {e}")
            continue

        df_category = df[
            (df["category"] == category.value)
            & (df["language_pair"].isin(applicable_lps))
        ].copy()
        if len(df_category) == 0:
            logger.info(
                "category=%s: 0 input candidates (applicable lps: %s)",
                category.value,
                applicable_lps,
            )
            continue

        # Cap per-(lp, src) input candidates. WMT has many system outputs
        # per source segment; without this cap a single source dominates
        # a category and pref pairs end up near-duplicate.
        if max_samples_per_source is not None and "source" in df_category.columns:
            n_before_src_cap = len(df_category)
            df_category = _sample_per_group(
                df_category,
                by=["language_pair", "source"],
                n=max_samples_per_source,
                random_seed=random_seed,
            ).reset_index(drop=True)
            logger.info(
                "category=%s: per-source cap %d -> %d candidates",
                category.value,
                n_before_src_cap,
                len(df_category),
            )

        transformed_rows = df_category.apply(helper.negative_sample_transform, axis=1)
        valid_transforms = [row for row in transformed_rows if row is not None]
        n_rule_based = len(valid_transforms)
        n_rule_failed = len(df_category) - n_rule_based

        n_before_sampling = len(valid_transforms)
        if n_before_sampling > 0:
            df_category = pd.DataFrame(valid_transforms)
            for col in expected_columns:
                if col not in df_category.columns:
                    df_category[col] = None
            if samples_per_category is not None and "language_pair" in df_category.columns:
                df_category = _sample_per_group(
                    df_category,
                    by=["language_pair"],
                    n=samples_per_category,
                    random_seed=random_seed,
                )
            negative_samples = pd.concat(
                [negative_samples, df_category], ignore_index=True
            )

        n_after_sampling = len(df_category) if n_before_sampling > 0 else 0
        logger.info(
            "category=%s: %d candidates -> %d rule-based (%d failed) -> %d sampled",
            category.value,
            len(
                df[
                    (df["category"] == category.value)
                    & (df["language_pair"].isin(applicable_lps))
                ]
            ),
            n_rule_based,
            n_rule_failed,
            n_after_sampling,
        )

    logger.info("Output: %d total negative samples", len(negative_samples))
    return negative_samples


def subsample_challengeset(
    df: pd.DataFrame,
    lang_pairs: tuple[str, ...],
    samples_per_category: int = 50,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Sub-sample a challenge set to *lang_pairs* and a per-(lp, category) cap."""
    for col in ("language_pair", "category"):
        if col not in df.columns:
            raise ValueError(
                f"DataFrame missing `{col}` column; got {list(df.columns)}"
            )

    kept = df[df["language_pair"].isin(lang_pairs)]
    sampled = _sample_per_group(
        kept,
        by=["language_pair", "category"],
        n=samples_per_category,
        random_seed=random_seed,
    )
    mask = df.index.isin(sampled.index)
    logger.info(
        "Subsampled %d/%d rows (%d after lp filter) across %d lang pairs",
        int(mask.sum()),
        len(df),
        len(kept),
        kept["language_pair"].nunique(),
    )
    return df[mask]
