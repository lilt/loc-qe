"""End-to-end generation of the public LocCheck challenge set.

Reproduces the paper's public split: the checklist heuristics applied to the
WMT25 human-eval system outputs and the Met-BOUQuET test set, merged, then
sub-sampled to the LocCheck language pairs with at most 50 pairs per
(language_pair, category).

Usage:
    loccheck-generate out/
    loccheck-generate out/ --skip-bouquet          # WMT25 only (no HF auth needed)
    loccheck-generate out/ --tgt-languages de,fr
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from loccheck.constants import ChecklistCategory
from loccheck.data_sources import WMT25_URL, load_bouquet, load_wmt_jsonl
from loccheck.generate import (
    COLUMN_TYPES_MAP,
    create_negative_samples,
    sample_candidates,
    subsample_challengeset,
)

logger = logging.getLogger(__name__)

# Columns shared between the WMT and BOUQuET challenge sets, used for the merge
# and kept in the released public split.
SHARED_COLUMNS = [
    "source",
    "positive",
    "negative",
    "reference",
    "language_pair",
    "domain",
    "category",
]

# Target languages of the paper's public LocCheck split (the source is
# always English).
PAPER_TGT_LANGUAGES = "ar,de,es,fi,fr,hi,ja,pl,ru,th,zh"


def build_challengeset(raw_csv: Path, out_dir: Path) -> Path:
    """Sample candidates and inject negatives for a single data source."""
    categories = list(ChecklistCategory)

    df = pd.read_csv(raw_csv, dtype=COLUMN_TYPES_MAP)
    candidates = sample_candidates(
        df,
        categories=categories,
        language_pair=None,
        samples_per_category=None,
    )
    candidates_csv = out_dir / "candidates.csv"
    candidates.to_csv(candidates_csv, index=False)
    logger.info("Wrote %d candidates to %s", len(candidates), candidates_csv)

    candidates = pd.read_csv(candidates_csv, dtype=COLUMN_TYPES_MAP)
    challengeset = create_negative_samples(
        candidates,
        categories=categories,
        language_pair=None,
        samples_per_category=100,
        max_samples_per_source=2,
        french_nbsp_regular_space_perturb_rate=0.25,
        french_nbsp_strict_filter=True,
    )
    challengeset_csv = out_dir / "challengeset.csv"
    challengeset.to_csv(challengeset_csv, index=False)
    logger.info("Wrote %d pairs to %s", len(challengeset), challengeset_csv)
    return challengeset_csv


def merge_challengesets(csvs: list[Path], out_csv: Path) -> None:
    """Concatenate challenge sets, keeping only the shared columns."""
    frames = []
    for path in csvs:
        df = pd.read_csv(
            path,
            keep_default_na=False,
            na_values=[""],
            low_memory=False,
            dtype=COLUMN_TYPES_MAP,
        )
        if df.empty:
            logger.warning("%s has no pairs, skipping", path)
            continue
        frames.append(df[SHARED_COLUMNS])
    if frames:
        merged = pd.concat(frames, ignore_index=True)
    else:
        logger.warning("No challenge set contributed any pairs")
        merged = pd.DataFrame(columns=SHARED_COLUMNS)
    merged.to_csv(out_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--wmt-url",
        default=WMT25_URL,
        help="WMT human-eval JSONL to use (default: WMT25)",
    )
    parser.add_argument(
        "--skip-bouquet",
        action="store_true",
        help="Skip the BOUQuET source (it is gated on HuggingFace and needs authentication)",
    )
    parser.add_argument(
        "--tgt-languages",
        default=PAPER_TGT_LANGUAGES,
        help="Comma-separated target language codes; the source is always "
        "English. All input data is filtered to these en->X pairs "
        "(default: the paper's target languages)",
    )
    parser.add_argument(
        "--samples-per-lp-category",
        type=int,
        default=50,
        help="Per-(language_pair, category) cap for the final subsampling (default: 50)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    lang_pairs = [f"en-{tgt}" for tgt in args.tgt_languages.split(",")]

    # WMT
    wmt_dir = args.out_dir / "wmt25"
    wmt_dir.mkdir(parents=True, exist_ok=True)
    wmt_raw = wmt_dir / "raw.csv"
    load_wmt_jsonl(args.wmt_url, language_pairs=lang_pairs).to_csv(wmt_raw, index=False)
    challengeset_csvs = [build_challengeset(wmt_raw, wmt_dir)]

    # BOUQuET
    if not args.skip_bouquet:
        bouquet_dir = args.out_dir / "bouquet"
        bouquet_dir.mkdir(parents=True, exist_ok=True)
        bouquet_raw = bouquet_dir / "raw.csv"
        load_bouquet(
            config="met_bouquet_xstsrp", split="test", language_pairs=lang_pairs
        ).to_csv(bouquet_raw, index=False)
        challengeset_csvs.append(build_challengeset(bouquet_raw, bouquet_dir))

    # Merge + subsample to the final public set
    merged_csv = args.out_dir / "merged.csv"
    merge_challengesets(challengeset_csvs, merged_csv)

    merged = pd.read_csv(merged_csv, dtype=COLUMN_TYPES_MAP)
    public = subsample_challengeset(
        merged,
        lang_pairs=tuple(lang_pairs),
        samples_per_category=args.samples_per_lp_category,
    )
    public_csv = args.out_dir / "loccheck_public.csv"
    public.to_csv(public_csv, index=False)

    print(f"wrote {len(public)} pairs to {public_csv}")
    print(public.groupby(["category"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
