"""Public data sources for challenge-set generation: WMT human-eval data and BOUQuET.

Both loaders return a flat DataFrame with one row per (segment, system) pair
and the columns expected by :mod:`loccheck.generate`::

    language_pair, source, translation, reference, translation_reference_chrfpp,
    domain, ...
"""

import json
import logging
import re
import tempfile
import urllib.request

import pandas as pd
from sacrebleu import CHRF

logger = logging.getLogger(__name__)

WMT25_URL = "https://github.com/wmt-conference/wmt25-general-mt/raw/refs/heads/main/data/wmt25-genmt-humeval.jsonl"

# language pair in doc_id is e.g. "en-de_DE" or "cs-de_DE"; we want "en-de"
_DOC_ID_LP_PATTERN = re.compile(r"^([a-z]{2,3})-([a-z]{2,3})")

# Default pattern for identifying reference systems in WMT tgt_text keys
_DEFAULT_REF_PATTERN = re.compile(r"^ref[A-Z]?$", re.IGNORECASE)


def sentence_chrfpp(hyps: list[str], refs: list[str]) -> list[float]:
    chrf = CHRF(word_order=2)
    return [
        chrf.sentence_score(hypothesis=hyp, references=[ref]).score
        for hyp, ref in zip(hyps, refs, strict=True)
    ]


def _parse_doc_id(doc_id: str) -> dict:
    """Extract language pair, domain, document name, and segment id from a WMT doc_id.

    Format: ``{lp_with_variant}_#_{domain}_#_{documentname}_#_{segmentid}``
    """
    parts = doc_id.split("_#_")
    lp_raw = parts[0] if len(parts) >= 1 else ""
    domain = parts[1] if len(parts) >= 2 else ""
    document_name = parts[2] if len(parts) >= 3 else ""
    segment_id = parts[3] if len(parts) >= 4 else ""

    lp_match = _DOC_ID_LP_PATTERN.match(lp_raw)
    language_pair = f"{lp_match.group(1)}-{lp_match.group(2)}" if lp_match else lp_raw

    return {
        "language_pair": language_pair,
        "domain": domain,
        "document_id": document_name,
        "segment_id": segment_id,
    }


def load_wmt_jsonl(
    url: str = WMT25_URL,
    ref_pattern: str = _DEFAULT_REF_PATTERN.pattern,
    ref_systems: list[str] | None = None,
    language_pairs: list[str] | None = None,
) -> pd.DataFrame:
    """Download a WMT shared-task human-eval JSONL and expand it into a flat DataFrame.

    Each row in the output corresponds to one system's translation for one
    source segment, paired with its ``reference``.  Systems whose names
    match *ref_pattern* are treated as references and excluded from the system
    rows.  When multiple references match, the first one (alphabetically) is
    used as ``final``.

    If *ref_systems* is provided it takes precedence over *ref_pattern*.
    When *language_pairs* is given, only these exact LPs (e.g. ``["en-de"]``)
    are kept.
    """

    def _is_ref_system(name: str) -> bool:
        if ref_systems is not None:
            return name in ref_systems
        return bool(re.match(ref_pattern, name, re.IGNORECASE))

    logger.info("Downloading %s", url)
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        lines = open(tmp.name).readlines()

    rows = []
    ref_system_names: set[str] = set()
    segments_without_ref: list[str] = []  # doc_ids
    for line in lines:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        src_text = obj.get("src_text", "")
        tgt_text = obj.get("tgt_text", {})
        doc_id = obj.get("doc_id", "")
        meta = _parse_doc_id(doc_id)

        # separate references from system outputs
        ref_translations = {}
        sys_translations = {}
        for sys_name, translation in tgt_text.items():
            if _is_ref_system(sys_name):
                ref_translations[sys_name] = translation
                ref_system_names.add(sys_name)
            else:
                sys_translations[sys_name] = translation

        if not isinstance(src_text, str) or not src_text:
            continue

        if not ref_translations:
            segments_without_ref.append(doc_id)
            continue

        # pick first reference alphabetically
        ref_name = sorted(ref_translations.keys())[0]
        reference = ref_translations[ref_name]
        if not isinstance(reference, str) or not reference:
            segments_without_ref.append(doc_id)
            continue

        for system_name, translation in sys_translations.items():
            if not isinstance(translation, str) or not translation:
                continue
            rows.append({
                **meta,
                "system": system_name,
                "source": src_text,
                "translation": translation,
                "reference": reference,
            })

    if segments_without_ref:
        logger.warning(
            "%d segments skipped for missing reference", len(segments_without_ref)
        )

    df = pd.DataFrame(rows)
    if language_pairs is not None and not df.empty:
        df = df[df["language_pair"].isin(language_pairs)].reset_index(drop=True)
    if not df.empty:
        df["translation_reference_chrfpp"] = sentence_chrfpp(
            df["translation"].tolist(), df["reference"].tolist(),
        )
    logger.info(
        "WMT %s: %d rows, %d lang pairs, refs: %s",
        url.rsplit("/", 1)[-1],
        len(df),
        df["language_pair"].nunique() if not df.empty else 0,
        sorted(ref_system_names),
    )
    return df


def _normalize_flores_lp(src_lang: str, tgt_lang: str) -> str:
    """Convert FLORES-style lang codes (``hin_Deva``) to short codes (``hi``)."""
    import langcodes

    src = langcodes.Language.get(src_lang.split("_")[0]).language
    tgt = langcodes.Language.get(tgt_lang.split("_")[0]).language
    return f"{src}-{tgt}"


def load_bouquet(
    config: str = "met_bouquet_xstsrp",
    split: str = "test",
    language_pairs: list[str] | None = None,
) -> pd.DataFrame:
    """Download a Met-BOUQuET config/split from HuggingFace and convert to a flat DataFrame.

    BOUQuET is a gated dataset — requires a valid ``HF_TOKEN`` environment
    variable or ``huggingface-cli login``.

    Language pair codes are normalised from FLORES-style (``hin_Deva``) to
    short ISO codes (``hi``).  When *language_pairs* is given, only these
    exact LPs (e.g. ``["en-de"]``) are kept (applied after normalisation).
    """
    from datasets import load_dataset

    if not config.startswith("met_bouquet"):
        raise ValueError(f"Only Met-BOUQuET configs are supported, got {config!r}")

    ds = load_dataset("facebook/bouquet", config, split=split)
    df: pd.DataFrame = ds.to_pandas()

    out = pd.DataFrame()
    out["language_pair"] = [
        _normalize_flores_lp(s, t)
        for s, t in zip(df["src_lang"], df["tgt_lang"])
    ]
    out["source"] = df["src_text"]
    out["translation"] = df["mt_text"]
    out["reference"] = df["ref_text"]
    out["score"] = df["consensus_score"]
    out["domain"] = df["domain"]
    out["register"] = df.get("register_label", "")
    out["system"] = df.get("system", "")
    out["n_annotators"] = df.get("n_annotators", 0)
    out["translation_reference_chrfpp"] = sentence_chrfpp(
        out["translation"].tolist(), out["reference"].tolist()
    )

    if language_pairs is not None:
        out = out[out["language_pair"].isin(language_pairs)]

    logger.info(
        "BOUQUET %s/%s: %d rows, %d LPs",
        config,
        split,
        len(out),
        out["language_pair"].nunique(),
    )
    return out
