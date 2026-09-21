# LocQE: Localisation Quality Estimation — Data & Code

Companion repository for the paper **"LocQE: Principled Domain Adaptation for
Localisation Quality Estimation by Leveraging Post-Edits"**. [arXiv](https://arxiv.org/abs/2609.18720)

Learned quality estimation (QE) models such as COMETKiwi work well for general
machine translation evaluation but struggle in real-world localisation: they
are insensitive to phenomena that professional translators care about (exact
whitespace, non-breaking spaces, number fidelity, typography conventions) and
their segment-level ranking degrades under the domain shift. The paper
proposes a data-efficient adaptation recipe — deriving both continuous scores
and preference pairs from post-edits, combining them in multi-task
fine-tuning, and adjusting the tokeniser so the model can *see* the relevant
special characters. Two evaluation resources are introduced and released
here:

- **LocHD** (**Loc**alisation **H**uman **D**ata): 3,000 Error Span Annotation
  (ESA) human judgements of MT output for real localisation content across ten
  language pairs.
- **LocCheck**: a challenge set of minimal pairs targeting
  localisation-specific long-tail issues, together with the code that
  generates its public split from open data (WMT25 human-eval outputs and
  Met-BOUQuET).

## Repository contents

| path | description |
|---|---|
| `data/lochd/` | LocHD dataset — one JSONL file per language pair |
| `src/loccheck/` | `loccheck` Python package: LocCheck challenge-set generation |
| `pyproject.toml` | package metadata; installs the `loccheck-generate` CLI |

## LocHD

3,000 ESA annotations collected from professional translators (paid standard
rates): 300 English source segments per target language (ar, de, es, fi, fr,
ja, pl, pt, sv, tr), each translated by one of three MT systems (100 per
system per language). The source data originates from a real client project
and consists primarily of user-interface text and help documentation. LocHD
contains hypotheses but no references; each segment has exactly one scored
translation, so the intended use is computing z-scores per annotator and
evaluating QE models on global correlation (Kendall's τ) with those z-scores.

One JSONL file per language pair; fields per record:

| field | description |
|---|---|
| `id` | segment id, unique within each language file |
| `src_lang`, `trg_lang` | language pair |
| `system` | MT system pseudonym (`system_1`–`system_3`) |
| `src_prev`, `src`, `src_next` | source segment with document context |
| `trg_prev`, `trg`, `trg_next` | MT output with document context |
| `score` | annotator quality score, 0–100 |
| `annotator` | annotator pseudonym (`annotator_01`–`annotator_33`), stable across all files |
| `annotator_updated` | pseudonym of the last score update (differs on 23 QA-rescored rows) |
| `rows` | ESA error spans: list of `{start, end, text, severity}` over `trg` |
| `qa_assessment_comment` | free-text QA reviewer comment |
| `qa_overall_assessment` | QA verdict: `Correctly Assessed` / `Under Penalized` / `Over Penalized` |

Annotator identities and MT vendor names are replaced with stable pseudonyms;
the mapping is not part of this repository. Pseudonyms are consistent across
all language files, so per-annotator normalisation works on the merged data.
Sources, translations, scores, error spans, and QA comments are unmodified.


## LocCheck generation

The `loccheck` package generates the public LocCheck split: each row is a
minimal pair — a source, a positive sample (correct translation), and a
negative sample that is identical to the positive except for one localisation
phenomenon. Categories: copy-urls, long-number, leading-trailing-space,
hallucinated-period, all-caps, spanish-punct, french-nbsp, unicode-spaces.

```bash
uv sync                                     # or: pip install -e .
uv run loccheck-generate out/               # full public split (WMT25 + BOUQuET)
uv run loccheck-generate out/ --skip-bouquet    # WMT25 only, no HF login needed
uv run loccheck-generate out/ --tgt-languages fr
```

The final set is written to `out/loccheck_public.csv`; all intermediate
artifacts (raw data, candidates, per-source challenge sets) are kept alongside
for inspection. Columns:

| column | description |
|---|---|
| `source` | source segment (English) |
| `positive` | correct translation |
| `negative` | identical to `positive` except for the phenomenon under study |
| `reference` | reference translation |
| `language_pair` | e.g. `en-fr` |
| `domain` | domain label from the underlying data source |
| `category` | LocCheck category of the injected error |

A metric passes a pair when it scores `positive` above `negative`.

Notes:

- **BOUQuET is a gated dataset** — accept its terms on HuggingFace and
  authenticate (`hf auth login` or `HF_TOKEN`) before running without
  `--skip-bouquet`.

## Citation

If you use LocHD, LocCheck, or the generation code, please cite:

```bibtex
@inproceedings{locqe2026,
  title     = {{LocQE}: Principled Domain Adaptation for Localisation Quality
               Estimation by Leveraging Post-Edits},
  author    = {Hämmerl, Kathy and Bretschner, Gabriel and Wübker, Jörn},
  booktitle = {Proceedings of the Eleventh Conference on Machine Translation (WMT)},
  year      = {2026},
  note      = {To appear. Citation will be updated upon publication.}
}
```

## License

This repository — the LocHD data and the `loccheck` code — is released under
the [MIT License](LICENSE).

## Contact

For questions about the data or code, please open an issue in this
repository.
