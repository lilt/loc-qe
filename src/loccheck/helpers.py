"""Per-category checklist helpers.

Each helper implements three operations for its category:

- ``candidate_condition``: does a data row qualify as a candidate for this
  category?
- ``negative_sample_transform``: turn a candidate row into a
  (``src``, ``mt``, ``neg``, ``ref``) minimal pair, where ``neg`` is identical
  to ``mt`` except for the phenomenon under test.
"""

import re
import random
import unicodedata
from abc import abstractmethod, ABC
from typing import Optional

import pandas as pd
from pandas import Series

from loccheck.constants import ChecklistCategory, category_applies_to_language_pair
from loccheck.number_match import HeuristicNumberMatch


class AbstractChecklistHelper(ABC):
    """Abstract class for checklist helpers.
    """

    def __init__(self, random_seed: int = 42):
        """Initialize the helper with a random state seeded for reproducible operations."""
        self.random_state = random.Random(random_seed)

    def applies_to_language_pair(self, language_pair: str) -> bool:
        return category_applies_to_language_pair(self.category, language_pair)

    @property
    def category(self) -> ChecklistCategory:
        raise NotImplementedError

    @abstractmethod
    def candidate_condition(self, row: Series) -> bool:
        """ Expected to be applied to pre-filtered, pre-processed data with at least the text columns
        `source`, `translation`, and `reference`.
        This filter should only allow rows where we're sure how to create a negative sample.
        """
        raise NotImplementedError

    @abstractmethod
    def negative_sample_transform(self, row: Series) -> Optional[dict]:
        """ Expected to be applied to pre-filtered, pre-processed data.  Required text columns are
        `source`, `translation`, and `reference`.
        The method should return a dict with the keys `source`, `positive`, `negative`,
        and `reference`.
        Returns None if the row cannot be transformed into a valid negative sample.
        """
        raise NotImplementedError

    @classmethod
    def from_category(cls, category: ChecklistCategory, random_seed: int = 42) -> "AbstractChecklistHelper":
        if category not in CATEGORY_TO_HELPER_CLASS:
            raise ValueError(f"Unsupported/non-implemented category: {category}")
        helper_class = CATEGORY_TO_HELPER_CLASS[category]
        return helper_class(random_seed=random_seed)

    @staticmethod
    def _replace_newlines(text: str) -> str:
        return text.replace("\r", " ").replace("\n", " ")

    def _format_negative_sample(self, src: str, pos: str, neg: str, ref: str, original_row: Series) -> dict:
        """Format a negative sample by merging transformed columns with all other columns from the original row.

        Args:
            original_row: The original pandas Series row containing all metadata and other columns

        Returns:
            A dict combining the transformed columns with all other columns from the original row,
            excluding the original text columns that were transformed.
        """
        result = original_row.to_dict()
        text_columns_to_exclude = ['source', 'translation', 'reference']
        for col in text_columns_to_exclude:
            result.pop(col, None)
        transformed_dict = {
            "source": self._replace_newlines(src),
            "positive": self._replace_newlines(pos),
            "negative": self._replace_newlines(neg),
            "reference": self._replace_newlines(ref),
        }
        result.update(transformed_dict)
        return result


class URLCheckHelper(AbstractChecklistHelper):
    @property
    def category(self) -> ChecklistCategory:
        return ChecklistCategory.copy_urls

    def candidate_condition(self, row: Series) -> bool:
        src_urls = self._extract_urls(row.source)
        mt_urls = self._extract_urls(row.translation)
        final_urls = self._extract_urls(row.reference)
        if not src_urls or not mt_urls or not final_urls:
            return False
        if len(src_urls) > 1 or len(mt_urls) > 1 or len(final_urls) > 1:
            return False
        urls_len = sum([len(url) for url in src_urls])
        if len(row.source) <= urls_len + 5:  # look for segments with more content, not only the url
            return False
        return True

    def negative_sample_transform(self, row: Series) -> Optional[dict]:
        src_urls = self._extract_urls(row.source)
        mt_urls = self._extract_urls(row.translation)
        final_urls = self._extract_urls(row.reference)

        src_url = list(src_urls)[0]
        mt_url = list(mt_urls)[0]
        final_url = list(final_urls)[0]

        if src_url != mt_url:
            bad_url = mt_url
            negative_sample = row.translation
            positive_sample = row.translation.replace(bad_url, src_url)
        else:
            return None

        reference = row.reference
        if src_url != final_url:
            reference = reference.replace(final_url, src_url)
        return self._format_negative_sample(
            src=row.source, pos=positive_sample, neg=negative_sample, ref=reference, original_row=row
        )

    @staticmethod
    def _extract_urls(text: str) -> set[str]:
        # Pattern 1: http:// or https:// URLs
        # Pattern 2: www. URLs
        # Pattern 3: Domain-like patterns (e.g., example.com/path)
        #   - Must have a valid TLD (2+ letters)
        #   - For bare domains (without http:// or www.), TLD must be followed by path/query chars (/?#)
        #     to distinguish from file extensions like .cpp, .py, etc. (which don't have /?# after them)
        #   - Must NOT be followed by another dot+letters (to exclude things like UiPath.System.Activities)
        # Character class for http/www URLs - match ASCII printable chars (excluding whitespace, angle brackets, and sentence-ending punctuation)
        # Period (\x2E) is included as it's needed in domains, but we'll strip trailing punctuation after matching
        url_pattern = (
            r'(?:https?://[\x21-\x3B\x3D\x3F-\x7E]+|'
            r'www\.[\x21-\x3B\x3D\x3F-\x7E]+|'
            r'(?<![.@])(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}(?![a-zA-Z0-9.-]*\.[a-zA-Z])(?:[/?#][\x21-\x3B\x3D\x3F-\x7E]*)))'
        )
        urls = re.findall(url_pattern, text)
        # Strip trailing sentence-ending punctuation from URLs
        # Common trailing punctuation: , . ) ] ! ? ; : " '
        cleaned_urls = set()
        for url in urls:
            # Remove trailing sentence-ending punctuation
            # Note: ? is in query strings, but trailing ? is usually punctuation
            cleaned = url.rstrip(',.)]!?;:"\'')
            cleaned_urls.add(cleaned)
        return cleaned_urls


class NumberChecklistHelper(AbstractChecklistHelper):
    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed=random_seed)
        self.number_matcher = HeuristicNumberMatch()

    @property
    def category(self) -> ChecklistCategory:
        return ChecklistCategory.long_number

    def candidate_condition(self, row: Series) -> bool:
        source = self.number_matcher._normalize(row.get('source'))
        groups = self.number_matcher._get_digit_runs(source)
        if not groups:
            return False
        char_lengths = [len(group_key) for group_key, _ in groups.items()]
        if any([char_len >= 5 for char_len in char_lengths]):
            return True
        return False

    def negative_sample_transform(self, row: Series) -> Optional[dict]:
        source = row.get('source')
        mt = row.get('translation')
        reference = row.get('reference')
        if not self.number_matcher.numbers_probably_match(source, reference):
            return None
        if not self.number_matcher.numbers_probably_match(source, mt):  # try finding a real data error
            negative_sample = mt
            positive_sample = self._fix_mismatched_numbers(source, mt)
            if not positive_sample:
                return None
        else:
            positive_sample = mt
            negative_sample = self._perturb_number(mt)
            if not negative_sample:
                return None

        # Final check: ensure positive and negative samples are different
        if positive_sample == negative_sample:
            return None

        return self._format_negative_sample(
            src=source, pos=positive_sample, neg=negative_sample, ref=reference, original_row=row
        )

    def _perturb_number(self, translation: str) -> Optional[str]:
        """ At this point we already know that the source and translations' number runs match. However,
        there might be differences in things like decimal separators.
        """
        def _swap_adjacent_digits(number: str) -> str:
            # Try up to 10 times to find a valid swap position (different adjacent chars)
            for _ in range(10):
                idx = self.random_state.randrange(len(number) - 1)
                if number[idx] != number[idx + 1]:
                    chars = list(number)
                    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                    return ''.join(chars)
            # If all adjacent pairs are identical, just swap the first two anyway so we return something
            chars = list(number)
            chars[0], chars[1] = chars[1], chars[0]
            return ''.join(chars)

        def _add_zero(number: str) -> str:
            return number + "0"

        def _remove_random_digit(number: str) -> str:
            idx = self.random_state.randrange(len(number))
            if number[idx] in self.number_matcher.strictly_ordered_separators:
                idx = idx - 1
            # Remove only the character at position idx, not all occurrences
            return number[:idx] + number[idx + 1:]

        translation = self.number_matcher._normalize(translation)

        translation_numbers = self.number_matcher.merge_pattern.finditer(translation)
        for match in translation_numbers:
            number = match.group(0)
            if len(number) < 5:
                continue
            perturbation = self.random_state.choice([_swap_adjacent_digits, _add_zero, _remove_random_digit])(number)
            return translation.replace(number, perturbation)
        return None

    def _fix_mismatched_numbers(self, source: str, translation: str) -> Optional[str]:
        source = self.number_matcher._normalize(source)
        translation = self.number_matcher._normalize(translation)
        source_runs = self.number_matcher._get_digit_runs(source)
        translation_runs = self.number_matcher._get_digit_runs(translation)

        non_matched_source = source_runs - translation_runs
        non_matched_translation = translation_runs - source_runs
        non_matched_source_set = set(non_matched_source.keys())
        non_matched_translation_set = set(non_matched_translation.keys())
        for run_key, run_count in non_matched_source.items():
            if run_count == 0:
                non_matched_source_set.discard(run_key)
            if run_count > 1:
                return None
            if len(run_key) < 5:
                return None

        for run_key, run_count in non_matched_translation.items():
            if run_count == 0:
                non_matched_translation_set.discard(run_key)
            if run_count > 1:
                return None

        if not len(non_matched_source_set) == 1 or not len(non_matched_translation_set) == 1:
            return None

        unmatched_source_run = list(non_matched_source_set)[0]
        unmatched_translation_run = list(non_matched_translation_set)[0]
        unmatched_source_number = None
        unmatched_translation_number = None

        source_numbers = self.number_matcher.merge_pattern.finditer(source)
        for source_number in source_numbers:
            source_run = "".join(c for c in source_number.group(0) if c not in self.number_matcher.strictly_ordered_separators)
            if source_run == unmatched_source_run:
                unmatched_source_number = source_number.group(0)

        translation_numbers = self.number_matcher.merge_pattern.finditer(translation)
        for translation_number in translation_numbers:
            translation_run = "".join(c for c in translation_number.group(0) if c not in self.number_matcher.strictly_ordered_separators)
            if translation_run == unmatched_translation_run:
                unmatched_translation_number = translation_number.group(0)

        if unmatched_source_number is None or unmatched_translation_number is None:
            return None
        return translation.replace(unmatched_translation_number, unmatched_source_number)

class LeadingTrailingSpacesChecklistHelper(AbstractChecklistHelper):
    def __init__(self, random_seed: int = 42):
        """ Python's re supposedly supports all Unicode whitespace characters with \\s. """
        super().__init__(random_seed=random_seed)
        self.leading_space_pattern = re.compile(r"^\s+")
        self.trailing_space_pattern = re.compile(r"\s+$")

    @property
    def category(self) -> ChecklistCategory:
        return ChecklistCategory.leading_trailing_space

    def candidate_condition(self, row: Series) -> bool:
        leading_space_match = self.leading_space_pattern.search(row.source)
        trailing_space_match = self.trailing_space_pattern.search(row.source)
        return bool(leading_space_match) or bool(trailing_space_match)

    def negative_sample_transform(self, row: Series) -> Optional[dict]:
        source = row.get('source')
        if not isinstance(source, str) or not source:
            return None

        leading_match = self.leading_space_pattern.search(source)
        trailing_match = self.trailing_space_pattern.search(source)
        leading_spaces = leading_match.group(0) if leading_match else ""
        trailing_spaces = trailing_match.group(0) if trailing_match else ""

        base_text = row.get('translation')
        if not base_text:
            return None
        text_core = base_text.strip()
        positive_sample = f"{leading_spaces}{text_core}{trailing_spaces}"

        # space, tab, NBSP, narrow NBSP, ZWSP, ZWNJ, ZWJ, ideographic space, LTR/RTL marks
        whitespace_variants = [
            " ", "\t", " ", " ", "​", "‌",
            "‍", "　", "‎", "‏",
        ]

        def _random_whitespace(exclude: Optional[str] = None) -> str:
            choices = [w for w in whitespace_variants if w != exclude] if exclude else whitespace_variants
            return self.random_state.choice(choices) if choices else " "

        def _remove_char(spaces: str) -> str:
            if not spaces:
                return spaces
            idx = self.random_state.randrange(len(spaces))
            return spaces[:idx] + spaces[idx + 1:]

        def _replace_char(spaces: str) -> str:
            if not spaces:
                return spaces
            idx = self.random_state.randrange(len(spaces))
            current = spaces[idx]
            replacement = _random_whitespace(current)
            return spaces[:idx] + replacement + spaces[idx + 1:]

        def _add_char(spaces: str) -> str:
            addition = _random_whitespace()
            if not spaces:
                return addition
            if self.random_state.choice([True, False]):
                return addition + spaces
            return spaces + addition

        mutations = []
        if leading_spaces:
            mutations.extend([
                lambda: (_remove_char(leading_spaces), trailing_spaces),
                lambda: (_replace_char(leading_spaces), trailing_spaces),
                lambda: (_add_char(leading_spaces), trailing_spaces),
            ])
        if trailing_spaces:
            mutations.extend([
                lambda: (leading_spaces, _remove_char(trailing_spaces)),
                lambda: (leading_spaces, _replace_char(trailing_spaces)),
                lambda: (leading_spaces, _add_char(trailing_spaces)),
            ])
        if not mutations:
            return None

        neg_leading, neg_trailing = leading_spaces, trailing_spaces
        for _ in range(3):
            neg_leading_candidate, neg_trailing_candidate = self.random_state.choice(mutations)()
            if neg_leading_candidate != leading_spaces or neg_trailing_candidate != trailing_spaces:
                neg_leading, neg_trailing = neg_leading_candidate, neg_trailing_candidate
                break

        negative_sample = f"{neg_leading}{text_core}{neg_trailing}"
        if negative_sample == positive_sample:
            return None

        reference_text = row.get('reference')
        reference = f"{leading_spaces}{reference_text.strip()}{trailing_spaces}"

        return self._format_negative_sample(
            src=source, pos=positive_sample, neg=negative_sample, ref=reference, original_row=row
        )

class FrenchNBSPChecklistHelper(AbstractChecklistHelper):
    # Substrings that should NOT appear in a canonical French reference —
    # regular space (or no separator) where NBSP is required. Used to
    # reject mistyped refs at the candidate stage so we don't try to
    # `_fix_nbsp` a hyp against a ref that's already inconsistent.
    _BAD_SEPARATOR_SUBSTRINGS = ("« ", " »", " ;", " !", " ?", " :")

    def __init__(
        self,
        random_seed: int = 42,
        regular_space_perturb_rate: float = 0.5,
        strict_filter: bool = False,
    ):
        super().__init__(random_seed=random_seed)
        # French typography: NBSP (U+00A0) or narrow NBSP (U+202F) before
        # ; ! ? : and around « ». Both NBSP variants are accepted since they
        # are used interchangeably in practice.
        _punct = [";", "!", "?", ":"]
        self.nbsp_substrings = (
            [f"{nbsp}{p}" for nbsp in (" ", " ") for p in _punct]
            + [f"«{nbsp}" for nbsp in (" ", " ")]
            + [f"{nbsp}»" for nbsp in (" ", " ")]
        )
        if not 0.0 <= regular_space_perturb_rate <= 1.0:
            raise ValueError(
                "regular_space_perturb_rate must be in [0, 1]; got "
                f"{regular_space_perturb_rate}"
            )
        self.regular_space_perturb_rate = regular_space_perturb_rate
        # When True, `negative_sample_transform` enforces canonical French
        # typography on both ref and hyp before emitting any pair (rejects
        # mistyped refs entirely; falls back to ref when hyp is mistyped).
        self.strict_filter = strict_filter
        # `«` followed by / `»` preceded by an "attaching" non-space char.
        # Catches `«N…` / `…N»` no-separator typography errors. The
        # character class excludes regular ASCII space, NBSP, and narrow
        # NBSP — i.e. only "non-space attaching" chars trigger.
        self._wrong_open_quote = re.compile("«[^\\s  ]")
        self._wrong_close_quote = re.compile("[^\\s  ]»")

    @property
    def category(self) -> ChecklistCategory:
        return ChecklistCategory.french_nbsp

    def _has_wrong_french_typography(self, text: str) -> bool:
        """True if *text* contains French-typography errors (regular space
        or missing separator) next to «, », ;, !, ?, :.

        We refuse to build pref pairs from such refs/hyps because
        :meth:`_fix_nbsp` can only patch one substring class at a time and
        would leave the others wrong on the positive side.
        """
        if any(s in text for s in self._BAD_SEPARATOR_SUBSTRINGS):
            return True
        if self._wrong_open_quote.search(text):
            return True
        if self._wrong_close_quote.search(text):
            return True
        return False

    def candidate_condition(self, row: Series) -> bool:
        reference = row.get('reference')
        if not isinstance(reference, str):
            return False
        # Accept refs that have *any* French-typography signal — either a
        # canonical NBSP slot or a mistyped one (regular space or missing
        # separator next to «»;!?:). The strict-mode transform will
        # canonicalize the mistyped variants before emitting the pair,
        # so there's no reason to drop them at the candidate stage. The
        # legacy (non-strict) `_fix_nbsp` path also handles them — what
        # it can't fix, it falls back on ``pos = ref``.
        if any(substr in reference for substr in self.nbsp_substrings):
            return True
        if self._has_wrong_french_typography(reference):
            return True
        return False

    def negative_sample_transform(self, row: Series) -> Optional[dict]:
        reference = row.get('reference')
        hyp = row.get('translation')
        if hyp == reference:
            return None

        if self.strict_filter:
            # Strict mode builds canonical pairs from the reference and
            # ignores the model hypothesis entirely. We *canonicalize*
            # the reference (fix any mistyped slots — regular space or
            # missing separator next to `«»;!?:`) and emit:
            #   pos = canonicalized reference
            #   neg = perturbation of that canonical reference
            # This eliminates two failure modes in one move:
            #   (a) hyp leaking wrong typography into pos, and
            #   (b) row loss from refusing mistyped refs — we now fix
            #       them in place instead of dropping the row.
            if not isinstance(reference, str):
                return None
            positive_sample = self._canonicalize_french_typography(reference)
            if not any(
                s in positive_sample for s in self.nbsp_substrings
            ):
                # Canonicalization produced a string with no NBSP slots
                # to perturb — nothing useful to learn from this row.
                return None
            negative_sample = self._perturb_nbsp(positive_sample)
            if not negative_sample or negative_sample == positive_sample:
                return None
            # Store the canonicalised reference too. Otherwise the CSV
            # ends up with `mt` (canonical) next to `ref` (original
            # mistyped), which is confusing to inspect and provides no
            # upside — the original ref is still recoverable upstream in
            # the candidates CSV.
            reference = positive_sample
        else:
            # Legacy behaviour of the internal pipeline. Buggy for
            # mistyped refs and for hyps with wrong NBSP positions but
            # matching count; new pipelines should pass `strict_filter=True`.
            if self._recall(hyp, reference) == 1.0:
                positive_sample = hyp
                negative_sample = self._perturb_nbsp(hyp)
                if not negative_sample:
                    return None
            else:
                positive_sample = self._fix_nbsp(hyp, reference)
                if positive_sample is None:
                    positive_sample = reference
                    negative_sample = self._perturb_nbsp(reference)
                    if not negative_sample:
                        return None
                else:
                    negative_sample = hyp

        return self._format_negative_sample(
            src=row.get('source'), pos=positive_sample, neg=negative_sample, ref=reference, original_row=row
        )

    def _recall(self, hyp, ref):
        ref_count = 0
        hyp_count = 0
        for substring in self.nbsp_substrings:
            if substring in ref:
                ref_count += ref.count(substring)
            if substring in hyp:
                hyp_count += hyp.count(substring)
        if ref_count == 0:
            return 1.0  # no NBSPs in reference → not applicable, treat as pass
        return float(hyp_count / ref_count)

    def _canonicalize_french_typography(self, text: str) -> str:
        """Replace common French-typography typos in *text* with NBSP forms.

        Used by the strict-mode path to fix up references (and any hyp
        that's brave enough to ask) before emitting a pair. Conservative:
        only touches the patterns we know are unambiguously wrong in
        French (regular space adjacent to ``«»;!?:`` or missing
        separator next to ``«»``). Idempotent — applying twice yields
        the same result.

        Always normalises to ``U+00A0`` (NBSP). The reference may contain
        narrow NBSP (``U+202F``) elsewhere; those are left untouched.
        """
        nbsp = " "
        for p in (";", "!", "?", ":"):
            text = text.replace(f" {p}", f"{nbsp}{p}")
        text = text.replace("« ", f"«{nbsp}")
        text = text.replace(" »", f"{nbsp}»")
        # Missing-separator cases: «X / X» where X is a non-whitespace,
        # non-NBSP char. Insert an NBSP between them.
        text = self._wrong_open_quote.sub(
            lambda m: f"«{nbsp}{m.group(0)[1:]}", text
        )
        text = self._wrong_close_quote.sub(
            lambda m: f"{m.group(0)[0]}{nbsp}»", text
        )
        return text

    @staticmethod
    def _strip_nbsp(s: str) -> str:
        """Remove NBSP and narrow NBSP from a string."""
        return s.replace(" ", "").replace(" ", "")

    @staticmethod
    def _nbsp_to_space(s: str) -> str:
        """Replace NBSP and narrow NBSP with regular space."""
        return s.replace(" ", " ").replace(" ", " ")

    def _perturb_nbsp(self, hyp) -> Optional[str]:
        if not any(substring in hyp for substring in self.nbsp_substrings):
            return None
        for substring in self.nbsp_substrings:
            if substring in hyp:
                stripped = self._strip_nbsp(substring)
                normal_space = self._nbsp_to_space(substring)
                # Weighted: "regular space" perturbations produce noisier
                # pairs because the result often still reads as plausible
                # French, so they can be downweighted via
                # `regular_space_perturb_rate`.
                replacement = (
                    normal_space
                    if self.random_state.random() < self.regular_space_perturb_rate
                    else stripped
                )
                hyp = hyp.replace(substring, replacement)
                if self.random_state.choice([True, False]):
                    return hyp
        return hyp

    def _fix_nbsp(self, hyp, ref) -> Optional[str]:
        """Patch missing NBSP substrings in *hyp* using *ref* as the target.

        Only replaces a wrong-form occurrence in *hyp* when its count
        matches the expected count in *ref* exactly. This rules out the
        bug where a global ``str.replace`` over-applied — e.g. converting
        every "space + colon" into "NBSP + colon" when only some of them
        should have been NBSP. Returns ``None`` whenever the alignment
        is ambiguous; the caller is expected to fall back to ``pos=ref``.
        """
        missing = [
            s for s in self.nbsp_substrings if s in ref and s not in hyp
        ]
        if not missing:
            return None
        for substring in missing:
            stripped = self._strip_nbsp(substring)
            normal_space = self._nbsp_to_space(substring)
            ref_count = ref.count(substring)
            # Prefer `normal_space` matching: it's the more specific
            # pattern. Only fire when the hyp count matches the ref count
            # exactly, otherwise we don't know which positions are NBSP
            # slots and which were legitimately separated by a real
            # space (e.g. the colon in `« objet : » plus » Y :`).
            if normal_space in hyp and hyp.count(normal_space) == ref_count:
                hyp = hyp.replace(normal_space, substring)
            elif stripped in hyp and hyp.count(stripped) == ref_count:
                hyp = hyp.replace(stripped, substring)
            else:
                return None
        return hyp


class UnicodeSpacesChecklistHelper(AbstractChecklistHelper):
    """Challenge set for Unicode space/format characters that nmt_nfkc normalisation collapses.

    Tests whether the metric can detect when a special Unicode character in the
    reference is replaced with a regular space or removed entirely in the hypothesis.
    """

    # Characters that carry meaning in translated text but are collapsed by nmt_nfkc.
    UNICODE_CHARS: list[tuple[str, str]] = [
        (" ", "NBSP"),
        (" ", "narrow NBSP"),
        ("​", "ZWS"),
        ("‌", "ZWNJ"),
        ("‍", "ZWJ"),
        ("　", "ideographic space"),
        ("‎", "LTR mark"),
        ("‏", "RTL mark"),
    ]

    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed=random_seed)
        self._char_pattern = re.compile(
            "[" + "".join(re.escape(ch) for ch, _ in self.UNICODE_CHARS) + "]"
        )

    @property
    def category(self) -> ChecklistCategory:
        return ChecklistCategory.unicode_spaces

    def candidate_condition(self, row: Series) -> bool:
        ref = row.get("reference", "")
        return bool(self._char_pattern.search(ref))

    def negative_sample_transform(self, row: Series) -> Optional[dict]:
        reference = row.get("reference", "")
        hyp = row.get("translation", "")
        if not reference or not hyp:
            return None

        ref_chars = set(self._char_pattern.findall(reference))
        if not ref_chars:
            return None

        hyp_chars = set(self._char_pattern.findall(hyp))
        hyp_has_all = ref_chars.issubset(hyp_chars)

        if hyp_has_all:
            positive_sample = hyp
            negative_sample = self._perturb(hyp)
        else:
            # Don't try to position-anchored restore — too ambiguous for
            # space/format chars (the first N regular spaces are rarely
            # the right slots). Fall back to using the reference as the
            # positive (canonical by definition) and perturbing it to get
            # the negative.
            positive_sample = reference
            negative_sample = self._perturb(reference)

        if positive_sample is None or negative_sample is None:
            return None
        if positive_sample == negative_sample:
            return None

        return self._format_negative_sample(
            src=row.get("source", ""),
            pos=positive_sample,
            neg=negative_sample,
            ref=reference,
            original_row=row,
        )

    def _perturb(self, text: str) -> Optional[str]:
        """Replace each special Unicode char with regular space or remove it."""
        if not self._char_pattern.search(text):
            return None

        def _replace(m: re.Match) -> str:
            return self.random_state.choice([" ", ""])

        return self._char_pattern.sub(_replace, text)

class AllCapsChecklistHelper(AbstractChecklistHelper):
    @property
    def category(self) -> ChecklistCategory:
        return ChecklistCategory.all_caps

    def candidate_condition(self, row: Series) -> bool:
        """ If more than 60% of the segment is uppercase letters, reject, because I actually want to take normal
        segments and uppercase them for the transform. """

        source = row.get('source')
        uppercase_count = 0
        for char in source:
            if char.isupper():
                uppercase_count += 1
        return uppercase_count / len(source) < 0.6

    def negative_sample_transform(self, row: Series) -> Optional[dict]:
        source = row.get('source').upper()
        reference = row.get('reference').upper()
        positive_sample = row.get('translation').upper()
        negative_sample = row.get('translation')
        return self._format_negative_sample(
            src=str(source), pos=str(positive_sample), neg=negative_sample, ref=str(reference), original_row=row
        )

class HallucinatedPeriodChecklistHelper(AbstractChecklistHelper):
    def __init__(self, random_seed: int = 42):
        """ Python's re supposedly supports all Unicode whitespace characters with \\s. """
        super().__init__(random_seed=random_seed)
        self.trailing_space_pattern = re.compile(r"\s+$")
        # Maps target language to its period symbol; languages not listed here
        # default to "." (ASCII full stop) which covers Latin/Cyrillic scripts.
        self.period_symbols = {
            "ja": "。",  # 。
            "zh": "。",
            "hi": "।",  # ।
        }

    @property
    def category(self) -> ChecklistCategory:
        return ChecklistCategory.hallucinated_period

    @staticmethod
    def _is_unicode_punct(char):
        return unicodedata.category(char).startswith("P")

    def candidate_condition(self, row: Series) -> bool:
        # ...some minimum number of words...and let's go with segments that actually *don't* end in a period
        # realistic data distribution etc etc
        source = row.get('source')
        # basic whitespace splitting is enough here but only because we always have english on source side
        length = len(source.split())
        if length < 10:
            return False
        last_nonspace_char = source.strip()[-1]
        return not self._is_unicode_punct(last_nonspace_char)

    def negative_sample_transform(self, row: Series) -> Optional[dict]:
        tgt_lang = row.get('language_pair')[-2:]
        source = row.get('source')
        mt = row.get('translation')
        reference = row.get('reference')

        if not source or not mt or not reference:
            return None

        reference_stripped = reference.rstrip()
        if not reference_stripped:
            return None
        last_ref_char = reference_stripped[-1]
        # at this point we already only have sources that don't end in punct, so force the reference to not end in punct.
        if self._is_unicode_punct(last_ref_char):
            reference = self._remove_final_punct_keep_spaces(reference)
            if reference is None:
                return None

        mt_stripped = mt.strip()
        if not mt_stripped:
            return None
        last_mt_char = mt_stripped[-1]
        if self._is_unicode_punct(last_mt_char):
            negative_sample = mt
            # force positive sample to not end in punct
            positive_sample = self._remove_final_punct_keep_spaces(mt)
            if positive_sample is None:
                return None
        else:
            positive_sample = mt
            trailing_space_match = self.trailing_space_pattern.search(mt)
            trailing_space = trailing_space_match.group(0) if trailing_space_match else ""
            mt_without_trailing_space = mt.rstrip()
            period_symbol = self.period_symbols.get(tgt_lang, ".")
            negative_sample = f"{mt_without_trailing_space}{period_symbol}{trailing_space}"

        return self._format_negative_sample(
            src=source, pos=positive_sample, neg=negative_sample, ref=reference, original_row=row
        )

    def _remove_final_punct_keep_spaces(self, translation):
        """Strip a single run of sentence-terminal punctuation from *translation*.

        Returns ``None`` when the typography is ambiguous (last
        non-whitespace char is a closing quote/bracket): languages
        disagree on whether the terminal period sits inside the quote
        (American ``"word."``) or outside it (British ``"word".``), so
        stripping risks emitting a malformed positive that differs from
        the negative on more than just the period.

        Also returns ``None`` if the last char is punctuation but not
        sentence-terminal (e.g. an em-dash, ellipsis, or comma) — we
        only know how to invert the "hallucinated terminal period"
        signal, not arbitrary punctuation drops.
        """
        if not translation.strip():
            return None

        trailing_space_match = self.trailing_space_pattern.search(translation)
        trailing_space = trailing_space_match.group(0) if trailing_space_match else ""
        core = translation.rstrip()
        last_char = core[-1]

        if last_char in self._TRAILING_QUOTES:
            return None

        terminal_puncts = set(self.period_symbols.values()) | {".", "!", "?"}
        if last_char not in terminal_puncts:
            return None

        while core and core[-1] in terminal_puncts:
            core = core[:-1]
        if not core:
            return None
        return f"{core}{trailing_space}"

    # Closing quotes/brackets that may follow sentence-final punctuation.
    _TRAILING_QUOTES = set('"\'")»」』】）>›‹')

class SpanishInitialPunctuationHelper(AbstractChecklistHelper):
    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed)
        self.initial_question_mark = "¿"
        self.initial_exclamation_mark = "¡"

    @property
    def category(self) -> ChecklistCategory:
        return ChecklistCategory.spanish_punct

    def candidate_condition(self, row: Series) -> bool:
        # Hard gate: spanish-punct is Spanish-target-only. Upstream
        # callers (`sample_candidates`) already restrict to applicable
        # LPs, but enforce it here too so any direct caller can't
        # accidentally sample non-Spanish data into this category.
        if not self.applies_to_language_pair(row.get('language_pair', '')):
            return False
        ref = row.get('reference')
        src = row.get('source')
        return ((self.initial_question_mark in ref or self.initial_exclamation_mark in ref)
                and ("?" in src or "!" in src) and ("?" in ref or "!" in ref))

    def negative_sample_transform(self, row: Series) -> Optional[dict]:
        # here we only have segments that we already know are questions or exclamations
        # and the reference has both the initial and final punctuation marks.
        source = row.get('source')
        reference = row.get('reference')
        hyp = row.get('translation')
        if self.initial_question_mark in reference:
            if self.initial_question_mark in hyp:
                positive_sample = hyp
                negative_sample = hyp.replace(self.initial_question_mark, "")
            else:
                return None
        elif self.initial_exclamation_mark in reference:
            if self.initial_exclamation_mark in hyp:
                positive_sample = hyp
                negative_sample = hyp.replace(self.initial_exclamation_mark, "")
            else:
                return None
        else:
            return None
        return self._format_negative_sample(
            src=source, pos=positive_sample, neg=negative_sample, ref=reference, original_row=row
        )

# Registry mapping categories to their helper class implementations
CATEGORY_TO_HELPER_CLASS = {
    ChecklistCategory.copy_urls: URLCheckHelper,
    ChecklistCategory.long_number: NumberChecklistHelper,
    ChecklistCategory.leading_trailing_space: LeadingTrailingSpacesChecklistHelper,
    ChecklistCategory.french_nbsp: FrenchNBSPChecklistHelper,
    ChecklistCategory.all_caps: AllCapsChecklistHelper,
    ChecklistCategory.hallucinated_period: HallucinatedPeriodChecklistHelper,
    ChecklistCategory.spanish_punct: SpanishInitialPunctuationHelper,
    ChecklistCategory.unicode_spaces: UnicodeSpacesChecklistHelper,
}
