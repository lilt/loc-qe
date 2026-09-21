"""Heuristic check whether the numbers in a source and a translation match."""

import unicodedata
from collections import Counter

import regex as re


def _check_single(alphabet: Counter, compound_word: str) -> Counter | None:
    if len(compound_word) == 0:
        return alphabet

    for block, count in alphabet.items():
        if count > 0 and compound_word.startswith(block):
            new_alphabet = alphabet.copy()
            new_alphabet[block] -= 1
            if new_alphabet[block] == 0:
                del new_alphabet[block]

            new_compound_word = compound_word[len(block) :]

            if len(new_compound_word) == 0:
                return new_alphabet
            else:
                if any(new_alphabet.values()):
                    result = _check_single(new_alphabet, new_compound_word)
                    if result is not None:
                        return result

    return None


def _check_all(source_multiset: Counter, target_multiset: Counter) -> bool:
    source_chars = sum(len(key) * count for key, count in source_multiset.items())
    target_chars = sum(len(key) * count for key, count in target_multiset.items())

    if source_chars != target_chars:
        return False
    elif source_chars == 0:
        return True

    for target, count in target_multiset.items():
        for _ in range(count):
            opt_source_multiset = _check_single(source_multiset, target)

            if opt_source_multiset is not None:
                source_multiset = opt_source_multiset
            else:
                return False

    return all(count == 0 for count in source_multiset.values())


def can_merge(multiset_a: Counter, multiset_b: Counter) -> bool:
    non_matched_a = multiset_a - multiset_b
    non_matched_b = multiset_b - multiset_a

    return _check_all(non_matched_a, non_matched_b) or _check_all(non_matched_b, non_matched_a)


class HeuristicNumberMatch:
    """Check whether the numbers used in two strings (source and translation) are a probable match.

    The algorithm used is Digit Run Merge Equivalence. The main insight is that the order of digits
    in any consecutive block of digits, called a digit run, is almost always reproduced verbatim in
    a translation. There are very few other true invariants: if there are multiple digit runs, they
    often appear in the translation in a different order. Digit frequency is another true invariant
    but is not useful, as it doesn't catch one of the most common mistakes for humans: swapping two
    adjacent digits.

    Examples of source and target sentences and their digit runs:
      - English: "In 2014 we had sold 5341 units by June 1st":      {"2014", "5341", "1"}
        German:  "2014 verkauften wir bis den 1. Juni 5.341 Stück": {"2014", "1", "5", "341"}
      - English: "08/16/2032: We read Psalms 17:1":                 {"08", "16", "2032", "17", "1"}
        German:  "16.08.2032: Heute war Ps. 17,1":                  {"16", "08", "2032", "17", "1"}

    The basis of the algorithm is checking whether the two sets of digit runs are merge equivalent,
    i.e., can you match each digit run in a set with digit runs or concatenated digit runs from the
    other set? The first example above is satisfied by this – "2014" and "1" match exactly, and the
    digit runs "5" and "341" can be concatenated to match the "5341".

    Note that merge equivalence isn't symmetric. You could make it symmetric by allowing splitting,
    but then the algorithm becomes equivalent to checking digit frequency.

    However there are some cases where the relative order of a series of digit runs is significant:
      - Numbers with thousands and/or decimal separators (3.14159, 10.250,11725)
      - IPv4 addresses (127.0.0.1)
      - Times with colons (16:24, 3:20 p.m.)

    These are almost entirely cases where the digit runs are separated by a single character from a
    closed set: thousands and decimal separators, periods, and colons. To force correct ordering of
    these cases, we automatically concatenate any series of digit runs separated just by them. This
    means the examples from above now look like this:
      - English: "In 2014 we had sold 5341 units by June 1st":      {"2014", "5341", "1"}
        German:  "2014 verkauften wir bis den 1. Juni 5.341 Stück": {"2014", "1", "5341"}
      - English: "08/16/2032: We read Psalms 17:1":                 {"08", "16", "2032", "17", "1"}
        German:  "16.08.2032: Heute war Ps. 17,1":                  {"16082032", "17", "1"}

    This works great for the first example, but not the second – which leads to our last exception:
    We match dates with the forms XX.XX.XXXX, XX.XX.XX or XXXX.XX.XX, and keep digit runs separate.
    """

    def __init__(self):
        # From a hopefully comprehensive list of decimal and thousands separators found here:
        # https://en.wikipedia.org/wiki/Decimal_separator#Unicode_characters
        self.strictly_ordered_separators = " ,.'\u00b7\u2009\u00a0\u202f\u02d9\u066b\u066c\u2396"

        # Add in colons for times
        self.strictly_ordered_separators += ":"

        # Any character in the Unicode category "Numeric"
        digit = r"\p{N}"

        # Pattern for a sequence of digits interrupted only by known separators
        self.merge_pattern = re.compile(f"({digit}+[{self.strictly_ordered_separators}])*{digit}+")

    def numbers_probably_match(self, source: str, target: str) -> bool:
        # Normalize the source and target texts
        source_text = self._normalize(source)
        target_text = self._normalize(target)

        # Get the digit runs – sequences of digits only interrupted by known decimal or thousands place separators
        source_digit_runs = self._get_digit_runs(source_text)
        target_digit_runs = self._get_digit_runs(target_text)

        # Try to solve the merge-equivalence problem, by either matching or merging source digit runs until
        # we get the target digit runs, or vice versa.
        runs_are_merge_equiv = can_merge(source_digit_runs, target_digit_runs) or can_merge(
            target_digit_runs, source_digit_runs
        )

        return runs_are_merge_equiv

    @staticmethod
    def _normalize_char(char: str) -> str:
        if not char.isnumeric():
            return char

        float_value = unicodedata.numeric(char)
        if float_value.is_integer():
            return str(int(float_value))

        max_delta = 0.0001
        for i in range(1, 100):
            multiple = i * float_value
            if abs(multiple - int(multiple)) < max_delta:
                return f"{int(multiple)}/{i}"

        return str(float_value)

    def _normalize(self, s: str) -> str:
        normalized_str = "".join(self._normalize_char(c) for c in s)
        return normalized_str

    def _get_digit_runs(self, s: str) -> Counter[str]:
        groups: Counter[str] = Counter()
        for match in self.merge_pattern.finditer(s):
            group = "".join(c for c in match.group(0) if c not in self.strictly_ordered_separators)
            groups[group] += 1
        return groups
