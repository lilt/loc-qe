from enum import StrEnum

# Languages whose scripts have upper/lower case distinction
_CASED_LANGS = {"en", "fr", "de", "es", "pl", "fi", "cs", "ru", "uk", "is", "et", "it", "sr", "pt", "nl", "sv", "da", "no", "ro", "hu", "bg", "hr", "sk", "sl", "lt", "lv", "el", "ka", "hy"}

# Languages for which we know the correct period symbol for the hallucinated-period check
_PERIOD_SYMBOL_LANGS = {"en", "fr", "de", "es", "pl", "fi", "ja", "zh", "ru", "ar", "hi", "cs", "uk", "is", "et", "it", "sr", "pt", "nl", "sv", "da", "no", "ro", "hu", "bg", "hr", "sk", "sl", "lt", "lv", "ko"}


class ChecklistCategory(StrEnum):
    copy_urls = "copy-urls"
    long_number = "long-number"
    leading_trailing_space = "leading-trailing-space"
    hallucinated_period = "hallucinated-period"
    all_caps = "all-caps"
    spanish_punct = "spanish-punct"
    french_nbsp = "french-nbsp"
    unicode_spaces = "unicode-spaces"



CATEGORY_TO_LANG_PAIRS = {
    # Language-agnostic categories: any LP is fine
    ChecklistCategory.copy_urls: None,
    ChecklistCategory.long_number: None,
    ChecklistCategory.leading_trailing_space: None,
    ChecklistCategory.unicode_spaces: None,
    # Target-language-constrained categories
    ChecklistCategory.hallucinated_period: _PERIOD_SYMBOL_LANGS,
    ChecklistCategory.all_caps: _CASED_LANGS,
    # Single-language categories
    ChecklistCategory.spanish_punct: {"es"},
    ChecklistCategory.french_nbsp: {"fr"},
}


def category_applies_to_language_pair(
    category: ChecklistCategory, language_pair: str
) -> bool:
    """Check whether *category* applies to *language_pair* (format ``xx-yy``)."""
    constraint = CATEGORY_TO_LANG_PAIRS[category]
    if constraint is None:
        return True
    tgt_lang = language_pair.split("-")[-1]
    return tgt_lang in constraint
