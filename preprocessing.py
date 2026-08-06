import re
import unicodedata
import logging
from bs4 import BeautifulSoup
import emoji
from langdetect import detect, LangDetectException
import spacy

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global spacy nlp object placeholder
_nlp = None

def clean_html(text: str) -> str:
    """Removes HTML tags from the text."""
    try:
        # Using BeautifulSoup to clean HTML
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    except Exception as e:
        logger.warning(f"HTML parsing failed: {e}. Falling back to regex.")
        # Fallback to regex
        return re.sub(r'<[^>]*>', '', text)

def clean_urls(text: str) -> str:
    """Removes URLs from the text."""
    # Matches http/https and www.
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    return text

def clean_mentions(text: str) -> str:
    """Removes Twitter/Social media @mentions."""
    return re.sub(r'@\w+', '', text)

def normalize_unicode(text: str) -> str:
    """Normalizes Unicode characters to NFKC form."""
    return unicodedata.normalize('NFKC', text)

def normalize_emojis(text: str) -> str:
    """Converts emojis to text representation."""
    try:
        # Convert emoji to text (e.g. :laughing_face: -> laughing face)
        demojized = emoji.demojize(text, delimiters=(" ", " "))
        # Replace underscores with spaces
        return demojized.replace("_", " ").replace(":", "")
    except Exception as e:
        logger.warning(f"Emoji normalization failed: {e}")
        return text

def normalize_repeated_chars(text: str) -> str:
    """Compresses sequences of 3 or more identical characters to 2."""
    # E.g., "looooove" -> "loove"
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def split_camel_case(text: str) -> str:
    """Helper to split camel case words, useful for hashtags."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).strip()

def handle_hashtags(text: str) -> str:
    """Extracts and splits hashtags into words."""
    # Find all hashtags
    hashtags = re.findall(r'#\w+', text)
    for tag in hashtags:
        # Remove '#' and split camel case
        word = tag[1:]
        split_word = split_camel_case(word)
        text = text.replace(tag, split_word)
    return text

def normalize_whitespace(text: str) -> str:
    """Removes leading, trailing, and duplicate whitespaces."""
    return re.sub(r'\s+', ' ', text).strip()

def detect_language(text: str) -> str:
    """Detects the language of the text. Returns 'unknown' on failure."""
    if not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def spell_correction_light(text: str) -> str:
    """Lightweight spelling correction for common internet slangs/obfuscations."""
    # We map common obfuscations used to bypass hate speech filters
    obfuscations = {
        r'\ba[s\$][s\$]\b': 'ass',
        r'\bb[i\!][t\$]ch\b': 'bitch',
        r'\bf[u\*]ck\b': 'fuck',
        r'\bh[a4]te\b': 'hate',
        r'\bl0ve\b': 'love',
        r'\bp[e3]n[i1]s\b': 'penis',
        r'\br[a4]p[e3]\b': 'rape',
        r'\b[s\$]h[i1]t\b': 'shit',
    }
    for pattern, replacement in obfuscations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_html: bool = True,
    normalize_emoji: bool = True,
    normalize_repeated: bool = True,
    norm_unicode: bool = True,
    process_hashtags: bool = True,
    language_filter: str = "en",
    spell_correct: bool = True,
    mask_names: bool = True
) -> str:
    """
    Main preprocessing pipeline function.
    Returns cleaned text, or empty string if it fails filters (e.g. non-English).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if remove_html:
        text = clean_html(text)

    if norm_unicode:
        text = normalize_unicode(text)

    if remove_urls:
        text = clean_urls(text)

    if remove_mentions:
        text = clean_mentions(text)

    if process_hashtags:
        text = handle_hashtags(text)

    if normalize_emoji:
        text = normalize_emojis(text)

    if normalize_repeated:
        text = normalize_repeated_chars(text)

    if spell_correct:
        text = spell_correction_light(text)

    if mask_names:
        global _nlp
        if _nlp is None:
            try:
                # Load SpaCy model lazily
                _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except Exception as e:
                logger.warning(f"Failed to load SpaCy model for name masking: {e}")
        
        if _nlp is not None:
            try:
                doc = _nlp(text)
                person_ents = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
                tokens = []
                for token in doc:
                    is_person_name = (
                        token.text in person_ents or 
                        token.ent_type_ == "PERSON" or
                        (token.pos_ == "PROPN" and token.text.istitle() and token.ent_type_ not in ("GPE", "NORP", "ORG", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"))
                    )
                    if is_person_name:
                        tokens.append("someone")
                    else:
                        tokens.append(token.text)
                text = " ".join(tokens)
            except Exception as e:
                logger.warning(f"Name masking failed: {e}")

    if lowercase:
        text = text.lower()

    text = normalize_whitespace(text)

    # Language filter
    if language_filter:
        lang = detect_language(text)
        if lang != language_filter and lang != "unknown":
            # If it detects a non-English language definitely, skip it
            return ""

    return text
