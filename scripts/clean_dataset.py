import os
import sys
import re
import unicodedata
import pandas as pd
import emoji
import ftfy
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.config import load_config
from scripts.utils import setup_logging, save_json

logger = setup_logging("clean_dataset")

# Pre-compile regexes for high performance
HTML_TAG_RE = re.compile(r'<[^>]*>')
URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
MENTION_RE = re.compile(r'@\w+')
REPEATED_CHARS_RE = re.compile(r'(.)\1{2,}')
WHITESPACE_RE = re.compile(r'\s+')
INVISIBLE_CHARS_RE = re.compile(r'[\x00-\x1F\x7F-\x9F\u200b-\u200d\u200f\u202a-\u202e]')

# Obfuscations mappings to resolve bypass attempts
OBFUSCATIONS = {
    r'\ba[s\$][s\$]\b': 'ass',
    r'\bb[i\!][t\$]ch\b': 'bitch',
    r'\bf[u\*]ck\b': 'fuck',
    r'\bh[a4]te\b': 'hate',
    r'\bl0ve\b': 'love',
    r'\bp[e3]n[i1]s\b': 'penis',
    r'\br[a4]p[e3]\b': 'rape',
    r'\b[s\$]h[i1]t\b': 'shit',
    r'\bn[i1]gg[a4]r?\b': 'nigger',
    r'\bf[a4]gg[o0]t\b': 'faggot'
}

def clean_html(text: str) -> str:
    """Removes HTML tags from the text."""
    if not text:
        return ""
    try:
        # Check if text looks like HTML first
        if "<" in text and ">" in text:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text()
        return text
    except Exception:
        return HTML_TAG_RE.sub('', text)

def clean_urls(text: str) -> str:
    """Removes URLs from the text."""
    return URL_RE.sub('', text)

def clean_emails(text: str) -> str:
    """Removes email addresses."""
    return EMAIL_RE.sub('', text)

def clean_mentions(text: str, action: str = "remove") -> str:
    """Removes or replaces mentions."""
    if action == "replace":
        return MENTION_RE.sub('@user', text)
    return MENTION_RE.sub('', text)

def normalize_unicode(text: str) -> str:
    """Fixes encoding issues using ftfy and normalizes Unicode to NFKC."""
    fixed = ftfy.fix_text(text)
    return unicodedata.normalize('NFKC', fixed)

def normalize_emojis(text: str) -> str:
    """Converts emojis to text representation."""
    try:
        demojized = emoji.demojize(text, delimiters=(" ", " "))
        return demojized.replace("_", " ").replace(":", "")
    except Exception:
        return text

def normalize_repeated_chars(text: str) -> str:
    """Compresses sequences of 3 or more identical characters to 2."""
    return REPEATED_CHARS_RE.sub(r'\1\1', text)

def split_camel_case(text: str) -> str:
    """Helper to split camel case words, useful for hashtags."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).strip()

def handle_hashtags(text: str) -> str:
    """Extracts and splits hashtags into words."""
    # Find all hashtags
    hashtags = re.findall(r'#\w+', text)
    for tag in hashtags:
        word = tag[1:]
        split_word = split_camel_case(word)
        text = text.replace(tag, split_word)
    return text

def normalize_quotes(text: str) -> str:
    """Converts curly quotes and apostrophes to standard straight ones."""
    # Double quotes
    text = re.sub(r'[\u201c\u201d\u201e\u201f\u00ab\u00bb]', '"', text)
    # Single quotes
    text = re.sub(r'[\u2018\u2019\u201a\u201b\u00b4\u2032]', "'", text)
    return text

def normalize_punctuation(text: str) -> str:
    """Normalizes repeated punctuation (e.g. !!! -> !!, ??? -> ??)."""
    text = re.sub(r'!{2,}', '!!', text)
    text = re.sub(r'\?{2,}', '??', text)
    text = re.sub(r'\.{3,}', '...', text)
    return text

def handle_reddit_formatting(text: str) -> str:
    """Cleans Reddit specific formatting like ^, > quote marks, &gt;, etc."""
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    # Remove superscript ^
    text = text.replace("^", "")
    # Remove Reddit table syntax if any
    text = re.sub(r'\|.*\|', '', text)
    return text

def handle_markdown_formatting(text: str) -> str:
    """Removes standard markdown formatting characters like *bold*, _italic_, [link](url), etc."""
    # Remove link formatting: [text](url) -> text
    text = re.compile(r'\[(.*?)\]\((.*?)\)').sub(r'\1', text)
    # Remove emphasis/bold marks: * or _
    text = re.sub(r'[\*_`~]', '', text)
    return text

def remove_invisible_characters(text: str) -> str:
    """Removes control chars, zero-width spaces, and other invisible characters."""
    return INVISIBLE_CHARS_RE.sub('', text)

def detect_language(text: str) -> str:
    """Detects the language of the text. Returns 'unknown' on failure."""
    if not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def spell_correction_light(text: str) -> str:
    """Lightweight spelling correction for common obfuscations."""
    for pattern, replacement in OBFUSCATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def clean_text_sample(text: str, cfg) -> str:
    """Applies the complete preprocessing cleaning pipeline to a single string."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode Fixes
    if cfg.remove_corrupted_unicode:
        text = normalize_unicode(text)
    else:
        text = unicodedata.normalize('NFKC', text)

    # 2. HTML and Markdown tag stripping
    if cfg.remove_html:
        text = clean_html(text)
        
    if cfg.handle_markdown_formatting:
        text = handle_markdown_formatting(text)
        
    if cfg.handle_reddit_formatting:
        text = handle_reddit_formatting(text)

    # 3. URL and Email stripping
    if cfg.remove_urls:
        text = clean_urls(text)
        
    if cfg.remove_emails:
        text = clean_emails(text)

    # 4. Social handles
    if cfg.remove_mentions:
        text = clean_mentions(text, action="remove")

    # 5. Hashtag camel case expansion
    if cfg.handle_hashtags:
        text = handle_hashtags(text)

    # 6. Emojis conversions
    if cfg.normalize_emojis:
        text = normalize_emojis(text)

    # 7. Repeated character squeezing (e.g. looooove -> loove)
    if cfg.normalize_repeated_chars:
        text = normalize_repeated_chars(text)

    # 8. Punctuation and quotes normalizations
    if cfg.normalize_quotes:
        text = normalize_quotes(text)
        
    if cfg.normalize_punctuation:
        text = normalize_punctuation(text)

    # 9. Invisible chars
    if cfg.remove_invisible_chars:
        text = remove_invisible_characters(text)

    # 10. Obfuscation expansion (e.g., f*ck -> fuck)
    text = spell_correction_light(text)

    # 11. Lowercase normalization
    if cfg.lowercase:
        text = text.lower()

    # 12. Whitespace squeezing
    if cfg.normalize_whitespace:
        text = WHITESPACE_RE.sub(' ', text).strip()
    else:
        text = text.strip()

    return text

def clean_dataset_file(src_name: str, proc_dir: str, cfg) -> dict:
    src_path = os.path.join(proc_dir, f"{src_name}_normalized.csv")
    if not os.path.exists(src_path):
        logger.warning(f"Normalized file for {src_name} not found.")
        return None
        
    df = pd.read_csv(src_path)
    total_samples = len(df)
    
    cleaned_records = []
    skipped_non_english = 0
    skipped_empty = 0
    
    logger.info(f"Cleaning {src_name} ({total_samples} rows)...")
    
    for row in tqdm(df.itertuples(index=False), total=total_samples, desc=f"Cleaning {src_name}"):
        raw_text = str(row.text)
        label = row.label
        
        # Clean text
        cleaned_text = clean_text_sample(raw_text, cfg)
        
        if not cleaned_text:
            skipped_empty += 1
            continue
            
        # Language filtering
        if cfg.language_filter:
            lang = detect_language(cleaned_text)
            if lang != cfg.language_filter and lang != "unknown":
                skipped_non_english += 1
                continue
                
        cleaned_records.append({
            "text": cleaned_text,
            "label": label,
            "source": src_name
        })
        
    cleaned_df = pd.DataFrame(cleaned_records)
    dest_path = os.path.join(proc_dir, f"{src_name}_cleaned.csv")
    cleaned_df.to_csv(dest_path, index=False, encoding="utf-8")
    
    stats = {
        "dataset_name": src_name,
        "input_samples": total_samples,
        "output_samples": len(cleaned_df),
        "skipped_empty": skipped_empty,
        "skipped_non_english": skipped_non_english,
        "reduction_rate_pct": round((total_samples - len(cleaned_df)) / total_samples * 100, 2) if total_samples > 0 else 0
    }
    
    logger.info(f"{src_name} cleaned successfully: {len(cleaned_df)} rows. Reduction rate: {stats['reduction_rate_pct']}%")
    return stats

def main():
    config = load_config()
    proc_dir = config.dataset.processed_dir
    reports_dir = config.dataset.reports_dir
    cfg = config.preprocessing
    
    logger.info("--- STARTING DATASET CLEANING AND LANGUAGE FILTERING ---")
    
    datasets = ["davidson", "olid", "hatexplain", "jigsaw", "civil_comments"]
    all_stats = {}
    
    for d in datasets:
        stats = clean_dataset_file(d, proc_dir, cfg)
        if stats:
            all_stats[d] = stats
            
    # Generate cleaning report
    report_md = "# Text Cleaning and Preprocessing Report\n\n"
    report_md += "This report summarizes the results of the preprocessing and text cleaning phase, including language filtering.\n\n"
    
    report_md += "| Dataset | Input Rows | Cleaned Output Rows | Removed (Empty) | Removed (Non-English) | Reduction Rate |\n"
    report_md += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for name, stat in all_stats.items():
        report_md += f"| **{stat['dataset_name']}** | {stat['input_samples']:,} | {stat['output_samples']:,} | {stat['skipped_empty']:,} | {stat['skipped_non_english']:,} | {stat['reduction_rate_pct']}% |\n"
        
    report_md += "\n## Cleaning Pipeline Configuration\n"
    report_md += f"- **Unicode Normalization**: Yes (ftfy NFKC)\n"
    report_md += f"- **Lowercasing**: {cfg.lowercase}\n"
    report_md += f"- **URL Stripping**: {cfg.remove_urls}\n"
    report_md += f"- **Email Stripping**: {cfg.remove_emails}\n"
    report_md += f"- **HTML Stripping**: {cfg.remove_html}\n"
    report_md += f"- **Social Media Mentions**: {cfg.remove_mentions}\n"
    report_md += f"- **Hashtag Splitting**: {cfg.handle_hashtags}\n"
    report_md += f"- **Emoji Normalization (Demojize)**: {cfg.normalize_emojis}\n"
    report_md += f"- **Repeated Characters Normalization**: {cfg.normalize_repeated_chars}\n"
    report_md += f"- **Reddit Specific Formatting Cleanup**: {cfg.handle_reddit_formatting}\n"
    report_md += f"- **Markdown Formatting Cleanup**: {cfg.handle_markdown_formatting}\n"
    report_md += f"- **Obfuscated Slang Translation (Regex-based)**: Enabled (e.g. a$$ -> ass)\n"
    report_md += f"- **Language Filtering**: Keep English only ({cfg.language_filter})\n"
    
    report_path = os.path.join(reports_dir, "cleaning_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    logger.info(f"Cleaning report saved to {report_path}")
    logger.info("--- DATASET CLEANING AND LANGUAGE FILTERING COMPLETED ---")

if __name__ == "__main__":
    main()
