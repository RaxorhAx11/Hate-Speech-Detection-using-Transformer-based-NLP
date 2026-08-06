# Text Cleaning and Preprocessing Report

This report summarizes the results of the preprocessing and text cleaning phase, including language filtering.

| Dataset | Input Rows | Cleaned Output Rows | Removed (Empty) | Removed (Non-English) | Reduction Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **davidson** | 24,783 | 24,757 | 0 | 26 | 0.1% |
| **olid** | 14,047 | 14,042 | 0 | 5 | 0.04% |
| **hatexplain** | 20,148 | 20,123 | 0 | 25 | 0.12% |
| **jigsaw** | 159,571 | 159,215 | 10 | 346 | 0.22% |
| **civil_comments** | 49,205 | 49,153 | 47 | 5 | 0.11% |

## Cleaning Pipeline Configuration
- **Unicode Normalization**: Yes (ftfy NFKC)
- **Lowercasing**: True
- **URL Stripping**: True
- **Email Stripping**: True
- **HTML Stripping**: True
- **Social Media Mentions**: True
- **Hashtag Splitting**: True
- **Emoji Normalization (Demojize)**: True
- **Repeated Characters Normalization**: True
- **Reddit Specific Formatting Cleanup**: True
- **Markdown Formatting Cleanup**: True
- **Obfuscated Slang Translation (Regex-based)**: Enabled (e.g. a$$ -> ass)
- **Language Filtering**: Keep English only (en)
