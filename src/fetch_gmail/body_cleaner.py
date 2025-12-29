import re
import html
from typing import Optional
import ftfy
from bs4 import BeautifulSoup
import mailparser
from readability import Document  
from email import message_from_bytes
from email_reply_parser import EmailReplyParser

from src.config.fetch_config import (
    COLLAPSE_NEWLINES, LOWERCASE_OUTPUT,
    TRUNCATE_CHARS, MAX_TOKEN_LEN,
    REMOVE_EMOJIS, REMOVE_LONG_TOKENS,
    REMOVE_REPEATED_PUNCT, REPLACE_PERCENT_BLOBS,
    REDACT_EMAIL, EMAIL_TOKEN, REDACT_PHONE, PHONE_TOKEN,
    REPLACE_URL_WITH_TOKEN, URL_TOKEN,
    REMOVE_CURRENCY_LINES
)

_BOILERPLATE_PATTERNS = [
    r'unsubscribe', r'view in browser', r'privacy policy', r'terms apply', r'no cost emi',
    r'insta(nt)? cashback', r'click here', r'if you prefer not to receive',
    r'all rights reserved', r'trade in', r'pre[- ]?order', r'buy', r'visit our store',
    r'find a store', r'apple store', r'customer care', r'payment', r'learn more',
    r'offer valid', r'eligible', r'card', r'corporate employee', r'we apologise', r'we apologize',
]

_CURRENCY_CHARS = ["₹", "Rs.", "Rs", "$", "€", "£", "¥", "INR"]

URL_RE = re.compile(
    r'(?:<\s*(?:https?://[^\s<>"]+|www\.[^\s<>"]+|mailto:[^\s<>"]+)\s*>)'
    r'|https?://[^\s<>"\'\)\]]+|www\.[^\s<>"\'\)\]]+|mailto:[^\s<>"\'\)\]]+'
    , flags=re.IGNORECASE
)

TOKEN_RE = re.compile(r"^['\"]?\s*<\s*(url|email|phone|img|image|attachment)\s*>\s*['\"]?$", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', flags=re.IGNORECASE)
PHONE_RE = re.compile(
    r'(?:(?:\+?\d{1,3}[\s-])?(?:\(\d{2,4}\)|\d{2,4})[\s-]?)?\d{3,4}[\s-]?\d{3,6}',
    flags=re.IGNORECASE
)

EMOJI_RE = re.compile(
    "[" 
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)

REPEAT_PUNCT_RE = re.compile(r'([!?.\-_,;:])\1{2,}')
LONG_TOKEN_RE = re.compile(r'\S{' + str(MAX_TOKEN_LEN) + r',}')

_QUOTE_SEPARATORS = [
    r'^On .* wrote:$', r'^From: .*$', r'^Sent: .*$', r'^>.*$', r'^-----Original Message-----',
    r'^__+$', r'^-- $', r'^Regards,', r'^Best regards,', r'^Thanks,', r'^Cheers,'
]
_QUOTE_SEPARATORS_RE = re.compile('|'.join(_QUOTE_SEPARATORS), flags=re.IGNORECASE | re.MULTILINE)

_FOOTER_PATTERNS = [
    r'copyright', r'©', r'all rights reserved', r'registered trademark', r'trademark',
    r'visit our website', r'view in browser', r'unsubscribe', r'customer care', r'privacy policy',
    r'linkedin corporation', r'linkedin and the linkedin logo', r'accepted invitation', r'help:',
]

_ADDRESS_RE = re.compile(
    r'(?:(?:p\.?o\.?\s*box|po box)\s*\d+)|'
    r'(?:\d{1,5}\s+\w+(?:\s+\w+){0,6}\s+(?:avenue|ave|street|st|road|rd|boulevard|blvd|lane|ln|drive|dr)\b)|'
    r'(?:[A-Za-z .\-]+,\s*[A-Za-z]{2}\s*\d{5}(?:-\d{4})?)',
    flags=re.IGNORECASE
)

_SHORT_FOOTER_RE = re.compile(r'^(?:.*\b(?:' + r'|'.join([
    'copyright', '©', 'registered trademark', 'linkedin corporation', 'linkedin'
]) + r')\b.*)$', flags=re.IGNORECASE)

_READABILITY_MIN_LEN = 120


_ALLOWED_PUNCT_CHARS = ".,?:;'-@()[]{}/*$%#!=+<>~`|\\"
PCT_BLOB_RE = re.compile(r'%[0-9A-Fa-f]{2}(?:%[0-9A-Fa-f]{2}){6,}')
LONG_GARBLE_RE = re.compile( r'[^\w\s' + re.escape(_ALLOWED_PUNCT_CHARS) + r']{8,}', flags=re.UNICODE )
_SUBJ_JUNK_RE = re.compile(r'\b\S{120,}\b')

_ZERO_WIDTH = ["\u200B", "\u200C", "\u200D", "\uFEFF", "\u2060"]
_CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')



def preprocess_email_body(raw_body: Optional[str],
                          collapse_newlines: bool = COLLAPSE_NEWLINES,
                          lowercase: bool = LOWERCASE_OUTPUT,
                          max_chars: Optional[int] = TRUNCATE_CHARS) -> str:
    """
    Full pipeline to clean an email body string.
    """
    s = raw_body or ""
    # 1) Unescape + fix encoding
    s = html.unescape(s)
    try:
        s = ftfy.fix_text(s)
    except Exception:
        pass

    # 2) If looks like HTML -> extract text early
    if '<' in s and '>' in s:
        s = _html_to_text(s)

    # 3) Replace URLs, emails, phones first (so tokens are visible to line-based heuristics)
    s = URL_RE.sub(URL_TOKEN, s)
    if REDACT_EMAIL:
        s = EMAIL_RE.sub(EMAIL_TOKEN, s)
    if REDACT_PHONE:
        s = PHONE_RE.sub(PHONE_TOKEN, s)

    # 4) Remove angle-bracket tokens leftover like <URL> or <...>
    s = _remove_angle_bracket_tokens(s)

    # 5) Remove percent-encoded blobs / long garble
    if REPLACE_PERCENT_BLOBS:
        s = _remove_percent_blobs(s)
    s = _remove_long_garble(s)

    # 6) Strip quoted replies / signatures (EmailReplyParser works better once tokens removed)
    s = _strip_quoted_reply(s)

    # 7) Remove emojis
    if REMOVE_EMOJIS:
        s = _remove_emojis(s)

    # 8) Remove extremely long tokens
    if REMOVE_LONG_TOKENS:
        s = _remove_long_tokens(s)

    # 9) Collapse repeated punctuation
    if REMOVE_REPEATED_PUNCT:
        s = _collapse_repeated_punct(s)

    # 10) Strip boilerplate/product/price lines (now sees tokens and normalized text)
    s = _strip_boilerplate_lines(s)

    # 11) Normalize whitespace / newlines
    s = _normalize_whitespace(s, collapse_newlines=collapse_newlines)

    # 12) Lowercase optionally
    if lowercase:
        s = s.lower()

    # 13) Final tidyups and truncation
    s = re.sub(r'\s{2,}', ' ', s).strip()
    if max_chars and isinstance(max_chars, int) and max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars].rstrip()

    return s

def _remove_angle_bracket_tokens(s: str) -> str:
    """Remove tokens like <...> entirely (useful for stray <URL> or <...> placeholders)."""
    if not s:
        return s
    # remove tokens like <URL>, <EMAIL>, <0xabc123> etc, but don't aggressively remove <a> html tags (we already converted html -> text)
    # we keep only angle groups with 2+ upper-case letters or URLs already handled, to be conservative
    # Also remove leftover literal '<' '>' pairs containing short garbage
    s = re.sub(r'<\s*[A-Za-z0-9%:+\/\._@-]{1,200}\s*>', ' ', s)
    return s

# ---------- helper functions ----------
def _is_boilerplate_line(line: str) -> bool:
    if not line or len(line.strip()) < 2:
        return True

    raw = line
    low = raw.lower().strip()

    # 1) direct token-only matches: "<URL>", "<EMAIL>", etc.
    try:
        if TOKEN_RE.fullmatch(raw.strip()):
            return True
    except NameError:
        # if TOKEN_RE not defined, skip this check (but you should add TOKEN_RE)
        pass

    # 2) if line contains the URL token or email token and is short, treat as boilerplate
    if any(tok in low for tok in ['<url>', '<email>', '<phone>', '<img>', '<image>', '<attachment>']) and len(low) < 200:
        return True

    # 3) explicit footer/boilerplate keywords (copyright, unsubscribe, privacy, trademark, vendor names)
    for p in _FOOTER_PATTERNS + _BOILERPLATE_PATTERNS:
        if re.search(p, low):
            return True

    # 4) address-like detection: street addresses, PO Box, City, ST ZIP
    if _ADDRESS_RE.search(raw):
        return True

    # 5) short one-line footer tokens like "© 2025 LinkedIn Corporation, Sunnyvale, CA..."
    if _SHORT_FOOTER_RE.match(raw.strip()):
        return True

    # 6) currency/price lines (optional behavior handled elsewhere but keep here for safety)
    if REMOVE_CURRENCY_LINES and any(c in raw for c in _CURRENCY_CHARS):
        if len(raw) < 200:
            return True

    # 7) lines with many non-alphanumeric characters are likely separators / junk
    alpha = len(re.findall(r'\w', raw))
    non_alpha = len(re.findall(r'\W', raw))
    if alpha < 6 and non_alpha > alpha:
        return True

    # 8) lines that are mostly punctuation / whitespace
    if re.match(r'^[\s\W_]{3,}$', raw):
        return True

    return False

def _html_to_text(s: str) -> str:
    if not s:
        return ""
    try:
        if '<' in s and '>' in s and len(s) > _READABILITY_MIN_LEN:
            doc = Document(s)
            summary = doc.summary()
            bs = BeautifulSoup(summary, "html.parser")
            txt = bs.get_text(separator="\n", strip=True)
            if txt and len(txt) > 20:
                return txt
        bs = BeautifulSoup(s, "html.parser")
        for tag in bs(['script', 'style', 'head', 'meta', 'noscript', 'iframe', 'svg']):
            tag.decompose()
        return bs.get_text(separator="\n", strip=True)
    except Exception:
        return re.sub(r'<[^>]+>', ' ', s)

def _collapse_repeated_punct(s: str) -> str:
    if not s:
        return s
    return REPEAT_PUNCT_RE.sub(r'\1', s)

def _remove_long_tokens(s: str) -> str:
    if not s:
        return s
    return LONG_TOKEN_RE.sub(' ', s)

def _remove_percent_blobs(s: str) -> str:
    if not s:
        return s
    return PCT_BLOB_RE.sub(' ', s)

def _remove_long_garble(s: str) -> str:
    if not s:
        return s
    return LONG_GARBLE_RE.sub(' ', s)

def _remove_emojis(s: str) -> str:
    if not s:
        return s
    return EMOJI_RE.sub(' ', s)

def _replace_urls(s: str) -> str:
    if not s:
        return s
    if REPLACE_URL_WITH_TOKEN:
        return URL_RE.sub(URL_TOKEN, s)
    return URL_RE.sub(' ', s)

def _redact_emails(s: str) -> str:
    if not s:
        return s
    if REDACT_EMAIL:
        return EMAIL_RE.sub(EMAIL_TOKEN, s)
    return s

def _redact_phones(s: str) -> str:
    if not s:
        return s
    if REDACT_PHONE:
        return PHONE_RE.sub(PHONE_TOKEN, s)
    return s

def _normalize_whitespace(s: str, collapse_newlines: bool = True) -> str:
    if s is None:
        return ""
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    if collapse_newlines:
        s = re.sub(r'\n+', ' ', s)
    else:
        s = re.sub(r'\n{3,}', '\n\n', s)
    s = re.sub(r'[ \t]{2,}', ' ', s)
    return s.strip()


def _strip_boilerplate_lines(s: str) -> str:
    if not s:
        return ""
    lines = []
    for line in s.splitlines():
        if not line or _is_boilerplate_line(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()

def _strip_quoted_reply(s: str) -> str:
    if not s:
        return ""
    try:
        parsed = EmailReplyParser.parse_reply(s)
        if parsed and parsed.strip():
            return parsed.strip()
    except Exception:
        pass
    parts = re.split(r'(?m)^(On .* wrote:|From: .*|Sent: .*|>.*|-----Original Message-----)', s)
    if parts:
        return parts[0].strip()
    return s.strip()


def remove_zero_width_and_control(txt: str) -> str:
    if txt is None:
        return ""
    s = str(txt)
    for z in _ZERO_WIDTH:
        s = s.replace(z, "")
    s = _CTRL_RE.sub(" ", s)
    # Normalize line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # collapse many blank lines
    s = re.sub(r'\n{3,}', '\n\n', s)
    # strip leading/trailing whitespace
    return s.strip()

def sanitize_field_for_csv(field: str, collapse_newlines=False) -> str:
    """Clean field: remove zombies, collapse repeated spaces. If collapse_newlines True, convert newlines to spaces."""
    s = remove_zero_width_and_control(field)
    s = html.unescape(s)
    try:
        s = ftfy.fix_text(s)
    except Exception:
        pass
    # reduce repeated whitespace
    if collapse_newlines:
        s = s.replace("\n", " ")
    # collapse multiple spaces/tabs
    s = re.sub(r'[ \t]{2,}', ' ', s)
    return s.strip()

def minimal_text_fix(text: str) -> str:
    if not text:
        return ""
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8', errors='replace')
        except Exception:
            text = text.decode('latin1', errors='replace')
    text = html.unescape(text)
    try:
        text = ftfy.fix_text(text)
    except Exception:
        pass
    text = text.replace("\uFFFD", " ")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    text = PCT_BLOB_RE.sub(" ", text)
    text = LONG_GARBLE_RE.sub(" ", text)
    text = text.replace('\r\n', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()



