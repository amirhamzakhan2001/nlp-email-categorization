
import re
import html
import ftfy

from src.fetch_gmail.body_cleaner import (
    minimal_text_fix,
    remove_zero_width_and_control
)

_ALLOWED_PUNCT_CHARS = ".,?:;'-@()[]{}/*$%#!=+<>~`|\\"
PCT_BLOB_RE = re.compile(r'%[0-9A-Fa-f]{2}(?:%[0-9A-Fa-f]{2}){6,}')
LONG_GARBLE_RE = re.compile( r'[^\w\s' + re.escape(_ALLOWED_PUNCT_CHARS) + r']{8,}', flags=re.UNICODE )
_SUBJ_JUNK_RE = re.compile(r'\b\S{120,}\b')

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


def clean_subject_for_csv(subj: str) -> str:
    subj = (subj or "")
    subj = html.unescape(subj)
    try:
        subj = ftfy.fix_text(subj)
    except Exception:
        pass
    subj = _SUBJ_JUNK_RE.sub(' ', subj)
    subj = minimal_text_fix(subj)
    return subj

def clean_subject_strip_emoji(subj: str, max_len: int = 512) -> str:
    s = subj or ""
    s = html.unescape(s)
    try:
        s = ftfy.fix_text(s)
    except Exception:
        pass
    s = remove_zero_width_and_control(s)
    s = EMOJI_RE.sub(" ", s)
    s = PCT_BLOB_RE.sub(" ", s)
    s = LONG_GARBLE_RE.sub(" ", s)
    # remove empty brackets and brackets with only punctuation/space
    s = re.sub(r'\(\s*\)', ' ', s)
    s = re.sub(r'\[\s*\]', ' ', s)
    s = re.sub(r'\{\s*\}', ' ', s)
    s = re.sub(r'<\s*>', ' ', s)
    s = re.sub(r'\(\s*[^A-Za-z0-9]+\s*\)', ' ', s)
    # solitary punctuation tokens and pipes
    s = re.sub(r'\b[|,._\-~]+\b', ' ', s)
    s = re.sub(r'[!?.,|/_\-]{2,}', ' ', s)
    s = re.sub(r'^[^\w]+|[^\w]+$', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = minimal_text_fix(s)
    s = re.sub(r'\S{200,}', ' ', s)
    if max_len and len(s) > max_len:
        s = s[:max_len].rstrip()
    return s
