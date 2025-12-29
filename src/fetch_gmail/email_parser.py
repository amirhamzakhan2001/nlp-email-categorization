
import re
import html
import mailparser
from bs4 import BeautifulSoup
from readability import Document
from email import message_from_bytes
from email_reply_parser import EmailReplyParser

from src.fetch_gmail.body_cleaner import minimal_text_fix

def normalize_amp_to_html(html_content: str) -> str:
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(['script', 'style', 'link']):
        tag.decompose()
    for amp in soup.find_all(re.compile(r'^amp-', re.I)):
        if amp.name.lower() == 'amp-img':
            new_tag = soup.new_tag("img")
            if amp.has_attr("src"):
                new_tag['src'] = amp['src']
            if amp.has_attr("alt"):
                new_tag['alt'] = amp['alt']
            amp.replace_with(new_tag)
        else:
            amp.unwrap()
    for tag in soup.find_all(True):
        for a in list(tag.attrs.keys()):
            if a.startswith('amp-') or a in ['layout', 'sandbox', 'referrerpolicy']:
                try:
                    del tag.attrs[a]
                except Exception:
                    pass
    return str(soup)

def extract_html_fragments(html_content: str):
    if not html_content:
        return []
    unesc = html.unescape(html_content)
    frags = re.findall(r'(<html[\s\S]*?</html>)', unesc, flags=re.IGNORECASE)
    if frags:
        return frags
    return [html_content]

def readability_extract(html_content: str) -> str:
    if not html_content:
        return ""
    try:
        doc = Document(html_content)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        txt = soup.get_text(separator="\n", strip=True)
        return txt
    except Exception:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n", strip=True)

def clean_html_fragment(html_content: str) -> str:
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(['script','style','img','iframe','head','meta','noscript','footer','form','svg']):
        tag.decompose()
    for table in soup.find_all('table'):
        table.decompose()
    for tag in soup.find_all(True):
        for attr in ['style','class','id']:
            if tag.has_attr(attr):
                del tag[attr]
        for a in [k for k in list(tag.attrs.keys()) if k.startswith('data-') or k.startswith('aria-')]:
            if a in tag.attrs:
                del tag.attrs[a]
    text = soup.get_text(separator="\n", strip=True)
    cleaned_lines = []
    for line in text.splitlines():
        line_low = line.lower().strip()
        if not line_low:
            continue
        if any(k in line_low for k in ['unsubscribe','privacy','help','footer','login','copyright','disclaimer','view in browser']):
            continue
        if len(line_low) > 400:
            continue
        alpha_count = len(re.findall(r'\w', line_low))
        non_alpha_count = len(re.findall(r'\W', line_low))
        if alpha_count < non_alpha_count and alpha_count < 5:
            continue
        if re.search(r'font(-family)?|sans-serif|px;|http[s]?:\\/\\/', line_low):
            if re.match(r'^https?://', line_low) and len(line_low) < 200:
                cleaned_lines.append(line.strip())
            else:
                if 'http' in line_low and len(line_low) > 300:
                    continue
                if not re.search(r'\w', line_low):
                    continue
                cleaned_lines.append(line.strip())
        else:
            cleaned_lines.append(line.strip())
    clean_text = "\n".join(cleaned_lines)
    sig_pats = ["original message","--","best regards","sent from my iphone","regards,","cheers,","thanks,"]
    lower = clean_text.lower()
    cut = None
    for p in sig_pats:
        idx = lower.find(p)
        if idx != -1 and (cut is None or idx < cut):
            cut = idx
    if cut:
        clean_text = clean_text[:cut]
    clean_text = re.sub(r'\n{2,}', '\n\n', clean_text).strip()
    return clean_text

def extract_clean_body_from_raw(raw_bytes: bytes, msg_id: str = None) -> str:
    mp = None
    try:
        mp = mailparser.parse_from_bytes(raw_bytes)
    except Exception:
        mp = None

    text_plain = None
    html_content = None
    if mp:
        if getattr(mp, "text_plain", None):
            text_plain = "\n\n".join([p for p in mp.text_plain if p]).strip()
        if getattr(mp, "text_html", None):
            html_content = "\n\n".join([h for h in mp.text_html if h]).strip()
        if not text_plain and getattr(mp, "body", None):
            text_plain = (mp.body or "").strip()

    if not text_plain and not html_content:
        try:
            em = message_from_bytes(raw_bytes)
            if em.is_multipart():
                plain_parts = []
                html_parts = []
                for part in em.walk():
                    ctype = part.get_content_type()
                    disp = str(part.get("Content-Disposition") or "")
                    if ctype == "text/plain" and "attachment" not in disp:
                        payload = part.get_payload(decode=True)
                        if payload:
                            plain_parts.append(payload.decode(part.get_content_charset() or "utf-8", errors="replace"))
                if plain_parts:
                    text_plain = "\n\n".join(plain_parts).strip()
                else:
                    for part in em.walk():
                        if part.get_content_type() == "text/html":
                            payload = part.get_payload(decode=True)
                            if payload:
                                html_parts.append(payload.decode(part.get_content_charset() or "utf-8", errors="replace"))
                    if html_parts:
                        html_content = "\n\n".join(html_parts).strip()
            else:
                payload = em.get_payload(decode=True)
                if payload:
                    if em.get_content_type() == "text/plain":
                        text_plain = payload.decode(em.get_content_charset() or "utf-8", errors="replace")
                    elif em.get_content_type() == "text/html":
                        html_content = payload.decode(em.get_content_charset() or "utf-8", errors="replace")
        except Exception:
            pass

    candidate_text = None
    used_html = False
    raw_fragment = None
    if text_plain and len(text_plain.strip()) > 10:
        candidate_text = text_plain
    elif html_content:
        if '<amp-' in html_content.lower() or 'text/x-amp-html' in html_content.lower():
            html_content = normalize_amp_to_html(html_content)
        frags = extract_html_fragments(html_content)
        best_frag_text = ""
        for frag in frags:
            txt = readability_extract(frag)
            if not txt or len(txt) < 30:
                txt = clean_html_fragment(frag)
            if txt and len(txt) > len(best_frag_text):
                best_frag_text = txt
                raw_fragment = frag
        if best_frag_text:
            candidate_text = best_frag_text
            used_html = True
    else:
        try:
            candidate_text = raw_bytes.decode('utf-8', errors='replace')
        except Exception:
            candidate_text = str(raw_bytes)

    if not candidate_text:
        return ""

    if re.search(r'<[^>]+>', candidate_text):
        try:
            t_read = readability_extract(candidate_text)
        except Exception:
            t_read = ""
        if not t_read or len(t_read) < 30:
            t_read = clean_html_fragment(candidate_text)
        if not t_read or re.search(r'<[^>]+>', t_read):
            try:
                t_read = BeautifulSoup(candidate_text, "html.parser").get_text(separator="\n", strip=True)
            except Exception:
                t_read = re.sub(r'<[^>]+>', ' ', candidate_text)
        candidate_text = t_read

    try:
        parsed = EmailReplyParser.parse_reply(candidate_text)
        if parsed and parsed.strip():
            candidate_text = parsed.strip()
        else:
            candidate_text = re.split(r'(?m)^(On .* wrote:|From: .*|Sent: .*)', candidate_text)[0].strip()
    except Exception:
        candidate_text = re.split(r'(?m)^(On .* wrote:|From: .*|Sent: .*)', candidate_text)[0].strip()

    final_text = minimal_text_fix(candidate_text)
    return final_text
