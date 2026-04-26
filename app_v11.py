"""
Tarot Predictor V11
Methodology: homeostatic balancing via semantic resonance.

New in V11:
- Three-card hero strip at top (Wikimedia Rider-Waite images, rx rotated 180°)
- Three-column meanings comparison (Home | Away | MF) with full text
- Per-card MF Lookup sections showing column B + C from mf_lookup.xlsx
- Always-visible historical match tables for each card (no clicks):
    * matches as MF, as Home, as Away — both upright AND reversed
    * pair history (Home + Away together)
- Custom CSS styling: colour-coded sections per card role
- Removed: the "winners list" output (replaced by full match tables)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, urllib.parse
from analysis_engine import (analyse_match_full, analysis_to_prediction,
                              gather_history_text, ANALYSIS_PREFIX)
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Tarot Predictor V11", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CACHE_FILE      = "resonance_cache.json"
DATA_FILE       = "data.xlsx"
MF_LOOKUP_FILE  = "mf_lookup.xlsx"
API_MODEL = "claude-sonnet-4-6"

os.environ["ANTHROPIC_API_KEY"] = ""   # paste key here temporarily

BALANCE_MIN   = 0.35
ASPECTS_DELTA = 0.15

# Role colours (used in CSS and section headers)
COL_HOME = "#2e8b57"   # green
COL_AWAY = "#1e6fb8"   # blue
COL_MF   = "#c47a1c"   # orange

MAJOR_BASES = {
    'chariot','death','devil','empress','emperor','fool',
    'hanged man','heirophant','hermit','high priestess','judgement',
    'justice','lovers','magician','moon','star','strength','sun',
    'temperance','tower','wheel','world'
}
RANK_LABEL = {6:'Major Arcana', 5:'King', 4:'Queen', 3:'Knight', 2:'Page', 1:'pip'}

_STOP = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'by','from','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','not','no','nor','so','yet','both','either',
    'when','where','why','how','all','any','each','more','most','other',
    'some','such','than','too','very','just','that','this','these','those',
    'your','their','our','its','also','may','can','will','would','should',
    'could','about','what','into','them','they','you','him','her',
    'which','who','one','as','if','after','before','then','than','once',
    'even','still','often','never','always','much','many','itself',
    'own','same','back','over','out','off','up','down','there','here',
    'said','says','say','make','makes','made','take','takes','get','gets'
}


# ─────────────────────────────────────────────────────────────────────────────
# CSS — visual styling
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""
<style>
/* Card hero strip */
.card-strip {{
    display: flex;
    justify-content: space-around;
    align-items: flex-start;
    gap: 20px;
    margin: 12px 0 28px 0;
    padding: 20px;
    background: linear-gradient(180deg, #fafafa 0%, #f0f0f0 100%);
    border-radius: 12px;
    border: 1px solid #e0e0e0;
}}
.card-cell {{
    flex: 1;
    text-align: center;
    max-width: 240px;
}}
.card-cell img {{
    max-height: 320px;
    width: auto;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.18);
    border: 2px solid #fff;
}}
.card-cell.reversed img {{ transform: rotate(180deg); }}
.card-cell .card-role {{
    font-weight: 700;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}}
.card-cell .card-name {{
    font-size: 18px;
    font-weight: 600;
    margin-top: 10px;
}}
.card-cell .card-orient {{
    font-size: 12px;
    color: #666;
    font-style: italic;
}}
.card-cell.role-home .card-role {{ color: {COL_HOME}; }}
.card-cell.role-away .card-role {{ color: {COL_AWAY}; }}
.card-cell.role-mf   .card-role {{ color: {COL_MF}; }}

/* Section header */
.sec-header {{
    margin: 28px 0 12px 0;
    padding: 10px 14px;
    border-left: 5px solid #888;
    background: #f7f7f7;
    border-radius: 4px;
    font-size: 18px;
    font-weight: 600;
}}
.sec-header.role-home {{ border-left-color: {COL_HOME}; background: #f0f7f3; }}
.sec-header.role-away {{ border-left-color: {COL_AWAY}; background: #eef4fa; }}
.sec-header.role-mf   {{ border-left-color: {COL_MF};   background: #faf4eb; }}
.sec-header.role-pair {{ border-left-color: #6a4caf;    background: #f3eff8; }}

/* Three-column meaning panels */
.meaning-panel {{
    padding: 14px 16px;
    border-radius: 8px;
    border-top: 4px solid #888;
    background: #fcfcfc;
    height: 100%;
    font-size: 13px;
    line-height: 1.5;
}}
.meaning-panel.role-home {{ border-top-color: {COL_HOME}; }}
.meaning-panel.role-away {{ border-top-color: {COL_AWAY}; }}
.meaning-panel.role-mf   {{ border-top-color: {COL_MF}; }}
.meaning-panel .panel-title {{
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}
.meaning-panel.role-home .panel-title {{ color: {COL_HOME}; }}
.meaning-panel.role-away .panel-title {{ color: {COL_AWAY}; }}
.meaning-panel.role-mf   .panel-title {{ color: {COL_MF}; }}
.meaning-panel .card-label {{
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 6px;
    color: #222;
}}
.meaning-panel .orient-tag {{
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    background: #e0e0e0;
    color: #444;
    margin-left: 8px;
    vertical-align: middle;
}}
.meaning-panel .orient-tag.rx {{ background: #f5d8b8; color: #6a3a00; }}

/* Lookup section */
.lookup-block {{
    padding: 12px 16px;
    margin: 8px 0 16px 0;
    border-radius: 6px;
    background: #fcfcfc;
    border: 1px solid #ececec;
    font-size: 13px;
    line-height: 1.5;
}}
.lookup-block .lk-label {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    color: #888;
    letter-spacing: 0.6px;
    margin-bottom: 4px;
}}
.lookup-block .lk-text {{ color: #333; }}

/* Match table tweaks */
.match-table-wrap {{ margin: 6px 0 22px 0; }}
.match-table-wrap .table-caption {{
    font-size: 13px;
    color: #555;
    margin-bottom: 4px;
    font-weight: 500;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    s1 = pd.read_excel(DATA_FILE, sheet_name=0)
    md = pd.read_excel(DATA_FILE, sheet_name=1)
    s1.columns = s1.columns.str.strip()
    return s1, md

sheet1, meanings_df = load_data()


@st.cache_data
def build_meaning_map(_df):
    """
    Build {card_name: upright_meaning} from the meanings sheet.
    Used for API prompts and the existing logic.
    """
    mm = {}
    for _, row in _df.iterrows():
        card    = str(row.iloc[0]).strip()
        meaning = str(row.iloc[1]).strip()
        if card and card != 'nan' and meaning and meaning != 'nan':
            mm[card] = meaning
    return mm


@st.cache_data
def build_full_meaning_map(_df):
    """
    Build {card_name_lower: (upright_text, reversed_text)} for full lookup.
    Handles: 'AceW' → upright, 'AceWrx' → reversed, 'Sun' → upright, 'Sun rx' → reversed.
    """
    full = {}
    for _, row in _df.iterrows():
        card = str(row.iloc[0]).strip()
        if not card or card.lower() == 'nan':
            continue
        up = str(row.iloc[1]).strip() if not pd.isna(row.iloc[1]) else ''
        rv = str(row.iloc[2]).strip() if (len(row) > 2 and not pd.isna(row.iloc[2])) else ''
        if up.lower() == 'nan': up = ''
        if rv.lower() == 'nan': rv = ''
        full[card.lower()] = (up, rv)
    return full


meaning_map      = build_meaning_map(meanings_df)
full_meaning_map = build_full_meaning_map(meanings_df)
_l2c             = {k.lower(): k for k in meaning_map}


def canonical(x):
    return _l2c.get(str(x).strip().lower(), str(x).strip())

cards = sorted(meaning_map.keys())


# ─────────────────────────────────────────────────────────────────────────────
# RX NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
def base_card(card):
    c = card.strip()
    if c.lower().endswith(' rx'):
        return c[:-3].strip()
    if c.lower().endswith('rx'):
        return c[:-2].strip()
    return c


def is_reversed(card):
    return card.strip().lower().endswith('rx')


def card_variants(card):
    variants = {card}
    base = base_card(card)
    if base != card:
        variants.add(base)
    else:
        for suffix in (' rx', 'rx'):
            cand = card + suffix
            if cand in meaning_map:
                variants.add(cand)
    return variants


def all_orientation_strings(card):
    """
    Return every reasonable string form of the card (upright + reversed forms),
    used to filter historical match rows.
    """
    base = base_card(card)
    out = {base.lower()}
    out.add((base + 'rx').lower())
    out.add((base + ' rx').lower())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEMENT LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
def get_complement(card):
    if card.endswith(' rx'):
        base = card[:-3]
        return base if base in meaning_map else None
    if card.endswith('rx'):
        base = card[:-2]
        return base if base in meaning_map else None
    for suffix in (' rx', 'rx'):
        cand = card + suffix
        if cand in meaning_map:
            return cand
    return None


# ─────────────────────────────────────────────────────────────────────────────
# WIKIMEDIA CARD IMAGES — Rider-Waite from Commons
# ─────────────────────────────────────────────────────────────────────────────
_MAJOR_FILES = {
    'fool':           'RWS_Tarot_00_Fool.jpg',
    'magician':       'RWS_Tarot_01_Magician.jpg',
    'high priestess': 'RWS_Tarot_02_High_Priestess.jpg',
    'empress':        'RWS_Tarot_03_Empress.jpg',
    'emperor':        'RWS_Tarot_04_Emperor.jpg',
    'heirophant':     'RWS_Tarot_05_Hierophant.jpg',
    'hierophant':     'RWS_Tarot_05_Hierophant.jpg',
    'lovers':         'RWS_Tarot_06_Lovers.jpg',
    'chariot':        'RWS_Tarot_07_Chariot.jpg',
    'strength':       'RWS_Tarot_08_Strength.jpg',
    'hermit':         'RWS_Tarot_09_Hermit.jpg',
    'wheel':          'RWS_Tarot_10_Wheel_of_Fortune.jpg',
    'justice':        'RWS_Tarot_11_Justice.jpg',
    'hanged man':     'RWS_Tarot_12_Hanged_Man.jpg',
    'death':          'RWS_Tarot_13_Death.jpg',
    'temperance':     'RWS_Tarot_14_Temperance.jpg',
    'devil':          'RWS_Tarot_15_Devil.jpg',
    'tower':          'RWS_Tarot_16_Tower.jpg',
    'star':           'RWS_Tarot_17_Star.jpg',
    'moon':           'RWS_Tarot_18_Moon.jpg',
    'sun':            'RWS_Tarot_19_Sun.jpg',
    'judgement':      'RWS_Tarot_20_Judgement.jpg',
    'world':          'RWS_Tarot_21_World.jpg',
}

_SUIT_FILE_PREFIX = {
    'C': 'Cups',     # Cups
    'P': 'Pents',    # Pentacles / Coins
    'S': 'Swords',   # Swords
    'W': 'Wands',    # Wands
}

# Within each suit: 01=Ace, 02-10=numeric, 11=Page, 12=Knight, 13=Queen, 14=King
_COURT_NUM = {'P': 11, 'KN': 12, 'Q': 13, 'K': 14}


def _minor_card_filename(base):
    """
    Decode a base card code (e.g. 'AceW', '7S', 'KnP', 'QC', 'KS', 'PC')
    into the Commons filename (Wands01.jpg etc).
    """
    s = base.strip()
    # Court cards: prefix is letter(s) before the suit letter
    # K, Q -> 1 letter; Kn, P -> page/knight (Kn 2-letter, P 1-letter)
    suit = s[-1].upper()
    if suit not in _SUIT_FILE_PREFIX:
        return None
    rank_part = s[:-1].strip()

    rank_lower = rank_part.lower()
    # Court detection
    if rank_lower in ('k',):       num = 14
    elif rank_lower in ('q',):     num = 13
    elif rank_lower in ('kn',):    num = 12
    elif rank_lower in ('p',):     num = 11
    elif rank_lower in ('ace', 'a', '1'): num = 1
    else:
        # numeric pip: '2', '3', ..., '10'
        try:
            num = int(rank_lower)
        except ValueError:
            return None
        if num < 1 or num > 10:
            return None

    return f"{_SUIT_FILE_PREFIX[suit]}{num:02d}.jpg"


def card_image_url(card, width=260):
    """
    Build a Commons Special:FilePath URL for the given card.
    Returns None if the card name doesn't decode.
    """
    if not card:
        return None
    base = base_card(card).strip()
    base_lower = base.lower()

    if base_lower in _MAJOR_FILES:
        fname = _MAJOR_FILES[base_lower]
    else:
        fname = _minor_card_filename(base)

    if not fname:
        return None

    encoded = urllib.parse.quote(fname)
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded}?width={width}"


# ─────────────────────────────────────────────────────────────────────────────
# MF LOOKUP — loads mf_lookup.xlsx (col B + col C per card)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_mf_lookup():
    if not os.path.exists(MF_LOOKUP_FILE):
        return None
    try:
        df = pd.read_excel(MF_LOOKUP_FILE, sheet_name='MF Lookup', usecols='A:C',
                           header=0, names=['Idx', 'Match Force', 'Winners'])
        return df
    except Exception:
        try:
            # Fallback: read raw and trim
            df = pd.read_excel(MF_LOOKUP_FILE, sheet_name='MF Lookup')
            df = df.iloc[:, :3]
            df.columns = ['Idx', 'Match Force', 'Winners']
            return df
        except Exception:
            return None


_mf_lookup_df = load_mf_lookup()


def _extract_lookup_card_name(cell_text):
    """Pull just the card name from the front of a 'Match Force' cell."""
    txt = str(cell_text).strip()
    for sep in [' - ', '  ', ' (']:
        if sep in txt:
            return txt.split(sep)[0].strip()
    return txt


def _split_lookup_cell_b(cell_text):
    """Return (card_name, descriptive_note) from a col B cell."""
    txt = str(cell_text).strip()
    name = _extract_lookup_card_name(txt)
    rest = txt[len(name):].lstrip(' -').strip()
    return name, rest


def get_mf_lookup_for_base(card):
    """
    Return the mf_lookup row for a card's BASE (upright) form.
    e.g. 'PCrx' or 'PC' → row whose col B starts with 'PC'.
    Returns None if not found.
    """
    if _mf_lookup_df is None:
        return None
    base_lower = base_card(card).lower()
    # Pass 1: exact upright match
    for _, row in _mf_lookup_df.iterrows():
        cell = str(row['Match Force']).strip()
        if not cell or cell.lower() == 'nan':
            continue
        name = _extract_lookup_card_name(cell)
        n_lower = name.lower()
        # only accept upright entries (skip rows that themselves end in rx)
        if n_lower.endswith('rx') or n_lower.endswith(' rx'):
            continue
        if n_lower == base_lower or n_lower.replace(' ', '') == base_lower.replace(' ', ''):
            return row
    # Pass 2: any row matching base (including rx rows) as fallback
    for _, row in _mf_lookup_df.iterrows():
        cell = str(row['Match Force']).strip()
        if not cell or cell.lower() == 'nan':
            continue
        name = _extract_lookup_card_name(cell)
        if base_card(name).lower() == base_lower:
            return row
    return None


# Backwards-compat alias used by analysis_engine
def get_mf_lookup_entry(mf_card):
    return get_mf_lookup_for_base(mf_card)


# ─────────────────────────────────────────────────────────────────────────────
# LITERAL MEANING ANALYSIS  (kept for fallback path)
# ─────────────────────────────────────────────────────────────────────────────
def parse_meaning_phrases(text):
    if not text or str(text).strip().lower() == 'nan':
        return []
    phrases = []
    for chunk in re.split(r'[,;\n]', str(text)):
        p = chunk.strip().lower()
        p = re.sub(r'\s+', ' ', p)
        p = re.sub(r'^[^a-z]+|[^a-z]+$', '', p).strip()
        if p and len(p) >= 3:
            phrases.append(p)
    return phrases


def _sig_words(phrase, min_len=4):
    return {w for w in re.findall(r'\b[a-z]+\b', phrase.lower())
            if len(w) >= min_len and w not in _STOP}


def find_word_overlaps(team_phrases, target_phrases):
    hits = []
    seen = set()
    for tp in team_phrases:
        tw = _sig_words(tp)
        if not tw: continue
        for xp in target_phrases:
            xw = _sig_words(xp)
            shared = tw & xw
            if shared:
                key = frozenset(shared)
                if key not in seen:
                    seen.add(key)
                    hits.append((tp, xp, sorted(shared)))
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# TFIDF FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_tfidf(_mm):
    cl  = list(_mm.keys())
    vz  = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=1, max_features=2000)
    mt  = vz.fit_transform([_mm[c] for c in cl])
    c2i = {c: i for i, c in enumerate(cl)}
    ft  = vz.get_feature_names_out()
    return mt, c2i, ft

_tmat, _c2i, _feats = build_tfidf(meaning_map)


def _tsim(a, b):
    ia, ib = _c2i.get(a), _c2i.get(b)
    if ia is None or ib is None: return 0.0
    return float(cosine_similarity(_tmat[ia], _tmat[ib])[0][0])


def _tkw(a, b, n=3):
    ia, ib = _c2i.get(a), _c2i.get(b)
    if ia is None or ib is None: return []
    prod = _tmat[ia].multiply(_tmat[ib]).toarray().flatten()
    idx  = prod.argsort()[-n:][::-1]
    return [_feats[i] for i in idx if prod[i] > 0]


def tfidf_resonance(card, mf):
    sim  = _tsim(card, mf) * 5
    comp = get_complement(mf)
    opp  = (_tsim(card, comp) * 5 if comp else 0.0)
    sim  = min(sim, 1.0); opp = min(opp, 1.0)
    if opp > sim:
        return opp, ('OPP' if opp >= BALANCE_MIN else None), (_tkw(card, comp) if comp else [])
    return sim, ('SIM' if sim >= BALANCE_MIN else None), _tkw(card, mf)


# ─────────────────────────────────────────────────────────────────────────────
# RESONANCE CACHE
# ─────────────────────────────────────────────────────────────────────────────
def _load_disk_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_disk_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


if 'rcache' not in st.session_state:
    st.session_state.rcache = _load_disk_cache()


# ─────────────────────────────────────────────────────────────────────────────
# API CHECK
# ─────────────────────────────────────────────────────────────────────────────
_API_READY = None
_API_ERROR = None


def check_api():
    global _API_READY, _API_ERROR
    if _API_READY is not None:
        return _API_READY
    try:
        import anthropic as _a
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            key = os.environ.get("ANTHROPIC_API_KEY", "")
        key = key.strip()
        if not key:
            _API_READY = False; _API_ERROR = "No API key found."; return False
        os.environ["ANTHROPIC_API_KEY"] = key
        client = _a.Anthropic(api_key=key)
        client.messages.create(model=API_MODEL, max_tokens=5,
                               messages=[{"role": "user", "content": "ping"}])
        _API_READY = True; _API_ERROR = None
    except ImportError:
        _API_READY = False; _API_ERROR = "anthropic package not installed."
    except Exception as e:
        _API_READY = False; _API_ERROR = str(e)
    return _API_READY


_PROMPT = """You are analysing two tarot cards for a homeostatic balancing prediction system.

The Match Force sets the energetic context of the match. A team card balances it in one of two ways:
- SIM (Similarity): the card's energy mirrors or echoes the Match Force — they share the same quality. Find this in the listed keywords.
- OPP (Opposition / Deficit): the card provides what the Match Force lacks — the equal and opposite resolving force. Look for keywords in the team card meaning that directly negate or oppose specific keywords in the Match Force meaning.
- NONE: no meaningful energetic relationship.

Match Force card: {mf}
Match Force meaning (exact keywords): {mf_meaning}

Complement of Match Force — {comp_card} (this is what the MF LACKS / its opposite state):
{comp_meaning}

Team card: {card}
Team card meaning (exact keywords): {card_meaning}

Respond ONLY with a JSON object, no other text or markdown:
{{"type": "SIM", "score": 0.75, "aspects": ["phrase from team meaning", "phrase from MF meaning it connects to", "connection note"]}}
"""


def api_resonance(card, mf):
    key    = f"{card}||{mf}"
    cached = st.session_state.rcache.get(key)
    if cached:
        d  = cached
        sc = float(d.get('score', 0.0))
        rt = d.get('type', 'NONE')
        return sc, (None if rt == 'NONE' or sc < BALANCE_MIN else rt), d.get('aspects', [])

    import anthropic
    client = anthropic.Anthropic()

    comp      = get_complement(mf)
    comp_mean = meaning_map.get(comp, 'none')[:300] if comp else 'none'

    prompt = _PROMPT.format(
        mf=mf,
        mf_meaning=meaning_map.get(mf, '')[:500],
        comp_card=comp or 'none',
        comp_meaning=comp_mean,
        card=card,
        card_meaning=meaning_map.get(card, '')[:500],
    )
    try:
        resp = client.messages.create(
            model=API_MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        raw  = resp.content[0].text.strip()
        raw  = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
        d    = json.loads(raw)
    except Exception as e:
        st.warning(f"API call failed for ({card}, {mf}): {e} — falling back to TF-IDF")
        return tfidf_resonance(card, mf)

    st.session_state.rcache[key] = d
    _save_disk_cache(st.session_state.rcache)

    sc = float(d.get('score', 0.0))
    rt = d.get('type', 'NONE')
    return sc, (None if rt == 'NONE' or sc < BALANCE_MIN else rt), d.get('aspects', [])


def resonance(card, mf):
    if check_api():
        return api_resonance(card, mf)
    return tfidf_resonance(card, mf)


# ─────────────────────────────────────────────────────────────────────────────
# CARD HIERARCHY
# ─────────────────────────────────────────────────────────────────────────────
def card_rank(card):
    c = card.lower().replace(' rx','').replace('rx','').strip()
    if c in MAJOR_BASES:                               return 6
    if c.startswith('k') and not c.startswith('kn'):   return 5
    if c.startswith('q'):                              return 4
    if c.startswith('kn'):                             return 3
    if c.startswith('p') and len(c) == 2:              return 2
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL MEMORY — keeps original orientation strings on each match
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def build_history(_s1, _mm):
    lc = {k.lower(): k for k in _mm}
    def cl(x): return lc.get(str(x).strip().lower(), str(x).strip())
    def nm(x):
        x = str(x).strip()
        if x in ('1', '1.0'): return 'Home'
        if x in ('2', '2.0'): return 'Away'
        if x.upper().startswith('X'): return 'Draw'
        return None

    record   = defaultdict(lambda: {'Win': 0, 'Loss': 0, 'Draw': 0, 'matches': []})
    exact    = defaultdict(lambda: {'Home': 0, 'Away': 0, 'Draw': 0, 'matches': []})
    no_match = defaultdict(lambda: {'Home': 0, 'Away': 0, 'Draw': 0, 'matches': []})

    all_rows = []  # full list with row indices for the per-card tables

    for idx, r in _s1.iterrows():
        outcome = nm(r.get('Outcome', ''))
        if not outcome: continue
        h_raw = r.get('Home', ''); a_raw = r.get('Away', ''); mf_raw = r.get('Match Force', '')
        h = cl(h_raw); a = cl(a_raw); mf = cl(mf_raw)
        bh = base_card(h); ba = base_card(a); bmf = base_card(mf)

        # Score for display
        score = r.get('Score', '')
        if pd.isna(score): score = ''
        # Try to read team names if present
        h_team = r.get('Home Team', '') if 'Home Team' in r.index else ''
        a_team = r.get('Away Team', '') if 'Away Team' in r.index else ''
        league = r.get('A280', '') if 'A280' in r.index else r.get('League', '')

        match_ctx = {
            'home': h, 'away': a, 'mf': mf, 'outcome': outcome,
            'row_index': int(idx) + 2,   # excel row number (header at row 1)
            'score': str(score) if score else '',
            'home_team': str(h_team) if h_team and not pd.isna(h_team) else '',
            'away_team': str(a_team) if a_team and not pd.isna(a_team) else '',
            'league': str(league) if league and not pd.isna(league) else '',
        }
        all_rows.append(match_ctx)

        exact[(bh, ba, bmf)][outcome] += 1
        exact[(bh, ba, bmf)]['matches'].append(match_ctx)

        if outcome == 'Home':
            record[(bh, bmf)]['Win'] += 1; record[(ba, bmf)]['Loss'] += 1
        elif outcome == 'Away':
            record[(bh, bmf)]['Loss'] += 1; record[(ba, bmf)]['Win'] += 1
        else:
            record[(bh, bmf)]['Draw'] += 1; record[(ba, bmf)]['Draw'] += 1

        record[(bh, bmf)]['matches'].append(match_ctx)
        record[(ba, bmf)]['matches'].append(match_ctx)

        if bmf != bh and bmf != ba:
            no_match[bmf][outcome] += 1
            no_match[bmf]['matches'].append(match_ctx)

    co_occur = defaultdict(list); mf_index = defaultdict(list)
    for m in all_rows:
        for c in {base_card(m['home']), base_card(m['away']), base_card(m['mf'])}:
            co_occur[c].append(m)
        mf_index[base_card(m['mf'])].append(m)

    def fr(d): return {k: dict(v) for k, v in d.items()}
    return {
        'record':   fr(record),
        'exact':    fr(exact),
        'no_match': fr(no_match),
        'co_occur': dict(co_occur),
        'mf_index': dict(mf_index),
        'all':      all_rows,
    }


_history = build_history(sheet1, meaning_map)


def get_mf_draw_rate(mf):
    bmf = base_card(mf)
    matches = _history.get('mf_index', {}).get(bmf, [])
    total = len(matches)
    if total < 5:
        return {'draws': sum(1 for m in matches if m['outcome']=='Draw'),
                'total': total, 'rate': None, 'category': 'insufficient'}
    draws = sum(1 for m in matches if m['outcome'] == 'Draw')
    rate  = draws / total
    if rate >= 0.35:   cat = 'high'
    elif rate <= 0.15: cat = 'low'
    else:              cat = 'normal'
    return {'draws': draws, 'total': total, 'rate': rate, 'category': cat}


# ─────────────────────────────────────────────────────────────────────────────
# MF-IDENTICAL PATTERN
# ─────────────────────────────────────────────────────────────────────────────
def mf_identical_check(home, away, mf):
    home_match = (home.strip().lower() == mf.strip().lower())
    away_match = (away.strip().lower() == mf.strip().lower())
    if not home_match and not away_match: return None, None, None

    if home_match: matching_team='Home'; matching_card=home; other_card=away
    else:          matching_team='Away'; matching_card=away; other_card=home

    other_score, other_type, other_aspects = resonance(other_card, mf)
    kws = ', '.join(f'*{k}*' for k in other_aspects[:3]) if other_aspects else 'thematic relationship'
    lines = [f"⚠️ **MF identical to {matching_team} card** ({matching_card}) — applying MF-identical pattern."]

    if other_type == 'OPP':
        other_team = 'Away' if matching_team=='Home' else 'Home'
        other_name = away if matching_team=='Home' else home
        lines.append(f"**{other_name}** ({other_team}) carries **opposite energy** to MF via {kws} [OPP · {other_score:.2f}].")
        lines.append(f"Identical-card rule: OPP → **{other_team} wins**")
        return other_team, round(min(0.70+other_score*0.18,0.88),2), lines
    elif other_type == 'SIM':
        lines.append(f"**{other_card}** shares energy with MF via {kws} [SIM · {other_score:.2f}] — MF reinforces matching team.")
        lines.append(f"Identical-card rule: SIM → **{matching_team} wins**")
        return matching_team, round(min(0.68+other_score*0.15,0.83),2), lines
    else:
        lines.append(f"**{other_card}** has no clear relationship with MF [NONE · {other_score:.2f}] — falling through.")
        return None, None, lines


# ─────────────────────────────────────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def decide(home, away, mf):
    id_pred, id_conf, id_lines = mf_identical_check(home, away, mf)
    if id_pred is not None: return id_pred, id_conf, id_lines

    hs, ht, hkw  = resonance(home, mf)
    as_, at, akw = resonance(away, mf)
    hr, ar       = card_rank(home), card_rank(away)
    lines        = []
    if id_lines: lines.extend(id_lines)

    def desc(card, score, rtype, kw):
        rs = RANK_LABEL[card_rank(card)]
        if rtype is None:
            return f"**{card}** ({rs}) — no meaningful resonance with Match Force (strength {score:.2f})"
        verb = "mirrors the energy of" if rtype=='SIM' else "provides what's missing in"
        kws  = ', '.join(f'*{k}*' for k in kw[:3]) if kw else 'thematic overlap'
        return f"**{card}** ({rs}) — {verb} **{mf}** via {kws}  [{rtype} · {score:.2f}]"

    lines.append(desc(home, hs, ht, hkw))
    lines.append(desc(away, as_, at, akw))

    if ht is None and at is None:
        lines.append("Neither card achieves homeostatic balance with the Match Force → **Draw**")
        return 'Draw', 0.55, lines
    if ht is not None and at is None:
        lines.append(f"Only **{home}** balances the Match Force → **Home wins**")
        return 'Home', min(0.70+hs*0.20,0.90), lines
    if at is not None and ht is None:
        lines.append(f"Only **{away}** balances the Match Force → **Away wins**")
        return 'Away', min(0.70+as_*0.20,0.90), lines

    if ht != at:
        if ht=='OPP':
            lines.append(f"**{home}** provides deficit energy (OPP), **{away}** only mirrors (SIM) → **Home wins**")
            return 'Home', min(0.68+abs(hs-as_)*0.3,0.88), lines
        else:
            lines.append(f"**{away}** provides deficit energy (OPP), **{home}** only mirrors (SIM) → **Away wins**")
            return 'Away', min(0.68+abs(as_-hs)*0.3,0.88), lines

    diff = abs(hs - as_)
    if diff >= ASPECTS_DELTA:
        if hs > as_:
            lines.append(f"Both {ht}; **{home}** engages more strongly ({hs:.2f} vs {as_:.2f}) → **Home wins**")
            return 'Home', min(0.62+diff*0.35,0.82), lines
        else:
            lines.append(f"Both {at}; **{away}** engages more strongly ({as_:.2f} vs {hs:.2f}) → **Away wins**")
            return 'Away', min(0.62+diff*0.35,0.82), lines

    if hr != ar:
        if hr > ar:
            lines.append(f"Equal balance; **{home}** ({RANK_LABEL[hr]}) outranks **{away}** ({RANK_LABEL[ar]}) → **Home wins**")
            return 'Home', min(0.58+(hr-ar)*0.04,0.74), lines
        else:
            lines.append(f"Equal balance; **{away}** ({RANK_LABEL[ar]}) outranks **{home}** ({RANK_LABEL[hr]}) → **Away wins**")
            return 'Away', min(0.58+(ar-hr)*0.04,0.74), lines

    lines.append(f"Both cards provide equivalent **{ht}** balance at equal authority ({RANK_LABEL[hr]}) → **Draw**")
    return 'Draw', 0.60, lines


def predict(home, away, mf):
    if check_api():
        result = analyse_match_full(
            home, away, mf,
            meaning_map     = meaning_map,
            get_complement_fn = get_complement,
            history         = _history,
            mf_lookup_fn    = get_mf_lookup_entry,
            base_card_fn    = base_card,
            rcache          = st.session_state.rcache,
            save_cache_fn   = _save_disk_cache,
            api_model       = API_MODEL,
        )
        if result and '_error' not in result and 'call' in result:
            pred, conf = analysis_to_prediction(result)
            return pred, round(min(conf, 0.95), 2), [], None, 0.0, [], result

    pred, conf, lines = decide(home, away, mf)
    return pred, round(min(conf, 0.95), 2), lines, None, 0.0, [], None


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: HERO STRIP (three card images at top)
# ─────────────────────────────────────────────────────────────────────────────
def render_hero_strip(home, away, mf):
    cells = []
    for role, role_label, card in [
        ('home', 'Home',        home),
        ('mf',   'Match Force', mf),
        ('away', 'Away',        away),
    ]:
        url = card_image_url(card, width=320)
        rev = is_reversed(card)
        rev_class = 'reversed' if rev else ''
        orient_html = '<div class="card-orient">reversed</div>' if rev else ''
        if url:
            img_html = f'<img src="{url}" alt="{card}">'
        else:
            img_html = (f'<div style="height:280px;display:flex;align-items:center;'
                        f'justify-content:center;color:#999;border:1px dashed #ccc;'
                        f'border-radius:8px;">no image</div>')
        cells.append(f"""
          <div class="card-cell role-{role} {rev_class}">
            <div class="card-role">{role_label}</div>
            {img_html}
            <div class="card-name">{card}</div>
            {orient_html}
          </div>
        """)
    st.markdown(f'<div class="card-strip">{"".join(cells)}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: THREE-COLUMN MEANINGS COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def get_meaning_for_orientation(card):
    """
    Return the meaning text appropriate to the card's orientation.
    Handles both meanings-sheet layouts:
      - separate rows per orientation (AceW row + AceWrx row, each with text in col B)
      - consolidated row (AceW row only, upright in col B, reversed in col C)
    'AceW' → upright; 'AceWrx' / 'Sun rx' → reversed.
    """
    card_l = card.strip().lower()
    rev    = is_reversed(card)

    # 1) Direct exact-card lookup — handles separate-row layouts where the
    #    reversed card has its own row with text in col B.
    direct = full_meaning_map.get(card_l)
    if direct and direct[0]:
        return direct[0]

    # 2) Fall back to base card + col C — handles consolidated-row layout
    #    where one row holds both orientations.
    base = base_card(card)
    base_entry = full_meaning_map.get(base.lower())
    if base_entry is None:
        return None
    up_t, rv_t = base_entry
    if rev:
        return rv_t or up_t
    return up_t or rv_t


def render_meanings_three_column(home, away, mf):
    st.markdown('<div class="sec-header">📖 Meanings — Home · Away · Match Force</div>',
                unsafe_allow_html=True)

    cols = st.columns(3)
    for col, (role, role_label, card) in zip(cols, [
        ('home', 'Home',        home),
        ('away', 'Away',        away),
        ('mf',   'Match Force', mf),
    ]):
        with col:
            text = get_meaning_for_orientation(card) or '_(no meaning text found)_'
            orient_label = 'reversed' if is_reversed(card) else 'upright'
            tag_cls = 'rx' if is_reversed(card) else ''
            html = f"""
              <div class="meaning-panel role-{role}">
                <div class="panel-title">{role_label}</div>
                <div class="card-label">{card}
                  <span class="orient-tag {tag_cls}">{orient_label}</span>
                </div>
                <div>{text}</div>
              </div>
            """
            st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: PER-CARD MF LOOKUP NOTES (col B + col C)
# ─────────────────────────────────────────────────────────────────────────────
def render_mf_lookup_for_card(role, role_label, card):
    """
    Display the user's notes from mf_lookup.xlsx for a single card,
    pulling both col B (descriptive note after card name) and col C (Winners notes).
    Always uses the BASE card (PCrx → PC).
    """
    st.markdown(
        f'<div class="sec-header role-{role}">📋 {role_label} Notes — '
        f'<code>{card}</code> (from <code>mf_lookup.xlsx</code> for base <code>{base_card(card)}</code>)</div>',
        unsafe_allow_html=True
    )

    if _mf_lookup_df is None:
        st.caption(f"`{MF_LOOKUP_FILE}` not loaded.")
        return

    row = get_mf_lookup_for_base(card)
    if row is None:
        st.caption(f"No entry found in `{MF_LOOKUP_FILE}` for **{base_card(card)}**.")
        return

    cell_b  = str(row['Match Force']).strip()
    cell_c  = str(row['Winners']).strip() if 'Winners' in row.index else ''

    name, desc = _split_lookup_cell_b(cell_b)

    # Column B — the descriptive note (anything after the card name)
    if desc and desc.lower() != 'nan':
        st.markdown(
            f'<div class="lookup-block">'
            f'<div class="lk-label">Column B — context note</div>'
            f'<div class="lk-text">{desc}</div>'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="lookup-block">'
            f'<div class="lk-label">Column B — context note</div>'
            f'<div class="lk-text" style="color:#999;font-style:italic;">'
            f'No descriptive note recorded yet for {name}.</div>'
            f'</div>', unsafe_allow_html=True)

    # Column C — the contrasts/harmonies / themes
    if cell_c and cell_c.lower() != 'nan':
        st.markdown(
            f'<div class="lookup-block">'
            f'<div class="lk-label">Column C — contrasts &amp; harmonies</div>'
            f'<div class="lk-text">{cell_c}</div>'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="lookup-block">'
            f'<div class="lk-label">Column C — contrasts &amp; harmonies</div>'
            f'<div class="lk-text" style="color:#999;font-style:italic;">No notes recorded.</div>'
            f'</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: HISTORICAL MATCH TABLES (per card, both orientations)
# ─────────────────────────────────────────────────────────────────────────────
_OUT_GLYPH = {'Home': '1', 'Away': '2', 'Draw': 'X'}


def _matches_to_df(matches):
    """Convert match dicts → display DataFrame."""
    if not matches:
        return None
    rows = []
    for m in matches:
        rows.append({
            'Row':     m.get('row_index', ''),
            'Out':     _OUT_GLYPH.get(m.get('outcome',''), m.get('outcome','')),
            'League':  (m.get('league') or '')[:24],
            'Home Team': (m.get('home_team') or '')[:22],
            'Away Team': (m.get('away_team') or '')[:22],
            'H':       m['home'],
            'A':       m['away'],
            'MF':      m['mf'],
            'Score':   m.get('score',''),
        })
    df = pd.DataFrame(rows)
    return df.sort_values('Row', ascending=True).reset_index(drop=True)


def matches_where_card_is_mf(card_variant):
    """All matches where the EXACT card variant (e.g. 'AceW' OR 'AceWrx') was the MF."""
    target = card_variant.strip().lower()
    base = base_card(card_variant)
    pool = _history['mf_index'].get(base, [])
    return [m for m in pool if m['mf'].strip().lower() == target]


def matches_where_card_is_home(card_variant):
    target = card_variant.strip().lower()
    base = base_card(card_variant)
    pool = _history['co_occur'].get(base, [])
    return [m for m in pool if m['home'].strip().lower() == target]


def matches_where_card_is_away(card_variant):
    target = card_variant.strip().lower()
    base = base_card(card_variant)
    pool = _history['co_occur'].get(base, [])
    return [m for m in pool if m['away'].strip().lower() == target]


def render_card_history(role, role_label, card):
    """For one card, show all six tables (upright + reversed × MF/Home/Away)."""
    base = base_card(card)
    upright = base
    reversed_strs = [base + 'rx']
    if base + ' rx' not in reversed_strs and base.lower() in MAJOR_BASES:
        reversed_strs = [base + ' rx']

    st.markdown(
        f'<div class="sec-header role-{role}">🃏 {role_label} — historical matches for '
        f'<code>{card}</code> (base: <code>{base}</code>)</div>',
        unsafe_allow_html=True
    )

    blocks = [
        (f"{upright} as MATCH FORCE",   matches_where_card_is_mf(upright)),
        (f"{upright} as HOME card",     matches_where_card_is_home(upright)),
        (f"{upright} as AWAY card",     matches_where_card_is_away(upright)),
    ]
    for rev in reversed_strs:
        blocks += [
            (f"{rev} as MATCH FORCE",   matches_where_card_is_mf(rev)),
            (f"{rev} as HOME card",     matches_where_card_is_home(rev)),
            (f"{rev} as AWAY card",     matches_where_card_is_away(rev)),
        ]

    for label, matches in blocks:
        df = _matches_to_df(matches)
        st.markdown(
            f'<div class="match-table-wrap"><div class="table-caption">{label} — '
            f'{len(matches)} match{"es" if len(matches)!=1 else ""}</div></div>',
            unsafe_allow_html=True
        )
        if df is not None and not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("_no matches_")


def render_pair_history(home, away):
    """All matches where home and away appeared together as the two team cards (any orientation)."""
    bh = base_card(home); ba = base_card(away)
    home_strs = all_orientation_strings(home)
    away_strs = all_orientation_strings(away)

    matches = []
    for m in _history['co_occur'].get(bh, []):
        h_l = m['home'].strip().lower(); a_l = m['away'].strip().lower()
        if (h_l in home_strs and a_l in away_strs) or (h_l in away_strs and a_l in home_strs):
            matches.append(m)

    st.markdown(
        f'<div class="sec-header role-pair">🤝 Pair history — '
        f'<code>{home}</code> + <code>{away}</code> together as team cards (any orientation)</div>',
        unsafe_allow_html=True
    )
    df = _matches_to_df(matches)
    if df is not None and not df.empty:
        st.caption(f"{len(matches)} match{'es' if len(matches)!=1 else ''} found")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("_These two cards have not appeared together as team cards before._")


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: AI HOLISTIC ANALYSIS  (kept from V10)
# ─────────────────────────────────────────────────────────────────────────────
def render_full_analysis(result, home, away, mf):
    icons_type = {'OPP': '🟢', 'SIM': '🔵', 'NONE': '⚫'}

    st.markdown('<div class="sec-header">🧠 AI Reasoning</div>', unsafe_allow_html=True)

    st.markdown(f"**⚡ {mf}** — {result.get('mf_quality','')}")
    phrases = result.get('mf_key_phrases', [])
    if phrases:
        st.caption('  ·  '.join(f'*{p}*' for p in phrases))

    col1, col2 = st.columns(2)
    for col, card, sense_key, type_key, phrases_key, tied_key, note_key in [
        (col1, home, 'home_sense', 'home_type', 'home_phrases', 'home_tied_words', 'home_note'),
        (col2, away, 'away_sense', 'away_type', 'away_phrases', 'away_tied_words', 'away_note'),
    ]:
        with col:
            ctype  = result.get(type_key, 'NONE')
            icon   = icons_type.get(ctype, '⚫')
            label  = '🏠' if card == home else '✈'
            dm     = result.get(type_key.replace('type','domain_match'), False)
            dm_str = '' if dm else ' *(different domain)*'
            st.markdown(f"**{label} {card}**")
            st.markdown(f"*{result.get(sense_key, '')}*")
            st.markdown(f"{icon} **{ctype}**{dm_str}")
            for p in result.get(phrases_key, []):
                st.markdown(f"- *\"{p}\"*")
            tied = result.get(tied_key, [])
            if tied:
                st.caption(f"Tied with other card (neutral): {', '.join(tied)}")
            note = result.get(note_key, '')
            if note: st.caption(note)

    st.markdown("**📊 History summary**")
    for key, label in [
        ('history_mf_pattern', 'MF pattern'),
        ('history_direct',     'Direct'),
        ('history_co',         'Together'),
    ]:
        val = result.get(key, '')
        if val and val.lower() not in ('none', 'none found', ''):
            st.markdown(f"- **{label}:** {val}")

    hc = result.get('history_confirms', 'None')
    call = result.get('call', 'Draw')
    if hc and hc not in ('None', ''):
        if hc == call:
            st.markdown(f"- ✅ History agrees: **{hc}**")
        elif hc == 'Conflicts':
            st.markdown(f"- ⚠️ History conflicts with meaning analysis")
        else:
            st.markdown(f"- ➖ History neutral / insufficient data")

    draw_sig = result.get('draw_signal', '')
    if draw_sig and draw_sig.lower() not in ('no draw signal', 'none', ''):
        st.markdown(f"- 🎲 **Draw prior:** {draw_sig}")

    call_icons = {'Home': '🟢', 'Away': '🔵', 'Draw': '🟡'}
    ci = call_icons.get(call, '⚪')
    st.markdown(f"### 🎯 {ci} Call: **{call}**")
    reason = result.get('reason', '')
    if reason: st.markdown(f"*{reason}*")


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
inject_css()

st.title("🔮 Tarot Predictor V11")
st.caption(
    "Winner is the team whose card best balances the Match Force — "
    "through similarity (mirroring) or opposition (providing the deficit)."
)

api_ok     = check_api()
cache_size = len(st.session_state.rcache)
if api_ok:
    st.success(f"✅ Semantic API active — {cache_size} resonance pair{'s' if cache_size != 1 else ''} cached to disk")
else:
    err = f": {_API_ERROR}" if _API_ERROR else ""
    st.warning(f"⚠ Semantic API unavailable{err} — using TF-IDF keyword fallback.")

if _mf_lookup_df is not None:
    st.info(f"📋 MF Lookup loaded — {len(_mf_lookup_df)} entries from {MF_LOOKUP_FILE}")

st.divider()

# Tab-key navigation between the three selectboxes
st.components.v1.html("""
<script>
(function() {
  function hookTab() {
    var boxes = Array.from(
      window.parent.document.querySelectorAll('div[data-testid="stSelectbox"] input')
    ).slice(0, 3);
    if (boxes.length < 3) { setTimeout(hookTab, 300); return; }
    boxes.forEach(function(box, idx) {
      box.addEventListener("keydown", function(e) {
        if (e.key !== "Tab") return;
        e.preventDefault();
        boxes[(e.shiftKey ? idx-1+boxes.length : idx+1) % boxes.length].focus();
      });
    });
  }
  hookTab();
})();
</script>
""", height=0)

c1, c2, c3 = st.columns(3)
with c1: home = st.selectbox("🏠 Home Card", cards)
with c2: away = st.selectbox("✈ Away Card", cards)
with c3: mf   = st.selectbox("⚡ Match Force", cards)

# MF draw-rate prior indicator
_mfd = get_mf_draw_rate(mf)
if _mfd['category'] == 'high':
    st.warning(f"⚠ **{mf}** historically draws in **{int(_mfd['rate']*100)}%** of matches "
               f"({_mfd['draws']}/{_mfd['total']}) — static/stuck MF, draws common")
elif _mfd['category'] == 'low':
    st.success(f"✓ **{mf}** historically draws in only **{int(_mfd['rate']*100)}%** of matches "
               f"({_mfd['draws']}/{_mfd['total']}) — decisive MF, draws rare")
elif _mfd['category'] == 'normal':
    st.caption(f"📊 **{mf}** historical draw rate: {int(_mfd['rate']*100)}% ({_mfd['draws']}/{_mfd['total']}) — normal range")
else:
    st.caption(f"📊 **{mf}** has only {_mfd['total']} past appearances — insufficient history for draw prior")


if st.button("Predict", type="primary", use_container_width=True):
    with st.spinner("Reading cards and history…"):
        pred, conf, explanation, hist_pred, hist_weight, hist_notes, full_analysis = predict(home, away, mf)

    # ── HERO STRIP — three card images ──────────────────────────────────────
    render_hero_strip(home, away, mf)

    # ── Prediction header ────────────────────────────────────────────────────
    icons = {'Home':'🟢', 'Away':'🔵', 'Draw':'🟡'}
    col1, col2 = st.columns([3, 1])
    with col1: st.markdown(f"## {icons.get(pred,'⚪')} Prediction: **{pred}**")
    with col2: st.metric("Confidence", f"{conf:.0%}")
    st.progress(conf)

    if conf >= 0.80:   st.caption("High confidence.")
    elif conf >= 0.65: st.caption("Good confidence.")
    elif conf >= 0.55: st.caption("Moderate confidence — exercise judgement.")
    else:              st.caption("Lower confidence — use as a pointer only.")

    # ── 3-COLUMN MEANINGS ────────────────────────────────────────────────────
    render_meanings_three_column(home, away, mf)

    # ── AI ANALYSIS ──────────────────────────────────────────────────────────
    if full_analysis and '_error' not in full_analysis:
        render_full_analysis(full_analysis, home, away, mf)
    elif explanation:
        st.markdown('<div class="sec-header">Reasoning (TF-IDF fallback)</div>',
                    unsafe_allow_html=True)
        for i, line in enumerate(explanation):
            st.markdown(f"{'→' if i < 2 else '⟹'} {line}")
        if full_analysis and '_error' in full_analysis:
            st.warning(f"Holistic analysis failed: {full_analysis['_error']}")

    # ── PER-CARD MF LOOKUP (col B + C) ───────────────────────────────────────
    render_mf_lookup_for_card('home', 'Home',        home)
    render_mf_lookup_for_card('away', 'Away',        away)
    render_mf_lookup_for_card('mf',   'Match Force', mf)

    # ── HISTORICAL TABLES PER CARD ───────────────────────────────────────────
    render_card_history('home', 'Home',        home)
    render_card_history('away', 'Away',        away)
    render_card_history('mf',   'Match Force', mf)

    # ── PAIR HISTORY ─────────────────────────────────────────────────────────
    render_pair_history(home, away)

    # ── Cache stats footer ───────────────────────────────────────────────────
    analysis_count = sum(1 for k in st.session_state.rcache if k.startswith(ANALYSIS_PREFIX))
    pair_count     = sum(1 for k in st.session_state.rcache if not k.startswith(ANALYSIS_PREFIX))
    st.caption(f"Cache: {analysis_count} match analyses · {pair_count} pair resonances stored.")


st.divider()
with st.expander("Decision rules (in priority order)"):
    st.markdown("""
**Rule 0 — MF-identical pattern (highest priority):**
When the MF = a team card: compare MF with the OTHER team's card.
- OPP → other team wins | SIM → matching team wins | NONE → fall through

**Resonance rules:**
1. Only one card balances MF → that team wins
2. Both balance, different types → OPP beats SIM (deficit energy is stronger)
3. Both balance, same type, score gap ≥ 0.15 → higher score wins
4. Both balance, same type, similar scores → card hierarchy (Major > King > Queen > Knight > Page > pip)
5. Equal everything → Draw  |  6. Neither balances → Draw

**Per-card data shown after each prediction:**
- 3-column meanings (full text from meanings sheet)
- MF Lookup notes — col B (context) + col C (contrasts & harmonies) for each card, looked up by base card name
- Historical matches — for each card, all matches where it was MF / Home / Away, in both upright AND reversed orientations
- Pair history — every previous match where the home + away pair appeared together
""")
