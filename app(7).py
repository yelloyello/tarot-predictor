"""
Tarot Predictor V11.2 (INTERIM)
Methodology: a team CONTROLS the Match Force via mirror (SIM) or destabiliser (OPP).
Rules engine: v6 interim (analysis_engine.py).

What's new in V11.2 (interim):
- Hierarchy logic removed entirely (no rank weighting, no pip-pip exception,
  no hierarchy exception).
- SIM/OPP precedence flipped: the mirror (SIM) now wins over the destabiliser
  (OPP). The MF prefers an aligning partner first; opposition is the fallback.
- Contested-trait check removed from the rule layer and from the UI. It was
  firing too often and was the main cause of the "everything is a draw"
  behaviour in V11.1.
- Prompt updated: removed rank/pip/hierarchy/contested mentions, flipped the
  tie-break advice to favour SIM, added explicit guidance on disambiguating
  reversed cards with dual/opposing meanings using the other cards in play.
- Duplicate-card rules preserved unchanged.

Carry-over from V11.1:
- Three-card hero strip at top (Wikimedia Rider-Waite images, rx rotated 180°)
- Three-column meanings comparison (Home | Away | MF) with full text
- Per-card MF Lookup sections showing column B + C from mf_lookup.xlsx
- Custom CSS styling: colour-coded sections per card role
- Historical match blocks rendered as HTML tables
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

st.set_page_config(page_title="Tarot Predictor V11.2", layout="wide")

CACHE_FILE      = "resonance_cache.json"
DATA_FILE       = "data.xlsx"
MF_LOOKUP_FILE  = "mf_lookup.xlsx"
API_MODEL = "claude-sonnet-4-6"

# Resolve files relative to the script directory so deployment paths work
_HERE = os.path.dirname(os.path.abspath(__file__))

def _resolve(filename):
    """Try multiple plausible locations for a data file."""
    candidates = [
        filename,
        os.path.join(_HERE, filename),
        os.path.join(os.getcwd(), filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return filename  # return original even if missing, so error messages show it

os.environ.setdefault("ANTHROPIC_API_KEY", "")

BALANCE_MIN   = 0.35
ASPECTS_DELTA = 0.15

COL_HOME = "#2e8b57"
COL_AWAY = "#1e6fb8"
COL_MF   = "#c47a1c"

MAJOR_BASES = {
    'chariot','death','devil','empress','emperor','fool',
    'hanged man','heirophant','hermit','high priestess','judgement',
    'justice','lovers','magician','moon','star','strength','sun',
    'temperance','tower','wheel','world'
}

_STOP = set()


def inject_css():
    st.markdown(f"""
<style>
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

.meaning-panel {{
    padding: 14px 16px;
    border-radius: 8px;
    border-top: 4px solid #888;
    background: #fcfcfc;
    height: 100%;
    font-size: 15px;
    line-height: 1.55;
}}
.meaning-panel ul {{
    margin: 8px 0 0 0;
    padding-left: 22px;
}}
.meaning-panel ul li {{
    margin-bottom: 7px;
}}
.meaning-panel.role-home {{ border-top-color: {COL_HOME}; }}
.meaning-panel.role-away {{ border-top-color: {COL_AWAY}; }}
.meaning-panel.role-mf   {{ border-top-color: {COL_MF}; }}
.meaning-panel .panel-title {{
    font-weight: 700; font-size: 14px; margin-bottom: 8px;
    text-transform: uppercase; letter-spacing: 0.8px;
}}
.meaning-panel.role-home .panel-title {{ color: {COL_HOME}; }}
.meaning-panel.role-away .panel-title {{ color: {COL_AWAY}; }}
.meaning-panel.role-mf   .panel-title {{ color: {COL_MF}; }}
.meaning-panel .card-label {{
    font-weight: 600; font-size: 16px; margin-bottom: 6px; color: #222;
}}
.meaning-panel .orient-tag {{
    display: inline-block; font-size: 11px; padding: 2px 8px;
    border-radius: 10px; background: #e0e0e0; color: #444;
    margin-left: 8px; vertical-align: middle;
}}
.meaning-panel .orient-tag.rx {{ background: #f5d8b8; color: #6a3a00; }}

.lookup-block {{
    padding: 12px 16px; margin: 8px 0 16px 0;
    border-radius: 6px; background: #fcfcfc;
    border: 1px solid #ececec; font-size: 13px; line-height: 1.5;
}}
.lookup-block .lk-label {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    color: #888; letter-spacing: 0.6px; margin-bottom: 4px;
}}
.lookup-block .lk-text {{ color: #333; }}

.match-table-wrap {{ margin: 6px 0 22px 0; }}
.match-table-wrap .table-caption {{
    font-size: 13px; color: #555; margin-bottom: 4px; font-weight: 500;
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    path = _resolve(DATA_FILE)
    s1 = pd.read_excel(path, sheet_name=0)
    md = pd.read_excel(path, sheet_name=1)
    s1.columns = s1.columns.str.strip()
    return s1, md

sheet1, meanings_df = load_data()


@st.cache_data
def build_meaning_map(_df):
    mm = {}
    for _, row in _df.iterrows():
        card    = str(row.iloc[0]).strip()
        meaning = str(row.iloc[1]).strip()
        if card and card != 'nan' and meaning and meaning != 'nan':
            mm[card] = meaning
    return mm


@st.cache_data
def build_full_meaning_map(_df):
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

cards = sorted(meaning_map.keys())


def base_card(card):
    c = card.strip()
    if c.lower().endswith(' rx'): return c[:-3].strip()
    if c.lower().endswith('rx'):  return c[:-2].strip()
    return c


def is_reversed(card):
    return card.strip().lower().endswith('rx')


def all_orientation_strings(card):
    """Every textual form a card might appear as in the data."""
    base = base_card(card)
    out = {base.lower(), (base + 'rx').lower(), (base + ' rx').lower()}
    return out


def get_complement(card):
    if card.endswith(' rx'):
        b = card[:-3]
        return b if b in meaning_map else None
    if card.endswith('rx'):
        b = card[:-2]
        return b if b in meaning_map else None
    for suffix in (' rx', 'rx'):
        cand = card + suffix
        if cand in meaning_map: return cand
    return None


# ─── Wikimedia card images ──────────────────────────────────────────────
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
_SUIT_FILE_PREFIX = {'C':'Cups','P':'Pents','S':'Swords','W':'Wands'}


def _minor_card_filename(base):
    s = base.strip()
    if not s: return None
    suit = s[-1].upper()
    if suit not in _SUIT_FILE_PREFIX: return None
    rank_lower = s[:-1].strip().lower()
    if rank_lower == 'k':         num = 14
    elif rank_lower == 'q':       num = 13
    elif rank_lower == 'kn':      num = 12
    elif rank_lower == 'p':       num = 11
    elif rank_lower in ('ace','a','1'): num = 1
    else:
        try: num = int(rank_lower)
        except ValueError: return None
        if num < 1 or num > 10: return None
    return f"{_SUIT_FILE_PREFIX[suit]}{num:02d}.jpg"


def card_image_url(card, width=320):
    if not card: return None
    base = base_card(card).strip()
    base_lower = base.lower()
    fname = _MAJOR_FILES.get(base_lower) or _minor_card_filename(base)
    if not fname: return None
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(fname)}?width={width}"


# ─── MF Lookup ─────────────────────────────────────────────────────────
_mf_lookup_load_error = None  # captured for UI display

@st.cache_data
def load_mf_lookup():
    """
    The mf_lookup.xlsx has 4 cols: index, Match Force (card name), Winners, Description.
    Header is on row 1; data starts on row 2.
    """
    path = _resolve(MF_LOOKUP_FILE)
    if not os.path.exists(path):
        return {'df': None,
                'error': f"File not found at {path} (also tried {_HERE} and CWD)",
                'path': None}
    try:
        xl = pd.ExcelFile(path)
        sheets = xl.sheet_names
        chosen = None
        for s in sheets:
            if s == 'MF Lookup':
                chosen = s; break
        if chosen is None:
            for s in sheets:
                if 'lookup' in s.lower():
                    chosen = s; break
        if chosen is None and sheets:
            chosen = sheets[0]
        if chosen is None:
            return {'df': None,
                    'error': f"No sheets found in {path}",
                    'path': path}

        df = pd.read_excel(path, sheet_name=chosen)
        cols = list(df.columns)
        if len(cols) >= 4:
            df = df.iloc[:, [1, 2, 3]].copy()
            df.columns = ['card_cell', 'winners', 'description']
        elif len(cols) == 3:
            df = df.iloc[:, [0, 1, 2]].copy()
            df.columns = ['card_cell', 'winners', 'description']
        else:
            return {'df': None,
                    'error': f"Sheet '{chosen}' has only {len(cols)} columns; expected 3 or 4",
                    'path': path}
        return {'df': df, 'error': None, 'path': path, 'sheet': chosen,
                'available_sheets': sheets}
    except Exception as e:
        return {'df': None,
                'error': f"{type(e).__name__}: {e}",
                'path': path}


_mf_lookup_result = load_mf_lookup()
_mf_lookup_df     = _mf_lookup_result['df']
_mf_lookup_error  = _mf_lookup_result['error']
_mf_lookup_path   = _mf_lookup_result['path']


def _extract_lookup_card_name(cell_text):
    """Pull the card name from the front of a cell — strip parenthetical/dash notes."""
    txt = str(cell_text).strip()
    for sep in [' - ', '  ', ' (']:
        if sep in txt:
            return txt.split(sep)[0].strip()
    return txt


def get_mf_lookup_row(card_variant):
    """
    Fetch the row matching this exact variant (upright OR reversed).
    'AceC' returns the AceC row, 'AceCrx' returns the AceCrx row.
    """
    if _mf_lookup_df is None:
        return None
    target = card_variant.strip().lower()
    target_norm = target.replace(' rx', 'rx').replace(' ', '')
    for _, row in _mf_lookup_df.iterrows():
        cell = str(row['card_cell']).strip()
        if not cell or cell.lower() == 'nan':
            continue
        name = _extract_lookup_card_name(cell)
        n_norm = name.lower().replace(' rx', 'rx').replace(' ', '')
        if n_norm == target_norm:
            return row
    return None


def get_mf_lookup_entry(mf_card):
    row = get_mf_lookup_row(mf_card)
    if row is None:
        row = get_mf_lookup_row(base_card(mf_card))
    if row is None:
        return None
    return {
        'Match Force': row['card_cell'],
        'Winners':     row['winners'],
        'Themes':      row['description'],
    }


# ─── TFIDF FALLBACK ─────────────────────────────────────────────────────
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


# ─── Cache ──────────────────────────────────────────────────────────────
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


# ─── API ────────────────────────────────────────────────────────────────
_API_READY = None
_API_ERROR = None


def check_api():
    global _API_READY, _API_ERROR
    if _API_READY is not None: return _API_READY
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


# ─── HISTORICAL MEMORY ──────────────────────────────────────────────────
@st.cache_data
def build_history(_s1, _mm):
    lc = {k.lower(): k for k in _mm}
    def cl(x): return lc.get(str(x).strip().lower(), str(x).strip())
    def nm(x):
        x = str(x).strip()
        if x in ('1','1.0'): return 'Home'
        if x in ('2','2.0'): return 'Away'
        if x.upper().startswith('X'): return 'Draw'
        return None

    all_rows = []
    for idx, r in _s1.iterrows():
        outcome = nm(r.get('Outcome', ''))
        if not outcome: continue
        h = cl(r.get('Home', '')); a = cl(r.get('Away', '')); mf = cl(r.get('Match Force', ''))
        score = r.get('Score', '')
        if pd.isna(score): score = ''
        h_team = r.get('Home Team', '') if 'Home Team' in r.index else ''
        a_team = r.get('Away Team', '') if 'Away Team' in r.index else ''
        league = r.get('A280', '') if 'A280' in r.index else r.get('League', '')
        all_rows.append({
            'home': h, 'away': a, 'mf': mf, 'outcome': outcome,
            'row_index': int(idx) + 2,
            'score': str(score) if score else '',
            'home_team': str(h_team) if h_team and not pd.isna(h_team) else '',
            'away_team': str(a_team) if a_team and not pd.isna(a_team) else '',
            'league': str(league) if league and not pd.isna(league) else '',
        })

    co_occur = defaultdict(list); mf_index = defaultdict(list)
    record   = defaultdict(lambda: {'Win':0,'Loss':0,'Draw':0,'matches':[]})
    for m in all_rows:
        bh = base_card(m['home']); ba = base_card(m['away']); bmf = base_card(m['mf'])
        for c in {bh, ba, bmf}:
            co_occur[c].append(m)
        mf_index[bmf].append(m)
        if m['outcome'] == 'Home':
            record[(bh,bmf)]['Win']+=1; record[(ba,bmf)]['Loss']+=1
        elif m['outcome'] == 'Away':
            record[(bh,bmf)]['Loss']+=1; record[(ba,bmf)]['Win']+=1
        else:
            record[(bh,bmf)]['Draw']+=1; record[(ba,bmf)]['Draw']+=1
        record[(bh,bmf)]['matches'].append(m)
        record[(ba,bmf)]['matches'].append(m)

    return {
        'record':   {k: dict(v) for k,v in record.items()},
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


# ─── Predict ────────────────────────────────────────────────────────────
def predict(home, away, mf):
    if not check_api():
        return 'Draw', 0.55, ['(API unavailable — check the warning banner at the top of the page)'], \
               None, 0.0, [], None

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

    if result and '_error' in result:
        return 'Draw', 0.55, [f"API call failed: {result['_error']}"], \
               None, 0.0, [], result
    if result and 'call' not in result:
        return 'Draw', 0.55, ['API returned a parseable response but it was missing the required "call" field.'], \
               None, 0.0, [], result
    return 'Draw', 0.55, ['Analysis returned no result (unknown reason)'], None, 0.0, [], None


# ─── HERO STRIP ─────────────────────────────────────────────────────────
def render_hero_strip(home, away, mf):
    cells_html = []
    for role, role_label, role_color, card in [
        ('home', 'Home',        COL_HOME, home),
        ('away', 'Away',        COL_AWAY, away),
        ('mf',   'Match Force', COL_MF,   mf),
    ]:
        url = card_image_url(card, width=320)
        rev = is_reversed(card)
        rotate = 'transform: rotate(180deg);' if rev else ''
        orient_html = '<div style="font-size:12px;color:#666;font-style:italic;margin-top:4px;">reversed</div>' if rev else ''
        if url:
            img_html = f'<img src="{url}" alt="{card}" style="max-height:300px;width:auto;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.18);border:2px solid #fff;{rotate}">'
        else:
            img_html = '<div style="height:280px;display:flex;align-items:center;justify-content:center;color:#999;border:1px dashed #ccc;border-radius:8px;width:200px;">no image</div>'
        cells_html.append(
            f'<div style="flex:1;text-align:center;max-width:240px;">'
            f'<div style="font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:8px;color:{role_color};">{role_label}</div>'
            f'{img_html}'
            f'<div style="font-size:18px;font-weight:600;margin-top:10px;">{card}</div>'
            f'{orient_html}'
            f'</div>'
        )
    full_html = (
        f'<div style="display:flex;justify-content:space-around;align-items:flex-start;'
        f'gap:20px;margin:12px 0 28px 0;padding:20px;'
        f'background:linear-gradient(180deg,#fafafa 0%,#f0f0f0 100%);'
        f'border-radius:12px;border:1px solid #e0e0e0;font-family:sans-serif;">'
        f'{"".join(cells_html)}'
        f'</div>'
    )
    st.components.v1.html(full_html, height=440, scrolling=False)


# ─── 3-COLUMN MEANINGS ──────────────────────────────────────────────────
def get_meaning_for_orientation(card):
    card_l = card.strip().lower()
    direct = full_meaning_map.get(card_l)
    if direct and direct[0]:
        return direct[0]
    base = base_card(card)
    base_entry = full_meaning_map.get(base.lower())
    if base_entry is None: return None
    up_t, rv_t = base_entry
    if is_reversed(card): return rv_t or up_t
    return up_t or rv_t


def _meaning_text_to_html(text):
    if not text:
        return '<em>(no meaning text)</em>'
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if not lines:
        return '<em>(no meaning text)</em>'
    bullet_count = sum(1 for ln in lines if ln.startswith('- '))
    if bullet_count >= 2 and bullet_count >= len(lines) * 0.5:
        items = []
        for ln in lines:
            content = ln[2:].strip() if ln.startswith('- ') else ln
            items.append(f'<li>{content}</li>')
        return f'<ul>{"".join(items)}</ul>'
    return text.replace('\n', '<br>')


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
            text = get_meaning_for_orientation(card) or ''
            body_html = _meaning_text_to_html(text)
            orient_label = 'reversed' if is_reversed(card) else 'upright'
            tag_cls = 'rx' if is_reversed(card) else ''
            html = f"""
              <div class="meaning-panel role-{role}">
                <div class="panel-title">{role_label}</div>
                <div class="card-label">{card}
                  <span class="orient-tag {tag_cls}">{orient_label}</span>
                </div>
                <div>{body_html}</div>
              </div>
            """
            st.markdown(html, unsafe_allow_html=True)


# ─── MF LOOKUP — UPRIGHT + RX side-by-side ──────────────────────────────
def render_mf_lookup_section(mf):
    base = base_card(mf)
    upright_name  = base
    reversed_name = base + 'rx' if not base.lower() in MAJOR_BASES else base + ' rx'

    st.markdown(
        f'<div class="sec-header role-mf">📋 MF Lookup — '
        f'<code>{base}</code> &amp; <code>{reversed_name}</code></div>',
        unsafe_allow_html=True
    )

    if _mf_lookup_df is None:
        st.caption(f"`{MF_LOOKUP_FILE}` not loaded.")
        return

    upright_row  = get_mf_lookup_row(upright_name)
    reversed_row = get_mf_lookup_row(reversed_name)
    if reversed_row is None:
        reversed_row = get_mf_lookup_row(base + 'rx')
    if reversed_row is None:
        reversed_row = get_mf_lookup_row(base + ' rx')

    def cell_text(row, col):
        if row is None: return None
        v = row.get(col)
        if v is None or pd.isna(v): return None
        s = str(v).strip()
        return s if s and s.lower() != 'nan' else None

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            '<div class="lookup-block"><div class="lk-label">'
            f'Description — {upright_name} (upright)</div>'
            f'<div class="lk-text">{cell_text(upright_row, "description") or "<em style=\'color:#999\'>no entry</em>"}</div>'
            '</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="lookup-block"><div class="lk-label">'
            f'Description — {reversed_name} (reversed)</div>'
            f'<div class="lk-text">{cell_text(reversed_row, "description") or "<em style=\'color:#999\'>no entry</em>"}</div>'
            '</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown(
            '<div class="lookup-block"><div class="lk-label">'
            f'Winners — {upright_name} (upright)</div>'
            f'<div class="lk-text">{cell_text(upright_row, "winners") or "<em style=\'color:#999\'>no entry</em>"}</div>'
            '</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="lookup-block"><div class="lk-label">'
            f'Winners — {reversed_name} (reversed)</div>'
            f'<div class="lk-text">{cell_text(reversed_row, "winners") or "<em style=\'color:#999\'>no entry</em>"}</div>'
            '</div>', unsafe_allow_html=True)


# ─── HISTORY TABLES ─────────────────────────────────────────────────────
_OUT_GLYPH = {'Home':'1','Away':'2','Draw':'X'}


def _match_position_bases(match):
    return [
        base_card(match['home']).lower(),
        base_card(match['away']).lower(),
        base_card(match['mf']).lower(),
    ]


def _bases_satisfied_by_match(match, requested_bases):
    pos_bases = _match_position_bases(match)
    pos_remaining = list(pos_bases)
    for req in requested_bases:
        try:
            pos_remaining.remove(req)
        except ValueError:
            return False
    return True


def matches_with_mf_any_orientation(mf):
    bmf = base_card(mf)
    return list(_history['mf_index'].get(bmf, []))


def matches_with_two_cards_any_position(card_a, card_b):
    base_a = base_card(card_a).lower()
    base_b = base_card(card_b).lower()
    requested = [base_a, base_b]
    pool = _history['co_occur'].get(base_card(card_a), [])
    seen = set()
    out = []
    for m in pool:
        key = m.get('row_index', id(m))
        if key in seen: continue
        if _bases_satisfied_by_match(m, requested):
            seen.add(key); out.append(m)
    return out


def matches_with_three_cards_any_position(card_a, card_b, card_c):
    requested = [base_card(c).lower() for c in (card_a, card_b, card_c)]
    pool = _history['co_occur'].get(base_card(card_a), [])
    seen = set()
    out = []
    for m in pool:
        key = m.get('row_index', id(m))
        if key in seen: continue
        if _bases_satisfied_by_match(m, requested):
            seen.add(key); out.append(m)
    return out


def _outcome_cell_html(outcome):
    colour_map = {
        'Home': '#2e8b57',
        'Away': '#1e6fb8',
        'Draw': '#c47a1c',
    }
    colour = colour_map.get(outcome, '#666')
    return (f'<td style="padding:8px 14px;color:{colour};font-weight:600;font-size:15px;">'
            f'{outcome or "—"}</td>')


def _matches_to_html_table(matches):
    show_teams  = any((m.get('home_team') or m.get('away_team')) for m in matches)
    show_score  = any(m.get('score') for m in matches)
    show_league = any(m.get('league') for m in matches)

    headers = []
    if show_league: headers.append('League')
    headers.append('Outcome')
    if show_teams:
        headers.append('Home Team')
        headers.append('Away Team')
    if show_score: headers.append('Score')
    headers.append('Home')
    headers.append('Away')
    headers.append('MF')
    headers.append('Row')

    th_style = ('padding:10px 14px;text-align:left;border-bottom:2px solid #ccc;'
                'font-weight:700;font-size:14px;color:#333;background:#f5f5f5;')
    head_html = ''.join(f'<th style="{th_style}">{h}</th>' for h in headers)

    cell_style = 'padding:8px 14px;font-size:15px;color:#222;'
    mono_style = cell_style + 'font-family:Menlo,Consolas,monospace;font-size:14px;'
    dim_style  = cell_style + 'color:#888;font-size:13px;'
    row_style  = 'border-bottom:1px solid #eee;'

    rows_html = []
    for m in matches:
        cells = []
        if show_league:
            cells.append(f'<td style="{dim_style}">{m.get("league","")}</td>')
        cells.append(_outcome_cell_html(m.get('outcome', '')))
        if show_teams:
            cells.append(f'<td style="{cell_style}">{m.get("home_team","")}</td>')
            cells.append(f'<td style="{cell_style}">{m.get("away_team","")}</td>')
        if show_score:
            cells.append(f'<td style="{cell_style}">{m.get("score","")}</td>')
        cells.append(f'<td style="{mono_style}">{m.get("home","")}</td>')
        cells.append(f'<td style="{mono_style}">{m.get("away","")}</td>')
        cells.append(f'<td style="{mono_style}">{m.get("mf","")}</td>')
        cells.append(f'<td style="{dim_style}">{m.get("row_index","")}</td>')
        rows_html.append(f'<tr style="{row_style}">{"".join(cells)}</tr>')

    return (
        f'<table style="width:100%;border-collapse:collapse;margin:8px 0 24px 0;'
        f'background:#fff;border:1px solid #e8e8e8;border-radius:6px;overflow:hidden;">'
        f'<thead><tr>{head_html}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table>'
    )


def render_history_block(role, label, matches):
    st.markdown(
        f'<div class="sec-header role-{role}">🃏 {label} — '
        f'{len(matches)} match{"es" if len(matches)!=1 else ""}</div>',
        unsafe_allow_html=True
    )
    if not matches:
        st.caption("_no matches found_")
        return
    matches_sorted = sorted(matches, key=lambda m: m.get('row_index', 0))
    st.markdown(_matches_to_html_table(matches_sorted), unsafe_allow_html=True)


def render_full_analysis(result, home, away, mf):
    """
    Renders the trait-level analysis: per-card distinct trait clusters,
    explicit SIM/OPP connections, shared-theme flag, and final rule-derived call.
    NB: contested-trait UI removed in V11.2 (interim).
    """
    icons_type = {'OPP': '🟢', 'SIM': '🔵', 'NONE': '⚫'}

    st.markdown('<div class="sec-header">🧠 AI Reasoning</div>', unsafe_allow_html=True)

    # ── MF traits ───────────────────────────────────────────────────────
    st.markdown(f"**⚡ MF — {mf}** — distinct trait clusters extracted:")
    mf_traits = result.get('mf_traits', [])
    if mf_traits:
        st.markdown('\n'.join(f"- *{t}*" for t in mf_traits))
    else:
        st.caption("_(no MF traits parsed — older cache?)_")

    st.markdown("")

    # ── Per-side trait lists + connection blocks ────────────────────────
    col1, col2 = st.columns(2)
    for col, card, role_label, role_icon, traits_key, type_key, conns_key, note_key in [
        (col1, home, 'Home', '🏠', 'home_traits', 'home_type', 'home_connections', 'home_note'),
        (col2, away, 'Away', '✈',  'away_traits', 'away_type', 'away_connections', 'away_note'),
    ]:
        with col:
            ctype = result.get(type_key, 'NONE')
            icon  = icons_type.get(ctype, '⚫')
            st.markdown(f"**{role_icon} {role_label} — {card}**")
            st.markdown(f"{icon} **{ctype}**")

            traits = result.get(traits_key, [])
            if traits:
                with st.expander(f"Distinct trait clusters extracted ({len(traits)})", expanded=False):
                    st.markdown('\n'.join(f"- *{t}*" for t in traits))

            conns = result.get(conns_key, [])
            if conns:
                st.markdown("*Trait-level connections to the MF:*")
                for c in conns:
                    kind     = c.get('kind', '')
                    kind_ico = icons_type.get(kind, '·')
                    mf_t     = c.get('mf_trait', '')
                    team_t   = c.get(f'{role_label.lower()}_trait', '') or c.get('team_trait', '')
                    note     = c.get('note', '')
                    st.markdown(
                        f"- {kind_ico} **{kind}** &nbsp; "
                        f"MF: *{mf_t}* &nbsp;↔&nbsp; {role_label}: *{team_t}*"
                    )
                    if note:
                        st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {note}")
            elif ctype == 'NONE':
                st.caption("_No literal trait-level mirror or opposite found._")

            note = result.get(note_key, '')
            if note:
                st.markdown(f"*{note}*")

    # ── Shared-theme flag ───────────────────────────────────────────────
    if result.get('shared_theme'):
        st_note = result.get('shared_theme_note', '') or 'all three cards share a dominant theme'
        st.warning(f"⚠ **Shared theme detected (all three cards):** {st_note} — rule layer will resolve to Draw.")

    # ── History signal ──────────────────────────────────────────────────
    hist_sig = result.get('history_signal', '')
    if hist_sig and hist_sig.lower() not in ('none', 'neutral', ''):
        st.markdown(f"**📊 History:** {hist_sig}")

    # ── Final call ──────────────────────────────────────────────────────
    call = result.get('call', 'Draw')
    call_icons = {'Home':'🟢','Away':'🔵','Draw':'🟡'}
    ci = call_icons.get(call, '⚪')
    st.markdown(f"### 🎯 {ci} Call: **{call}**")
    reason = result.get('reason', '')
    if reason:
        st.markdown(f"*{reason}*")

    # ── Rule path ───────────────────────────────────────────────────────
    rule_path = result.get('rule_path')
    if rule_path:
        st.caption(f"⚙️ Rule applied: **{rule_path}**")

    # ── Free-form AI reasoning ──────────────────────────────────────────
    ai_reasoning = result.get('reasoning', '')
    if ai_reasoning:
        with st.expander("AI's own reasoning notes", expanded=False):
            st.markdown(ai_reasoning)


# ─── UI ────────────────────────────────────────────────────────────────
inject_css()

st.title("🔮 Tarot Predictor V11.2 (interim)")
st.caption(
    "Winner is the team whose card best CONTROLS the Match Force. "
    "The mirror (SIM) takes precedence; the destabiliser (OPP) is the fallback."
)

api_ok     = check_api()
cache_size = len(st.session_state.rcache)
if api_ok:
    st.success(f"✅ Semantic API active — {cache_size} resonance pair{'s' if cache_size != 1 else ''} cached to disk")
else:
    err = f": {_API_ERROR}" if _API_ERROR else ""
    st.warning(f"⚠ Semantic API unavailable{err} — using TF-IDF keyword fallback.")

if _mf_lookup_df is not None:
    sheet_used = _mf_lookup_result.get('sheet', '?')
    st.info(f"📋 MF Lookup loaded — {len(_mf_lookup_df)} entries from `{_mf_lookup_path}` (sheet: `{sheet_used}`)")
else:
    st.warning(
        f"📋 MF Lookup NOT loaded — looked for `{MF_LOOKUP_FILE}`. "
        f"Error: **{_mf_lookup_error or 'unknown'}**. "
        f"Make sure the file is in the same folder as `app.py` "
        f"(currently `{_HERE}`)."
    )

st.divider()

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

_mfd = get_mf_draw_rate(mf)
if _mfd['category'] == 'high':
    st.warning(f"⚠ **{mf}** historically draws in **{int(_mfd['rate']*100)}%** of matches "
               f"({_mfd['draws']}/{_mfd['total']}) — static/stuck MF, draws common")
elif _mfd['category'] == 'low':
    st.success(f"✓ **{mf}** historically draws in only **{int(_mfd['rate']*100)}%** of matches "
               f"({_mfd['draws']}/{_mfd['total']}) — decisive MF, draws rare")
elif _mfd['category'] == 'normal':
    st.caption(f"📊 **{mf}** historical draw rate: {int(_mfd['rate']*100)}% "
               f"({_mfd['draws']}/{_mfd['total']}) — normal range")
else:
    st.caption(f"📊 **{mf}** has only {_mfd['total']} past appearances — insufficient history")


if st.button("Predict", type="primary", use_container_width=True):
    with st.spinner("Reading cards and history…"):
        pred, conf, explanation, hist_pred, hist_weight, hist_notes, full_analysis = predict(home, away, mf)

    # Hero strip
    render_hero_strip(home, away, mf)

    # Prediction header
    icons = {'Home':'🟢','Away':'🔵','Draw':'🟡'}
    col1, col2 = st.columns([3, 1])
    with col1: st.markdown(f"## {icons.get(pred,'⚪')} Prediction: **{pred}**")
    with col2: st.metric("Confidence", f"{conf:.0%}")
    st.progress(conf)
    if conf >= 0.80:   st.caption("High confidence.")
    elif conf >= 0.65: st.caption("Good confidence.")
    elif conf >= 0.55: st.caption("Moderate confidence — exercise judgement.")
    else:              st.caption("Lower confidence — use as a pointer only.")

    # Meanings 3-column
    render_meanings_three_column(home, away, mf)

    # AI analysis
    if full_analysis and '_error' not in full_analysis:
        render_full_analysis(full_analysis, home, away, mf)
    else:
        st.markdown('<div class="sec-header">⚠ Analysis unavailable</div>', unsafe_allow_html=True)
        if explanation:
            for line in explanation:
                st.error(line)
        if full_analysis and '_error' in full_analysis:
            st.markdown("**Raw error from the analysis engine:**")
            st.code(full_analysis['_error'], language=None)
            st.caption("If this is a JSON parse error, the model returned text that wasn't a valid JSON object. "
                       "Try clicking Predict again — transient parse failures are common.")

    # MF Lookup
    render_mf_lookup_section(mf)

    # ── Historical match blocks ─────────────────────────────────────────
    render_history_block(
        'home',
        f"<code>{base_card(mf)}</code> + <code>{base_card(home)}</code>",
        matches_with_two_cards_any_position(mf, home)
    )

    render_history_block(
        'away',
        f"<code>{base_card(mf)}</code> + <code>{base_card(away)}</code>",
        matches_with_two_cards_any_position(mf, away)
    )

    render_history_block(
        'pair',
        f"<code>{base_card(home)}</code> + <code>{base_card(away)}</code>",
        matches_with_two_cards_any_position(home, away)
    )

    render_history_block(
        'pair',
        f"All three: <code>{base_card(home)}</code> + <code>{base_card(away)}</code> + <code>{base_card(mf)}</code>",
        matches_with_three_cards_any_position(home, away, mf)
    )

    render_history_block(
        'mf',
        f"<code>{base_card(mf)}</code> as MF",
        matches_with_mf_any_orientation(mf)
    )

    # Footer
    analysis_count = sum(1 for k in st.session_state.rcache if k.startswith(ANALYSIS_PREFIX))
    pair_count     = sum(1 for k in st.session_state.rcache if not k.startswith(ANALYSIS_PREFIX))
    st.caption(f"Cache: {analysis_count} match analyses · {pair_count} pair resonances stored.")


st.divider()
with st.expander("Decision rules (v6 interim)"):
    st.markdown("""
**A team controls the MF by:**
- **SIM** — mirroring the MF (equal, aligning energy)
- **OPP** — destabilising the MF (equal, opposing energy)

**Primary rules:**
1. Only one card connects (other is NONE) → that team wins
2. One SIM, one OPP → **SIM wins** (the mirror controls; opposition is the fallback)
3. Both SIM → **Draw** (alignments compete, neither uniquely controls)
4. Both OPP → **Draw** (both block the MF, neither uniquely controls)
5. Both NONE → **Draw**
6. All three cards share one dominant theme → **Draw** (shared-theme)

**Duplicate-card rules** (highest priority — applied first):
A team card has the same base as the MF (e.g. Home = Death, MF = Death rx).
- Remaining card is SIM → MF matches with its duplicate → duplicate-side team wins
- Remaining card is OPP → opposing force takes control → remaining team wins
- Remaining card is NONE → duplicate is the only connection → duplicate-side team wins

**What's gone vs V11.1:**
- Hierarchy (Major > King > Queen > …) — removed entirely
- Pip-pip exception — removed
- Hierarchy exception (SIM-higher vs OPP-lower → Draw) — removed
- Contested-trait deadlock — removed (was firing too often, causing
  most matches to resolve to Draw)
- OPP was previously the active-control type that won SIM/OPP splits — flipped:
  SIM now wins, reflecting the rule "the MF prefers an aligning partner first;
  opposition is the fallback when no alignment exists"

**Specificity:** the AI still compares at the TRAIT level (6-10 distinct trait
clusters per card). Home and Away may connect to different MF aspects.
Lesser-used niche traits count. When a card has both SIM and OPP candidate
connections, the AI weighs both and tie-breaks in favour of **SIM**.

**Reversed cards with dual/opposing meanings:**
The AI is instructed to read all three cards together and pick the polarity
of each reversed card that makes the energetic story coherent — using the
other cards in play to disambiguate which polarity is active.

**Per-card data shown after each prediction:**
- Hero strip — three card images at top
- 3-column meanings (full text from meanings sheet)
- MF Lookup — Description (col D) + Winners (col C) for both upright AND reversed MF
- Historical match blocks — MF+Home, MF+Away, Home+Away, all three, then all MF games at the bottom

**Note on cache:**
Existing cached predictions from V11.1 will be re-processed under the new
rules. If you want fully fresh AI calls (rather than reusing old AI
classifications under the new rules), delete `resonance_cache.json` or set
`REUSE_OLDER_AI_OUTPUT = False` at the top of `analysis_engine.py`.
""")
