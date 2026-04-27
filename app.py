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
RANK_LABEL = {6:'Major Arcana', 5:'King', 4:'Queen', 3:'Knight', 2:'Page', 1:'pip'}

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
    font-size: 13px;
    line-height: 1.5;
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

    Tolerant to sheet naming: tries 'MF Lookup' first, then any sheet whose name
    contains 'lookup' (case-insensitive), then falls back to the first sheet.
    """
    path = _resolve(MF_LOOKUP_FILE)
    if not os.path.exists(path):
        return {'df': None,
                'error': f"File not found at {path} (also tried {_HERE} and CWD)",
                'path': None}
    try:
        # Find the right sheet
        xl = pd.ExcelFile(path)
        sheets = xl.sheet_names
        chosen = None
        # exact match first
        for s in sheets:
            if s == 'MF Lookup':
                chosen = s; break
        # case-insensitive 'lookup' match
        if chosen is None:
            for s in sheets:
                if 'lookup' in s.lower():
                    chosen = s; break
        # first sheet as final fallback
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


# Backwards-compat for analysis_engine.py
def get_mf_lookup_entry(mf_card):
    row = get_mf_lookup_row(mf_card)
    if row is None:
        # try base too
        row = get_mf_lookup_row(base_card(mf_card))
    if row is None:
        return None
    # Return as dict shaped like the analysis_engine expects
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


def card_rank(card):
    c = card.lower().replace(' rx','').replace('rx','').strip()
    if c in MAJOR_BASES: return 6
    if c.startswith('k') and not c.startswith('kn'): return 5
    if c.startswith('q'): return 4
    if c.startswith('kn'): return 3
    if c.startswith('p') and len(c) == 2: return 2
    return 1


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
    return 'Draw', 0.55, ['(API unavailable)'], None, 0.0, [], None


# ─── HERO STRIP — rendered via components.v1.html so HTML is parsed ─────
def render_hero_strip(home, away, mf):
    """
    Render the three card images using components.v1.html (real iframe).
    Order: Home, Away, Match Force.
    Reversed cards are rotated 180°.
    """
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


# ─── MF LOOKUP — UPRIGHT + RX side-by-side ──────────────────────────────
def render_mf_lookup_section(mf):
    """
    Two-column layout:
      Left: Description column (col D) for upright + rx
      Right: Winners column (col C) for upright + rx
    Displayed under header "MF Lookup — <mf base>".
    """
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

    # pandas Series can't be used with `or` — check explicitly for None
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

    # Left = Description (col D), Right = Winners (col C)
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


def _matches_to_df(matches):
    if not matches: return None
    rows = []
    for m in matches:
        rows.append({
            'Row':       m.get('row_index',''),
            'Out':       _OUT_GLYPH.get(m.get('outcome',''), m.get('outcome','')),
            'League':    (m.get('league') or '')[:24],
            'Home Team': (m.get('home_team') or '')[:22],
            'Away Team': (m.get('away_team') or '')[:22],
            'H':         m['home'],
            'A':         m['away'],
            'MF':        m['mf'],
            'Score':     m.get('score',''),
        })
    df = pd.DataFrame(rows)
    return df.sort_values('Row', ascending=True).reset_index(drop=True)


def _card_appears_anywhere(match, card):
    """True if card (any orientation) appears in any of home/away/mf positions."""
    forms = all_orientation_strings(card)
    return (match['home'].strip().lower() in forms
            or match['away'].strip().lower() in forms
            or match['mf'].strip().lower() in forms)


def _match_position_bases(match):
    """Return list of base cards present in this match across home/away/mf positions."""
    return [
        base_card(match['home']).lower(),
        base_card(match['away']).lower(),
        base_card(match['mf']).lower(),
    ]


def _bases_satisfied_by_match(match, requested_bases):
    """
    Given a list of requested base cards (e.g. ['10c','10c','10c']),
    return True iff the match's three positions contain ENOUGH copies of each.

    This handles duplicates correctly: if user asks for 10C+10C, we need 10C in
    at least TWO positions of the same match. If they ask 10C+10C+10C, all three
    positions must be 10C (any orientation).
    """
    pos_bases = _match_position_bases(match)
    pos_remaining = list(pos_bases)
    for req in requested_bases:
        try:
            pos_remaining.remove(req)
        except ValueError:
            return False
    return True


def matches_with_mf_any_orientation(mf):
    """Every match where the MF column == this card's base (any orientation)."""
    bmf = base_card(mf)
    return list(_history['mf_index'].get(bmf, []))


def matches_with_two_cards_any_position(card_a, card_b):
    """
    Every match where BOTH requested cards (any orientation) appear in any
    position. Handles duplicates: if card_a base == card_b base, the match
    must contain that base in at least TWO positions.
    """
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
    """
    Every match where ALL THREE requested cards (any orientation) appear in
    any position. Handles duplicates: if two or all three inputs share a base,
    that base must occupy that many positions in the match.
    """
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


def render_history_block(role, label, matches):
    st.markdown(
        f'<div class="sec-header role-{role}">🃏 {label} — '
        f'{len(matches)} match{"es" if len(matches)!=1 else ""}</div>',
        unsafe_allow_html=True
    )
    df = _matches_to_df(matches)
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("_no matches found_")


def render_full_analysis(result, home, away, mf):
    icons_type = {'OPP': '🟢', 'SIM': '🔵', 'NONE': '⚫'}
    st.markdown('<div class="sec-header">🧠 AI Reasoning</div>', unsafe_allow_html=True)
    st.markdown(f"**⚡ {mf}** — {result.get('mf_quality','')}")
    phrases = result.get('mf_key_phrases', [])
    if phrases: st.caption('  ·  '.join(f'*{p}*' for p in phrases))

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
            if tied: st.caption(f"Tied with other card (neutral): {', '.join(tied)}")
            note = result.get(note_key, '')
            if note: st.caption(note)

    st.markdown("**📊 History summary**")
    for key, label in [('history_mf_pattern','MF pattern'),('history_direct','Direct'),
                       ('history_co','Together')]:
        val = result.get(key, '')
        if val and val.lower() not in ('none','none found',''):
            st.markdown(f"- **{label}:** {val}")
    hc = result.get('history_confirms', 'None')
    call = result.get('call', 'Draw')
    if hc and hc not in ('None',''):
        if hc == call: st.markdown(f"- ✅ History agrees: **{hc}**")
        elif hc == 'Conflicts': st.markdown(f"- ⚠️ History conflicts with meaning analysis")
        else: st.markdown(f"- ➖ History neutral / insufficient data")
    draw_sig = result.get('draw_signal', '')
    if draw_sig and draw_sig.lower() not in ('no draw signal','none',''):
        st.markdown(f"- 🎲 **Draw prior:** {draw_sig}")
    call_icons = {'Home':'🟢','Away':'🔵','Draw':'🟡'}
    ci = call_icons.get(call, '⚪')
    st.markdown(f"### 🎯 {ci} Call: **{call}**")
    reason = result.get('reason', '')
    if reason: st.markdown(f"*{reason}*")

    # Show which rule path was applied (if any) and the AI's pre-rule call
    rule_path = result.get('rule_path')
    if rule_path:
        ai_call = result.get('_ai_call')
        if ai_call and ai_call != call:
            st.caption(f"⚙️ Rule applied: **{rule_path}** "
                       f"(AI's initial call was *{ai_call}*, overridden by rules)")
        else:
            st.caption(f"⚙️ Rule applied: **{rule_path}**")


# ─── UI ────────────────────────────────────────────────────────────────
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

    # Hero strip — three card images, rendered via components.v1.html
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
    elif explanation:
        st.markdown('<div class="sec-header">Reasoning (fallback)</div>', unsafe_allow_html=True)
        for i, line in enumerate(explanation):
            st.markdown(f"{'→' if i < 2 else '⟹'} {line}")
        if full_analysis and '_error' in full_analysis:
            st.warning(f"Holistic analysis failed: {full_analysis['_error']}")

    # MF Lookup — upright + rx, description left, winners right
    render_mf_lookup_section(mf)

    # ── Historical match blocks ─────────────────────────────────────────
    # 1. All matches where current MF was the MF (any orientation)
    render_history_block(
        'mf',
        f"All matches where <code>{base_card(mf)}</code> was the MF (any orientation)",
        matches_with_mf_any_orientation(mf)
    )

    # 2. MF + Home appeared together in any position, any orientation
    render_history_block(
        'home',
        f"<code>{base_card(mf)}</code> + <code>{base_card(home)}</code> appeared together (any position, any orientation)",
        matches_with_two_cards_any_position(mf, home)
    )

    # 3. MF + Away appeared together in any position, any orientation
    render_history_block(
        'away',
        f"<code>{base_card(mf)}</code> + <code>{base_card(away)}</code> appeared together (any position, any orientation)",
        matches_with_two_cards_any_position(mf, away)
    )

    # 4. Home + Away appeared together in any position, any orientation
    render_history_block(
        'pair',
        f"<code>{base_card(home)}</code> + <code>{base_card(away)}</code> appeared together (any position, any orientation)",
        matches_with_two_cards_any_position(home, away)
    )

    # 5. All three cards appeared together in any position, any orientation
    render_history_block(
        'pair',
        f"All three: <code>{base_card(home)}</code> + <code>{base_card(away)}</code> + <code>{base_card(mf)}</code> together (any position, any orientation)",
        matches_with_three_cards_any_position(home, away, mf)
    )

    # Footer
    analysis_count = sum(1 for k in st.session_state.rcache if k.startswith(ANALYSIS_PREFIX))
    pair_count     = sum(1 for k in st.session_state.rcache if not k.startswith(ANALYSIS_PREFIX))
    st.caption(f"Cache: {analysis_count} match analyses · {pair_count} pair resonances stored.")


st.divider()
with st.expander("Decision rules"):
    st.markdown("""
**Resonance rules:**
1. Only one card balances MF → that team wins
2. Both balance, different types → OPP beats SIM (deficit energy is stronger)
3. Both balance, same type, score gap ≥ 0.15 → higher score wins
4. Both balance, same type, similar scores → card hierarchy
5. Equal everything → Draw  |  6. Neither balances → Draw

**Per-card data shown after each prediction:**
- Hero strip — three card images at top
- 3-column meanings (full text from meanings sheet)
- MF Lookup — Description (col D) + Winners (col C) for both upright AND reversed MF
- Historical match tables — MF alone, MF+Home, MF+Away, Home+Away, all three together
""")
