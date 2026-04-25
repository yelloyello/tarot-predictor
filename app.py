"""
Tarot Predictor V10
Methodology: homeostatic balancing via semantic resonance.
New in V10: MF-identical pattern — when MF == a team card, the other team's
card is compared to the MF; OPP → other team wins, SIM → matching team wins.

New in V10.1:
- Literal Meaning Analysis: shows exact phrases from meanings sheet side-by-side
- MF Lookup Reference: reads your mf_lookup.xlsx notes for the current MF
- Improved AI prompt: references actual keyword phrases from meanings
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re
from analysis_engine import (analyse_match_full, analysis_to_prediction,
                              gather_history_text, ANALYSIS_PREFIX)
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Tarot Predictor V10", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CACHE_FILE      = "resonance_cache.json"
DATA_FILE       = "data.xlsx"
MF_LOOKUP_FILE  = "mf_lookup.xlsx"
API_MODEL       = "claude-sonnet-4-20250514"

os.environ["ANTHROPIC_API_KEY"] = ""   # paste key here temporarily

BALANCE_MIN   = 0.35
ASPECTS_DELTA = 0.15

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
    mm = {}
    for _, row in _df.iterrows():
        card    = str(row.iloc[0]).strip()
        meaning = str(row.iloc[1]).strip()
        if card and card != 'nan' and meaning and meaning != 'nan':
            mm[card] = meaning
    return mm

meaning_map = build_meaning_map(meanings_df)
_l2c        = {k.lower(): k for k in meaning_map}

def canonical(x):
    return _l2c.get(str(x).strip().lower(), str(x).strip())

cards = sorted(meaning_map.keys())


# ─────────────────────────────────────────────────────────────────────────────
# MF LOOKUP  — loads mf_lookup.xlsx (optional)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_mf_lookup():
    if not os.path.exists(MF_LOOKUP_FILE):
        return None
    try:
        df = pd.read_excel(MF_LOOKUP_FILE, sheet_name='MF Lookup')
        df.columns = ['Idx', 'Match Force', 'Winners', 'Themes']
        return df
    except Exception:
        return None

_mf_lookup_df = load_mf_lookup()

def _extract_lookup_card_name(cell_text):
    """
    The Match Force column sometimes has 'Death' and sometimes
    'Death - loss (AceSrx, 3S), preventing transformation...'.
    Pull out just the card name from the front.
    """
    txt = str(cell_text).strip()
    for sep in [' - ', '  ', ' (']:
        if sep in txt:
            return txt.split(sep)[0].strip()
    return txt

def get_mf_lookup_entry(mf_card):
    """Return the MF Lookup row for a card (None if not found)."""
    if _mf_lookup_df is None:
        return None
    target_lower = mf_card.strip().lower()
    target_base  = base_card(mf_card).lower()
    for _, row in _mf_lookup_df.iterrows():
        cell = str(row['Match Force']).strip()
        if not cell or cell == 'nan':
            continue
        name   = _extract_lookup_card_name(cell).lower()
        name_b = base_card(name)
        if (name == target_lower or name_b == target_base
                or name == target_lower.replace(' ', '')):
            return row
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LITERAL MEANING ANALYSIS  — NEW
# Parses comma-separated keyword phrases from the meanings sheet and identifies
# exact word-level overlaps so you can see WHICH specific phrases connect.
# ─────────────────────────────────────────────────────────────────────────────
def parse_meaning_phrases(text):
    """Split a meaning string into individual keyword phrases (comma/semicolon/newline separated)."""
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
    """
    Return (team_phrase, target_phrase, [shared_words]) for pairs sharing
    at least one significant word. Deduplicated by shared-word set.
    """
    hits = []
    seen = set()
    for tp in team_phrases:
        tw = _sig_words(tp)
        if not tw:
            continue
        for xp in target_phrases:
            xw = _sig_words(xp)
            shared = tw & xw
            if shared:
                key = frozenset(shared)
                if key not in seen:
                    seen.add(key)
                    hits.append((tp, xp, sorted(shared)))
    return hits


def literal_meaning_analysis(team_card, mf_card):
    """
    Compare team_card meaning to mf_card meaning (SIM word matches)
    and to mf_card's complement meaning (OPP indicators).
    Returns: team_phrases, mf_phrases, comp_card, comp_phrases, sim_hits, opp_hits
    """
    t_text   = meaning_map.get(team_card, '')
    mf_text  = meaning_map.get(mf_card, '')
    comp_card = get_complement(mf_card)
    comp_text = meaning_map.get(comp_card, '') if comp_card else ''

    t_phrases    = parse_meaning_phrases(t_text)
    mf_phrases   = parse_meaning_phrases(mf_text)
    comp_phrases = parse_meaning_phrases(comp_text) if comp_text else []

    sim_hits = find_word_overlaps(t_phrases, mf_phrases)
    opp_hits = find_word_overlaps(t_phrases, comp_phrases)

    return t_phrases, mf_phrases, comp_card, comp_phrases, sim_hits, opp_hits


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
# API RESONANCE — IMPROVED PROMPT references actual meaning keywords
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
- OPP (Opposition / Deficit): the card provides what the Match Force lacks — the equal and opposite resolving force. Look for keywords in the team card meaning that directly negate or oppose specific keywords in the Match Force meaning. Examples: "trust" opposes "repressed intuition"; "letting go" opposes "rigidity"; "clarity" opposes "confusion".
- NONE: no meaningful energetic relationship.

Match Force card: {mf}
Match Force meaning (exact keywords): {mf_meaning}

Complement of Match Force — {comp_card} (this is what the MF LACKS / its opposite state):
{comp_meaning}

Team card: {card}
Team card meaning (exact keywords): {card_meaning}

Respond ONLY with a JSON object, no other text or markdown:
{{"type": "SIM", "score": 0.75, "aspects": ["phrase from team meaning", "phrase from MF meaning it connects to", "connection note"]}}

Rules:
- type: "SIM", "OPP", or "NONE"
- score: 0.0–1.0 (strength of resonance — how specifically and directly the keyword phrases connect)
- aspects: 2–4 short items. QUOTE the actual keyword phrases from the meanings above, not generic descriptions.
  SIM example: ["transformation", "spiritual transformation", "shared theme: endings and renewal"]
  OPP example: ["repressed intuition", "opposes: letting go and trust", "HP rx lacks what Death calls for"]
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
# HISTORICAL MEMORY
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def build_history(_s1, _mm):
    lc = {k.lower(): k for k in _mm}
    def cl(x): return lc.get(str(x).strip().lower(), str(x).strip())
    def nm(x):
        x = str(x).strip()
        if x == '1': return 'Home'
        if x == '2': return 'Away'
        if x.upper().startswith('X'): return 'Draw'
        return None

    record   = defaultdict(lambda: {'Win': 0, 'Loss': 0, 'Draw': 0, 'matches': []})
    exact    = defaultdict(lambda: {'Home': 0, 'Away': 0, 'Draw': 0, 'matches': []})
    no_match = defaultdict(lambda: {'Home': 0, 'Away': 0, 'Draw': 0, 'matches': []})

    for _, r in _s1.iterrows():
        outcome = nm(r.get('Outcome', ''))
        if not outcome: continue
        h = cl(r.get('Home', '')); a = cl(r.get('Away', '')); mf = cl(r.get('Match Force', ''))
        bh = base_card(h); ba = base_card(a); bmf = base_card(mf)
        match_ctx = {'home': h, 'away': a, 'mf': mf, 'outcome': outcome}

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
    for _, r in _s1.iterrows():
        outcome = nm(r.get('Outcome', ''))
        if not outcome: continue
        h = cl(r.get('Home', '')); a = cl(r.get('Away', '')); mf = cl(r.get('Match Force', ''))
        match_ctx = {'home': h, 'away': a, 'mf': mf, 'outcome': outcome}
        for card in {base_card(h), base_card(a), base_card(mf)}: co_occur[card].append(match_ctx)
        mf_index[base_card(mf)].append(match_ctx)

    def fr(d): return {k: {'Win': v['Win'],'Loss': v['Loss'],'Draw': v['Draw'],'matches': v['matches']} for k,v in d.items()}
    def fe(d): return {k: {'Home': v['Home'],'Away': v['Away'],'Draw': v['Draw'],'matches': v['matches']} for k,v in d.items()}
    return {'record': fr(record), 'exact': fe(exact), 'no_match': fe(no_match),
            'co_occur': dict(co_occur), 'mf_index': dict(mf_index)}

_history = build_history(sheet1, meaning_map)


def get_mf_draw_rate(mf):
    """
    Empirical draw stats for a given MF card.
    Returns dict with: draws, total, rate (0-1 or None), category (high/normal/low/insufficient).
    Categories from analysis: high (>=35%), low (<=15%), normal in between.
    Requires >=5 historical appearances to compute a rate.
    """
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


def _wld_summary(rec):
    w = rec.get('Win',0); l = rec.get('Loss',0); d = rec.get('Draw',0)
    total = w + l + d
    if total == 0: return 0.0, 0, ''
    wr = w/total
    return wr, total, f"W{w} L{l} D{d} ({wr:.0%} win rate, {total} match{'es' if total>1 else ''})"

def _format_matches(matches):
    return [f"  `{m['home']}` vs `{m['away']}` / MF `{m['mf']}` → **{m['outcome']}**"
            for m in matches]


def historical_notes(home, away, mf):
    rec = _history['record']; exact = _history['exact']
    notes = []; MIN = 2
    bh, ba, bmf = base_card(home), base_card(away), base_card(mf)

    for (bc, team_label, hp_key, ap_key) in [
        (bh, f"**{home}** as team card when **{mf}** was MF", True, False),
        (ba, f"**{away}** as team card when **{mf}** was MF", False, True),
    ]:
        r = rec.get((bc, bmf), {})
        total = r.get('Win',0)+r.get('Loss',0)+r.get('Draw',0)
        if total >= MIN:
            wr, _, summary = _wld_summary(r)
            hp = round(wr,2) if hp_key else round(r.get('Loss',0)/total,2)
            ap = round(wr,2) if ap_key else round(r.get('Loss',0)/total,2)
            notes.append({'label': team_label, 'summary': summary,
                          'matches': _format_matches(r.get('matches',[])),
                          'home_pts': hp, 'away_pts': ap})

    for (bc2, reverse_mf, team_card_name, score_home) in [
        (bmf, bh, home, True), (bmf, ba, away, False)
    ]:
        r = rec.get((bc2, reverse_mf), {})
        total = r.get('Win',0)+r.get('Loss',0)+r.get('Draw',0)
        if total >= MIN:
            wr, _, summary = _wld_summary(r)
            hp = round(wr,2) if score_home else round(r.get('Loss',0)/total,2)
            ap = round(wr,2) if not score_home else round(r.get('Loss',0)/total,2)
            notes.append({'label': f"**{mf}** (current MF energy) as team card when **{team_card_name}** was MF",
                          'summary': summary+" — these energies have historical resonance",
                          'matches': _format_matches(r.get('matches',[])),
                          'home_pts': hp, 'away_pts': ap})

    r = exact.get((bh, ba, bmf), {})
    total = r.get('Home',0)+r.get('Away',0)+r.get('Draw',0)
    if total >= 1:
        parts = ', '.join(f"{o} {r[o]}×" for o in ('Home','Away','Draw') if r.get(o,0)>0)
        hp = r.get('Home',0)/total*2; ap = r.get('Away',0)/total*2
        notes.append({'label': f"Exact triplet **{home}** vs **{away}** / MF **{mf}**",
                      'summary': f"{parts} ({total} direct match{'es' if total>1 else ''})",
                      'matches': _format_matches(r.get('matches',[])),
                      'home_pts': hp, 'away_pts': ap})

    has_direct = any('as team card when' in n['label'] for n in notes)
    if not has_direct:
        nm_rec = _history['no_match'].get(bmf,{})
        nm_total = nm_rec.get('Home',0)+nm_rec.get('Away',0)+nm_rec.get('Draw',0)
        if nm_total >= 2:
            hn=nm_rec.get('Home',0); an=nm_rec.get('Away',0); dn=nm_rec.get('Draw',0)
            dom = max(('Home','Away','Draw'), key=lambda o: nm_rec.get(o,0))
            notes.append({
                'label': f"🔍 **{mf}** no-match background ({nm_total} matches where MF ≠ either team card)",
                'summary': (f"Home {hn}× Away {an}× Draw {dn}× — when **{mf}** didn't match either team, "
                            f"**{dom}** won {nm_rec.get(dom,0)/nm_total:.0%}. Weak directional signal only."),
                'matches': _format_matches(nm_rec.get('matches',[])[-10:]),
                'home_pts': round((hn+dn*0.5)/nm_total*0.5,2),
                'away_pts': round((an+dn*0.5)/nm_total*0.5,2),
            })
    return notes


def _co_occur_note(home, away, mf):
    mf_idx = _history.get('mf_index',{})
    bh=base_card(home); ba=base_card(away); bmf=base_card(mf)

    def _tstr(t): return f"Home {t['Home']}× Away {t['Away']}× Draw {t['Draw']}×"
    def _dom(t):
        tot=sum(t.values()); return (max(t,key=t.get), t[max(t,key=t.get)], tot) if tot else (None,0,0)

    def _scan(mf_base, targets, lbl_mf, lbl_tg):
        hits={}
        for m in mf_idx.get(mf_base,[]):
            if {base_card(m['home']),base_card(m['away'])} & targets:
                hits[(m['home'],m['away'],m['mf'])] = m
        if not hits: return None
        ah=list(hits.values()); tally={'Home':0,'Away':0,'Draw':0}
        for m in ah: tally[m['outcome']] = tally.get(m['outcome'],0)+1
        d,dn,tot = _dom(tally)
        return {'label': f"🔎 When **{lbl_mf}** was MF: {lbl_tg} appeared as team card in {tot} past match{'es' if tot>1 else ''}",
                'summary': f"{_tstr(tally)} — **{d}** dominant ({dn/tot:.0%}). Display only.",
                'matches': _format_matches(ah[:15]), 'home_pts':0.0,'away_pts':0.0,'_display_only':True}

    notes=[]
    if bh!=bmf and bh!=ba:
        n=_scan(bh,{bmf,ba},home,f"**{mf}** or **{away}**");
        if n: notes.append(n)
    if ba!=bmf and ba!=bh:
        n=_scan(ba,{bmf,bh},away,f"**{mf}** or **{home}**");
        if n: notes.append(n)
    n=_scan(bmf,{bh,ba},mf,f"**{home}** or **{away}**")
    if n: notes.append(n)
    return notes


def historical_signal(home, away, mf):
    notes = historical_notes(home, away, mf)
    for co in _co_occur_note(home, away, mf): notes.append(co)
    if not notes: return None, 0.0, notes
    hs = sum(n['home_pts'] for n in notes)
    as_ = sum(n['away_pts'] for n in notes)
    total = hs + as_
    if total==0 or abs(hs-as_)<0.1: return None, 0.0, notes
    if hs > as_:
        return 'Home', round(min((hs-as_)/max(total,1),1.0),2), notes
    return 'Away', round(min((as_-hs)/max(total,1),1.0),2), notes


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
    """
    Primary path: holistic single API call (analyse_match_full).
    Fallback: existing decide() + TF-IDF when API unavailable.
    Returns (pred, conf, lines, hist_pred, hist_weight, hist_notes, full_analysis)
    """
    # ── Try holistic analysis first ──────────────────────────────────────────
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

    # ── Fallback: original decide() + historical signal ──────────────────────
    pred, conf, lines = decide(home, away, mf)
    hist_pred, hist_weight, hist_notes = historical_signal(home, away, mf)
    if hist_pred is not None and hist_weight >= 0.25:
        conf = (conf * 0.65 + hist_weight * 0.35) if hist_pred == pred else conf * 0.72
    return pred, round(min(conf, 0.95), 2), lines, hist_pred, hist_weight, hist_notes, None


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS — Literal Meaning Analysis renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_meaning_analysis(home, away, mf):
    """Side-by-side literal phrase comparison for both team cards vs MF."""
    comp = get_complement(mf)

    h_phrases, mf_phrases, _, comp_phrases, h_sim, h_opp = literal_meaning_analysis(home, mf)
    a_phrases, _,           _, _,            a_sim, a_opp = literal_meaning_analysis(away, mf)

    h_sim_words = {w for _, _, ws in h_sim for w in ws}
    h_opp_words = {w for _, _, ws in h_opp for w in ws}
    a_sim_words = {w for _, _, ws in a_sim for w in ws}
    a_opp_words = {w for _, _, ws in a_opp for w in ws}

    st.caption(
        "🟢 = phrase shares a word with the MF meaning (SIM candidate)  |  "
        "🟠 = phrase shares a word with the MF complement (OPP indicator)  |  "
        "Unmarked = no direct word match found."
    )

    # ── MF and its complement ─────────────────────────────────────────
    st.markdown(f"#### ⚡ Match Force: **{mf}**")
    col_mf, col_comp = st.columns(2)
    with col_mf:
        st.markdown(f"**{mf}** meaning:")
        for p in mf_phrases:
            st.markdown(f"- {p}")
    with col_comp:
        if comp and comp_phrases:
            st.markdown(f"**{comp}** *(complement — what MF lacks)*:")
            for p in comp_phrases:
                st.markdown(f"- {p}")
        else:
            st.caption("No complement card found.")

    st.divider()

    # ── Home card ─────────────────────────────────────────────────────
    st.markdown(f"#### 🏠 Home: **{home}**")
    colA, colB = st.columns([3, 2])
    with colA:
        st.markdown("**Meaning phrases** (highlighted = word match found):")
        for p in h_phrases:
            pw = {w.lower() for w in re.findall(r'\b[a-z]+\b', p)}
            if pw & h_sim_words:
                st.markdown(f"- 🟢 **{p}**")
            elif pw & h_opp_words:
                st.markdown(f"- 🟠 **{p}**")
            else:
                st.markdown(f"- {p}")

    with colB:
        st.markdown("**🟢 SIM** — shared words with MF meaning:")
        if h_sim:
            for tp, mp, ws in h_sim:
                st.markdown(f"- *\"{tp}\"* ↔ *\"{mp}\"*  `{ws}`")
        else:
            st.caption("No direct word overlaps with MF.")

        st.markdown("**🟠 OPP indicator** — shared words with MF complement:")
        if h_opp:
            for tp, cp, ws in h_opp:
                st.markdown(f"- *\"{tp}\"* ↔ *\"{cp}\"*  `{ws}`")
        else:
            st.caption("No word overlaps with complement.")

    st.divider()

    # ── Away card ─────────────────────────────────────────────────────
    st.markdown(f"#### ✈ Away: **{away}**")
    colC, colD = st.columns([3, 2])
    with colC:
        st.markdown("**Meaning phrases** (highlighted = word match found):")
        for p in a_phrases:
            pw = {w.lower() for w in re.findall(r'\b[a-z]+\b', p)}
            if pw & a_sim_words:
                st.markdown(f"- 🟢 **{p}**")
            elif pw & a_opp_words:
                st.markdown(f"- 🟠 **{p}**")
            else:
                st.markdown(f"- {p}")

    with colD:
        st.markdown("**🟢 SIM** — shared words with MF meaning:")
        if a_sim:
            for tp, mp, ws in a_sim:
                st.markdown(f"- *\"{tp}\"* ↔ *\"{mp}\"*  `{ws}`")
        else:
            st.caption("No direct word overlaps with MF.")

        st.markdown("**🟠 OPP indicator** — shared words with MF complement:")
        if a_opp:
            for tp, cp, ws in a_opp:
                st.markdown(f"- *\"{tp}\"* ↔ *\"{cp}\"*  `{ws}`")
        else:
            st.caption("No word overlaps with complement.")

    # ── Guidance note ─────────────────────────────────────────────────
    st.divider()
    st.info(
        "💡 **Important:** Many OPP relationships are *conceptual* rather than word-level — "
        "e.g. *'repressed intuition'* and *'let go and trust'* are exact opposites in meaning "
        "but share no common words. If you see few highlights above but the Reasoning section "
        "shows a strong OPP score, use your MF Lookup notes and the phrase lists above to "
        "manually identify the conceptual opposition. This visual is your starting point, "
        "not the final word."
    )


def render_mf_lookup(mf):
    """Display the MF Lookup entry for the current Match Force."""
    entry = get_mf_lookup_entry(mf)
    if entry is None:
        st.caption(f"No entry found in {MF_LOOKUP_FILE} for **{mf}**.")
        return

    mf_cell    = str(entry['Match Force'])
    winners    = str(entry['Winners'])
    themes_raw = str(entry['Themes'])

    name_part = _extract_lookup_card_name(mf_cell)
    desc_part = mf_cell[len(name_part):].lstrip(' -').strip()

    if desc_part and desc_part.lower() != 'nan':
        st.markdown(f"**Cards and themes that connect to {mf}** (your notes):")
        st.markdown(f"> {desc_part}")
        st.markdown("")

    if winners and winners.lower() != 'nan':
        st.markdown("**Recorded winners when this card is MF:**")
        st.markdown(f"> {winners}")
        st.markdown("")

    if themes_raw and themes_raw.lower() != 'nan':
        st.markdown("**MF themes:**")
        st.markdown(f"> {themes_raw}")


def render_full_analysis(result, home, away, mf):
    """Render the holistic analysis in clean, scannable format."""
    icons_type = {'OPP': '🟢', 'SIM': '🔵', 'NONE': '⚫'}

    # ── MF sense ─────────────────────────────────────────────────────────────
    st.markdown(f"**⚡ {mf}** — {result.get('mf_quality','')}")
    phrases = result.get('mf_key_phrases', [])
    if phrases:
        st.caption('  ·  '.join(f'*{p}*' for p in phrases))

    st.markdown("---")

    # ── Team cards ───────────────────────────────────────────────────────────
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
            if note:
                st.caption(note)

    st.markdown("---")

    # ── History ───────────────────────────────────────────────────────────────
    st.markdown("**📊 History**")
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

    # Draw signal — show how the MF draw prior affected the call
    draw_sig = result.get('draw_signal', '')
    if draw_sig and draw_sig.lower() not in ('no draw signal', 'none', ''):
        st.markdown(f"- 🎲 **Draw prior:** {draw_sig}")

    st.markdown("---")

    # ── Call ──────────────────────────────────────────────────────────────────
    call_icons = {'Home': '🟢', 'Away': '🔵', 'Draw': '🟡'}
    ci = call_icons.get(call, '⚪')
    st.markdown(f"**🎯 {ci} {call}**")
    reason = result.get('reason', '')
    if reason:
        st.markdown(f"*{reason}*")


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔮 Tarot Predictor V10")
st.caption(
    "Winner is the team whose card best balances the Match Force — "
    "through similarity (mirroring) or opposition (providing the deficit). "
    "When MF = a team card, the other card's relationship to the MF determines direction."
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

comp = get_complement(mf)
if comp and comp in meaning_map:
    st.caption(f"**MF complement** (what the Match Force creates a need for): _{comp}_ — _{meaning_map[comp][:90]}…_")

# ── MF draw-rate prior indicator ─────────────────────────────────────────────
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

    st.divider()

    # ── HOLISTIC ANALYSIS (primary path when API active) ────────────────────
    if full_analysis and '_error' not in full_analysis:
        render_full_analysis(full_analysis, home, away, mf)

        # Full match history in expander
        with st.expander("📊 Full match history"):
            ht = gather_history_text(
                home, away, mf, _history, meaning_map,
                get_mf_lookup_entry, base_card
            )
            st.text(ht)

    else:
        # ── FALLBACK: original reasoning + history blocks ────────────────────
        st.markdown("#### Reasoning")
        for i, line in enumerate(explanation):
            st.markdown(f"{'→' if i < 2 else '⟹'} {line}")

        if full_analysis and '_error' in full_analysis:
            st.warning(f"Holistic analysis failed: {full_analysis['_error']} — showing TF-IDF fallback.")

        if hist_notes:
            st.divider()
            st.markdown("#### 📊 Historical card-MF relationships")
            scored  = [n for n in hist_notes if not n.get('_display_only')]
            display = [n for n in hist_notes if n.get('_display_only')]
            for n in scored:
                st.markdown(f"**{n['label']}**")
                st.markdown(f"→ {n['summary']}")
                if n['matches']:
                    with st.expander("Previous matches"):
                        for ml in n['matches']: st.markdown(ml)
                st.markdown("")
            if hist_pred is not None and hist_weight >= 0.25:
                if hist_pred == pred:
                    st.success(f"✅ Historical data agrees with **{pred}**")
                else:
                    st.warning(f"⚠ Historical data points to **{hist_pred}** ({hist_weight:.0%}) — conflicts")
            if display:
                st.divider()
                st.markdown("#### 🔎 Co-occurrence (informational)")
                for n in display:
                    st.markdown(f"**{n['label']}**"); st.markdown(f"→ {n['summary']}")
                    if n['matches']:
                        with st.expander("Matches"):
                            for ml in n['matches']: st.markdown(ml)

        # Literal meaning analysis (fallback only)
        with st.expander("📖 Literal Meaning Analysis"):
            render_meaning_analysis(home, away, mf)

    # ── MF Lookup always available ───────────────────────────────────────────
    with st.expander(f"📋 MF Lookup — your notes on {mf}"):
        if _mf_lookup_df is not None:
            render_mf_lookup(mf)
        else:
            st.caption(f"Add {MF_LOOKUP_FILE} to the app folder to see your notes here.")

    # ── Card meanings ─────────────────────────────────────────────────────────
    with st.expander("Card meanings (full text)"):
        for label, c in [("Home", home), ("Match Force", mf), ("Away", away)]:
            st.markdown(f"**{label}: {c}**")
            st.write(meaning_map.get(c, '—'))
            cp = get_complement(c)
            if cp and cp in meaning_map:
                st.markdown(f"*Complement — {cp}:* {meaning_map[cp][:200]}")
            st.divider()

    analysis_count = sum(1 for k in st.session_state.rcache if k.startswith(ANALYSIS_PREFIX))
    pair_count     = sum(1 for k in st.session_state.rcache if not k.startswith(ANALYSIS_PREFIX))
    st.caption(f"Cache: {analysis_count} match analyses · {pair_count} pair resonances stored.")

st.divider()
with st.expander("Decision rules (in priority order)"):
    st.markdown("""
**Rule 0 — MF-identical pattern (highest priority):**
When the MF = a team card: compare MF with the OTHER team's card.
- OPP → other team wins | SIM → matching team wins | NONE → fall through

**Resonance rules (primary):**
1. Only one card balances MF → that team wins
2. Both balance, different types → OPP beats SIM (deficit energy is stronger)
3. Both balance, same type, score gap ≥ 0.15 → higher score wins
4. Both balance, same type, similar scores → card hierarchy (Major > King > Queen > Knight > Page > pip)
5. Equal everything → Draw  |  6. Neither balances → Draw

**Literal Meaning Analysis (new — inside expander after predicting):**
- Shows every keyword phrase from each card's meaning
- 🟢 = phrase shares a word with MF meaning (direct SIM signal)
- 🟠 = phrase shares a word with MF's complement meaning (OPP indicator)
- Many OPP relationships are conceptual (different words, opposite meanings) — check MF Lookup for those

**MF Lookup Reference (new — inside expander after predicting):**
- Shows your personal mf_lookup.xlsx notes for the current MF
- Lists which cards you have observed winning with this MF and the connecting themes

**Historical signal (secondary):**
Four rx-normalised checks. Agrees → confidence boosted. Conflicts → flagged for review.
""")
