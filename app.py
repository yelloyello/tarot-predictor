"""
Tarot Predictor V10
Methodology: homeostatic balancing via semantic resonance.
New in V10: MF-identical pattern — when MF == a team card, the other team's
card is compared to the MF; OPP → other team wins, SIM → matching team wins.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os, re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Tarot Predictor V9", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CACHE_FILE    = "resonance_cache.json"
DATA_FILE     = "data.xlsx"
API_MODEL     = "claude-sonnet-4-5"

os.environ["ANTHROPIC_API_KEY"] = "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"   # ← paste key here temporarily

BALANCE_MIN   = 0.35
ASPECTS_DELTA = 0.15

MAJOR_BASES = {
    'chariot','death','devil','empress','emperor','fool',
    'hanged man','heirophant','hermit','high priestess','judgement',
    'justice','lovers','magician','moon','star','strength','sun',
    'temperance','tower','wheel','world'
}
RANK_LABEL = {6:'Major Arcana', 5:'King', 4:'Queen', 3:'Knight', 2:'Page', 1:'pip'}

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
# RX NORMALISATION
# Strip trailing rx/rx suffix so "Hermit rx" and "Hermit" match each other
# in historical lookups. We store variants so both directions are checked.
# ─────────────────────────────────────────────────────────────────────────────
def base_card(card):
    """Return the upright base of a card, stripping rx suffix."""
    c = card.strip()
    if c.lower().endswith(' rx'):
        return c[:-3].strip()
    if c.lower().endswith('rx'):
        return c[:-2].strip()
    return c

def card_variants(card):
    """Return a set of card name variants: the card itself + its rx/base counterpart."""
    variants = {card}
    base = base_card(card)
    if base != card:
        variants.add(base)
    else:
        # card is upright — also add rx variants if they exist in meaning_map
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
# API RESONANCE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
_API_READY = None

def check_api():
    global _API_READY
    if _API_READY is not None:
        return _API_READY
    try:
        import anthropic as _a   # noqa
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            key = os.environ.get("ANTHROPIC_API_KEY", "")
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
        _API_READY = bool(key)
    except ImportError:
        _API_READY = False
    return _API_READY

_PROMPT = """You are analysing two tarot cards for a homeostatic balancing prediction system.

The Match Force sets the energetic context of the match. A team card can balance it in one of two ways:
- **SIM** (Similarity): the card's energy mirrors or echoes the Match Force — they share the same quality.
- **OPP** (Opposition / Deficit): the card provides what the Match Force lacks — the equal and opposite force that completes or resolves the Match Force energy. Example: happiness resolves sadness; clarity resolves confusion; strength resolves fear; success resolves failure.
- **NONE**: no meaningful energetic relationship.

Match Force card: {mf}
Match Force meaning: {mf_meaning}

Team card: {card}
Team card meaning: {card_meaning}

Respond ONLY with a JSON object, no other text or markdown:
{{"type": "SIM", "score": 0.75, "aspects": ["aspect 1", "aspect 2", "aspect 3"]}}

Rules:
- type: "SIM", "OPP", or "NONE"
- score: 0.0–1.0 (strength of resonance — how clearly and strongly the card engages with the Match Force)
- aspects: 2–4 short noun phrases (under 5 words each) naming the specific themes that create the match or contrast
"""

def api_resonance(card, mf):
    key     = f"{card}||{mf}"
    cached  = st.session_state.rcache.get(key)
    if cached:
        d  = cached
        sc = float(d.get('score', 0.0))
        rt = d.get('type', 'NONE')
        return sc, (None if rt == 'NONE' or sc < BALANCE_MIN else rt), d.get('aspects', [])

    import anthropic
    client = anthropic.Anthropic()
    prompt = _PROMPT.format(
        mf=mf, mf_meaning=meaning_map.get(mf, '')[:400],
        card=card, card_meaning=meaning_map.get(card, '')[:400],
    )
    try:
        resp = client.messages.create(
            model=API_MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        raw  = resp.content[0].text.strip()
        raw  = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
        d    = json.loads(raw)
    except Exception:
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
#
# card_team_record[(card_base, mf_base)] = {
#   'Win': n, 'Loss': n, 'Draw': n,
#   'matches': [{'home': h, 'away': a, 'mf': mf, 'outcome': o}, ...]
# }
#
# All lookups are done on BASE cards (rx stripped) so that upright and
# reversed versions of the same card are treated as the same energy.
# The actual card names used in each match are stored for display.
#
# Four lookups per prediction:
#   1. base(home) as team card, base(mf) as MF context
#   2. base(away) as team card, base(mf) as MF context
#   3. base(mf)   as team card, base(home) as MF context  [energy resonance]
#   4. base(mf)   as team card, base(away) as MF context  [energy resonance]
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

    # key = (base_card, base_mf)
    record   = defaultdict(lambda: {'Win': 0, 'Loss': 0, 'Draw': 0, 'matches': []})
    exact    = defaultdict(lambda: {'Home': 0, 'Away': 0, 'Draw': 0, 'matches': []})
    # key = base_mf -> outcomes where MF appeared but did NOT match either team card
    no_match = defaultdict(lambda: {'Home': 0, 'Away': 0, 'Draw': 0, 'matches': []})

    for _, r in _s1.iterrows():
        outcome = nm(r.get('Outcome', ''))
        if not outcome: continue
        h   = cl(r.get('Home', ''))
        a   = cl(r.get('Away', ''))
        mf  = cl(r.get('Match Force', ''))
        bh  = base_card(h)
        ba  = base_card(a)
        bmf = base_card(mf)

        match_ctx = {'home': h, 'away': a, 'mf': mf, 'outcome': outcome}

        # exact triplet (base keys)
        exact[(bh, ba, bmf)][outcome] += 1
        exact[(bh, ba, bmf)]['matches'].append(match_ctx)

        # home card's team result under this MF
        if outcome == 'Home':
            record[(bh, bmf)]['Win']  += 1
            record[(ba, bmf)]['Loss'] += 1
        elif outcome == 'Away':
            record[(bh, bmf)]['Loss'] += 1
            record[(ba, bmf)]['Win']  += 1
        else:
            record[(bh, bmf)]['Draw'] += 1
            record[(ba, bmf)]['Draw'] += 1

        record[(bh, bmf)]['matches'].append(match_ctx)
        record[(ba, bmf)]['matches'].append(match_ctx)

        # no-match record: MF appeared but was NOT identical to either team card (rx-normalised)
        if bmf != bh and bmf != ba:
            no_match[bmf][outcome] += 1
            no_match[bmf]['matches'].append(match_ctx)

    # co_occur index: base_card -> list of every match that card appeared in
    # (any position: home, away, or mf)
    co_occur = defaultdict(list)
    for _, r in _s1.iterrows():
        outcome = nm(r.get('Outcome', ''))
        if not outcome: continue
        h   = cl(r.get('Home', ''))
        a   = cl(r.get('Away', ''))
        mf  = cl(r.get('Match Force', ''))
        match_ctx = {'home': h, 'away': a, 'mf': mf, 'outcome': outcome}
        for card in {base_card(h), base_card(a), base_card(mf)}:
            co_occur[card].append(match_ctx)

    # Freeze to plain dicts for caching
    def freeze_record(d):
        return {k: {'Win': v['Win'], 'Loss': v['Loss'], 'Draw': v['Draw'],
                    'matches': v['matches']} for k, v in d.items()}
    def freeze_exact(d):
        return {k: {'Home': v['Home'], 'Away': v['Away'], 'Draw': v['Draw'],
                    'matches': v['matches']} for k, v in d.items()}
    def freeze_no_match(d):
        return {k: {'Home': v['Home'], 'Away': v['Away'], 'Draw': v['Draw'],
                    'matches': v['matches']} for k, v in d.items()}

    return {
        'record':   freeze_record(record),
        'exact':    freeze_exact(exact),
        'no_match': freeze_no_match(no_match),
        'co_occur': dict(co_occur),
    }

_history = build_history(sheet1, meaning_map)


def _wld_summary(rec):
    w = rec.get('Win', 0); l = rec.get('Loss', 0); d = rec.get('Draw', 0)
    total = w + l + d
    if total == 0: return 0.0, 0, ''
    wr = w / total
    return wr, total, f"W{w} L{l} D{d} ({wr:.0%} win rate, {total} match{'es' if total > 1 else ''})"


def _format_matches(matches):
    """Format match context lines for display."""
    lines = []
    for m in matches:
        lines.append(f"  `{m['home']}` vs `{m['away']}` / MF `{m['mf']}` → **{m['outcome']}**")
    return lines


def historical_notes(home, away, mf):
    """
    Returns list of dicts:
      label, summary, match_lines, home_pts, away_pts
    """
    rec    = _history['record']
    exact  = _history['exact']
    notes  = []
    MIN    = 2
    bh, ba, bmf = base_card(home), base_card(away), base_card(mf)

    # 1. Home card as team card under current MF (rx-normalised)
    r = rec.get((bh, bmf), {})
    total = r.get('Win',0) + r.get('Loss',0) + r.get('Draw',0)
    if total >= MIN:
        wr, _, summary = _wld_summary(r)
        hp = round(wr, 2)
        lp = round(r.get('Loss',0)/total, 2)
        notes.append({
            'label':   f"**{home}** as team card when **{mf}** was MF",
            'summary': summary,
            'matches': _format_matches(r.get('matches', [])),
            'home_pts': hp, 'away_pts': lp
        })

    # 2. Away card as team card under current MF
    r = rec.get((ba, bmf), {})
    total = r.get('Win',0) + r.get('Loss',0) + r.get('Draw',0)
    if total >= MIN:
        wr, _, summary = _wld_summary(r)
        ap = round(wr, 2)
        lp = round(r.get('Loss',0)/total, 2)
        notes.append({
            'label':   f"**{away}** as team card when **{mf}** was MF",
            'summary': summary,
            'matches': _format_matches(r.get('matches', [])),
            'home_pts': lp, 'away_pts': ap
        })

    # 3. Current MF as team card when HOME card was the MF
    r = rec.get((bmf, bh), {})
    total = r.get('Win',0) + r.get('Loss',0) + r.get('Draw',0)
    if total >= MIN:
        wr, _, summary = _wld_summary(r)
        hp = round(wr, 2)
        lp = round(r.get('Loss',0)/total, 2)
        notes.append({
            'label':   f"**{mf}** (current MF energy) as team card when **{home}** was MF",
            'summary': summary + " — these energies have historical resonance",
            'matches': _format_matches(r.get('matches', [])),
            'home_pts': hp, 'away_pts': lp
        })

    # 4. Current MF as team card when AWAY card was the MF
    r = rec.get((bmf, ba), {})
    total = r.get('Win',0) + r.get('Loss',0) + r.get('Draw',0)
    if total >= MIN:
        wr, _, summary = _wld_summary(r)
        ap = round(wr, 2)
        lp = round(r.get('Loss',0)/total, 2)
        notes.append({
            'label':   f"**{mf}** (current MF energy) as team card when **{away}** was MF",
            'summary': summary + " — these energies have historical resonance",
            'matches': _format_matches(r.get('matches', [])),
            'home_pts': lp, 'away_pts': ap
        })

    # 5. Exact triplet (rx-normalised)
    r = exact.get((bh, ba, bmf), {})
    total = r.get('Home',0) + r.get('Away',0) + r.get('Draw',0)
    if total >= 1:   # show even single exact matches
        parts = ', '.join(f"{o} {r[o]}×" for o in ('Home','Away','Draw') if r.get(o,0) > 0)
        best  = max(('Home','Away','Draw'), key=lambda o: r.get(o,0))
        hp    = r.get('Home',0) / total * 2   # double weight
        ap    = r.get('Away',0) / total * 2
        notes.append({
            'label':   f"Exact triplet **{home}** vs **{away}** / MF **{mf}**",
            'summary': f"{parts} ({total} direct match{'es' if total>1 else ''})",
            'matches': _format_matches(r.get('matches', [])),
            'home_pts': hp, 'away_pts': ap
        })

    # 6. MF no-match background: how did this MF perform when it did NOT
    #    match either team card? Only shown when checks 1 & 2 found nothing
    #    (i.e. no direct team-card history) — acts as a fallback signal.
    has_direct = any(
        'as team card when' in n['label'] for n in notes
    )
    if not has_direct:
        nm_rec = _history['no_match'].get(bmf, {})
        nm_total = nm_rec.get('Home', 0) + nm_rec.get('Away', 0) + nm_rec.get('Draw', 0)
        if nm_total >= 2:
            home_n = nm_rec.get('Home', 0)
            away_n = nm_rec.get('Away', 0)
            draw_n = nm_rec.get('Draw', 0)
            dominant = max(('Home', 'Away', 'Draw'), key=lambda o: nm_rec.get(o, 0))
            dominant_pct = nm_rec.get(dominant, 0) / nm_total
            parts = f"Home {home_n}× Away {away_n}× Draw {draw_n}×"
            # Weight: nudge toward the dominant outcome; draws count as 0.5 each side
            hp = (home_n + draw_n * 0.5) / nm_total
            ap = (away_n + draw_n * 0.5) / nm_total
            notes.append({
                'label':   f"🔍 **{mf}** no-match background ({nm_total} matches where MF ≠ either team card)",
                'summary': (
                    f"{parts} — when **{mf}** appeared but didn't match either team, "
                    f"**{dominant}** won {dominant_pct:.0%} of the time. "
                    f"No direct team-card history found; use this as a weak directional signal."
                ),
                'matches': _format_matches(nm_rec.get('matches', [])[-10:]),  # cap at 10
                'home_pts': round(hp * 0.5, 2),   # half-weight vs direct checks
                'away_pts': round(ap * 0.5, 2),
            })

    return notes


def _co_occur_note(home, away, mf):
    """
    Broad co-occurrence search (display-only, no scoring weight).
    Finds every past match where the MF energy (or its rx flip) AND at least
    one team card energy (or their rx flips) appeared together in any position
    (home, away, or mf column).  Returns a note dict or None.
    """
    co = _history.get('co_occur', {})

    bh  = base_card(home)
    ba  = base_card(away)
    bmf = base_card(mf)

    # Candidate bases to look for: MF + both team cards (all rx-normalised)
    mf_bases   = {bmf}
    home_bases = {bh}
    away_bases = {ba}

    # Gather all matches that contain the MF energy in any position
    mf_matches = co.get(bmf, [])

    # Filter: row must also contain home OR away energy in any position
    def cards_in_row(m):
        return {base_card(m['home']), base_card(m['away']), base_card(m['mf'])}

    hits_home, hits_away, hits_both = [], [], []
    seen = set()
    for m in mf_matches:
        key = (m['home'], m['away'], m['mf'])
        if key in seen:
            continue
        seen.add(key)
        row_bases = cards_in_row(m)
        has_h = bh in row_bases
        has_a = ba in row_bases
        if has_h and has_a:
            hits_both.append(m)
        elif has_h:
            hits_home.append(m)
        elif has_a:
            hits_away.append(m)

    all_hits = hits_both + hits_home + hits_away
    if not all_hits:
        return None

    # Build outcome tally across all hits
    tally = {'Home': 0, 'Away': 0, 'Draw': 0}
    for m in all_hits:
        tally[m['outcome']] = tally.get(m['outcome'], 0) + 1
    total = sum(tally.values())

    parts = []
    if hits_both:
        parts.append(f"{len(hits_both)} with **both** team cards present")
    if hits_home:
        parts.append(f"{len(hits_home)} with **{home}** only")
    if hits_away:
        parts.append(f"{len(hits_away)} with **{away}** only")

    tally_str = f"Home {tally['Home']}× Away {tally['Away']}× Draw {tally['Draw']}×"
    dominant  = max(tally, key=tally.get)
    dom_pct   = tally[dominant] / total if total else 0

    formatted = _format_matches(all_hits[:15])  # cap display at 15

    return {
        'label':   (
            f"🔎 Co-occurrence: **{mf}** (MF energy) + team cards appeared together "
            f"in {total} past match{'es' if total > 1 else ''}"
        ),
        'summary': (
            f"{'; '.join(parts)}. Outcomes: {tally_str} — "
            f"**{dominant}** dominant ({dom_pct:.0%}). "
            f"Display only — not factored into confidence score."
        ),
        'matches':   formatted,
        'home_pts':  0.0,   # informational only, no scoring weight
        'away_pts':  0.0,
        '_display_only': True,
    }


def historical_signal(home, away, mf):
    notes = historical_notes(home, away, mf)

    # Always append the broad co-occurrence note at the bottom (display only)
    co_note = _co_occur_note(home, away, mf)
    if co_note:
        notes.append(co_note)

    if not notes:
        return None, 0.0, notes

    home_score = sum(n['home_pts'] for n in notes)
    away_score = sum(n['away_pts'] for n in notes)
    total      = home_score + away_score

    if total == 0 or abs(home_score - away_score) < 0.1:
        return None, 0.0, notes

    if home_score > away_score:
        weight = min((home_score - away_score) / max(total, 1), 1.0)
        return 'Home', round(weight, 2), notes
    else:
        weight = min((away_score - home_score) / max(total, 1), 1.0)
        return 'Away', round(weight, 2), notes

# ─────────────────────────────────────────────────────────────────────────────
# MF-IDENTICAL PATTERN
#
# When the MF card is exactly the same card as one of the team cards, the
# normal resonance logic doesn't apply. Instead we compare the MF meaning
# with the OTHER team's card:
#
#   • OPP (opposite meaning) → MF aligns with the opposite energy → OTHER
#     team wins.  (The identical card and MF cancel/mirror each other so
#     perfectly that the energy flows toward what resolves/completes them.)
#
#   • SIM (shared meaning) → MF aligns with itself → the MATCHING team wins.
#     (The identical card reinforces the MF; shared resonance confirms it.)
#
#   • NONE → fall through to normal decide() logic as a tie-break.
#
# Empirically validated on 19 same-card rows in Sheet1:
#   11 OPP cases → other team won   ✓
#    8 SIM cases → matching team won ✓
# ─────────────────────────────────────────────────────────────────────────────
def mf_identical_check(home, away, mf):
    """
    Returns (prediction, confidence, explanation_lines) if the MF-identical
    pattern applies, otherwise returns (None, None, None).
    """
    home_match = (home.strip().lower() == mf.strip().lower())
    away_match = (away.strip().lower() == mf.strip().lower())

    if not home_match and not away_match:
        return None, None, None

    if home_match:
        matching_team  = 'Home'
        matching_card  = home
        other_card     = away
    else:
        matching_team  = 'Away'
        matching_card  = away
        other_card     = home

    # Compare MF meaning with the OTHER team's card
    other_score, other_type, other_aspects = resonance(other_card, mf)

    kws = ', '.join(f'*{k}*' for k in other_aspects[:3]) if other_aspects else 'thematic relationship'

    lines = [
        f"⚠️ **MF identical to {matching_team} card** ({matching_card}) — "
        f"applying MF-identical pattern.",
    ]

    if other_type == 'OPP':
        # Other card is energetically opposite → MF flows toward opposite energy
        other_team = 'Away' if matching_team == 'Home' else 'Home'
        other_name = away if matching_team == 'Home' else home
        lines.append(
            f"**{other_name}** ({other_team}) carries the **opposite energy** to the MF "
            f"via {kws} [OPP · {other_score:.2f}] — MF resonates with the opposing force."
        )
        lines.append(
            f"Identical-card rule: OPP → **{other_team} wins** "
            f"(the MF energy flows toward what resolves/completes it, not its mirror)."
        )
        conf = min(0.70 + other_score * 0.18, 0.88)
        return other_team, round(conf, 2), lines

    elif other_type == 'SIM':
        # Other card shares meaning → MF reinforces the matching team
        lines.append(
            f"**{other_card}** shares similar energy with the MF "
            f"via {kws} [SIM · {other_score:.2f}] — but the MF is identical to **{matching_card}**, "
            f"so it reinforces the matching team rather than bridging to the other."
        )
        lines.append(
            f"Identical-card rule: SIM → **{matching_team} wins** "
            f"(shared resonance confirms the MF aligns with its own card)."
        )
        conf = min(0.68 + other_score * 0.15, 0.83)
        return matching_team, round(conf, 2), lines

    else:
        # NONE — no clear relationship with other card; fall through
        lines.append(
            f"**{other_card}** has no meaningful energetic relationship with the MF "
            f"[NONE · {other_score:.2f}] — falling through to standard resonance logic."
        )
        return None, None, lines   # caller will append these lines as context


# ─────────────────────────────────────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def decide(home, away, mf):
    # ── MF-identical pattern (highest priority) ──────────────────────────────
    id_pred, id_conf, id_lines = mf_identical_check(home, away, mf)
    if id_pred is not None:
        return id_pred, id_conf, id_lines

    hs, ht, hkw  = resonance(home, mf)
    as_, at, akw = resonance(away, mf)
    hr, ar       = card_rank(home), card_rank(away)
    lines        = []

    # If the identical check fired but returned NONE (no relationship found),
    # prepend its context lines so the analyst can see the flag.
    if id_lines:
        lines.extend(id_lines)

    def desc(card, score, rtype, kw):
        rs   = RANK_LABEL[card_rank(card)]
        if rtype is None:
            return f"**{card}** ({rs}) — no meaningful resonance with Match Force (strength {score:.2f})"
        verb = "mirrors the energy of" if rtype == 'SIM' else "provides what's missing in"
        kws  = ', '.join(f'*{k}*' for k in kw[:3]) if kw else 'thematic overlap'
        return f"**{card}** ({rs}) — {verb} **{mf}** via {kws}  [{rtype} · {score:.2f}]"

    lines.append(desc(home, hs, ht, hkw))
    lines.append(desc(away, as_, at, akw))

    if ht is None and at is None:
        lines.append("Neither card achieves homeostatic balance with the Match Force → **Draw**")
        return 'Draw', 0.55, lines

    if ht is not None and at is None:
        lines.append(f"Only **{home}** balances the Match Force → **Home wins**")
        return 'Home', min(0.70 + hs * 0.20, 0.90), lines
    if at is not None and ht is None:
        lines.append(f"Only **{away}** balances the Match Force → **Away wins**")
        return 'Away', min(0.70 + as_ * 0.20, 0.90), lines

    if ht != at:
        if ht == 'OPP':
            lines.append(f"**{home}** provides deficit energy (OPP), **{away}** only mirrors (SIM) → **Home wins**")
            return 'Home', min(0.68 + abs(hs - as_) * 0.3, 0.88), lines
        else:
            lines.append(f"**{away}** provides deficit energy (OPP), **{home}** only mirrors (SIM) → **Away wins**")
            return 'Away', min(0.68 + abs(as_ - hs) * 0.3, 0.88), lines

    diff = abs(hs - as_)
    if diff >= ASPECTS_DELTA:
        if hs > as_:
            lines.append(f"Both {ht}; **{home}** engages more strongly ({hs:.2f} vs {as_:.2f}) → **Home wins**")
            return 'Home', min(0.62 + diff * 0.35, 0.82), lines
        else:
            lines.append(f"Both {at}; **{away}** engages more strongly ({as_:.2f} vs {hs:.2f}) → **Away wins**")
            return 'Away', min(0.62 + diff * 0.35, 0.82), lines

    if hr != ar:
        if hr > ar:
            lines.append(f"Equal balance; **{home}** ({RANK_LABEL[hr]}) outranks **{away}** ({RANK_LABEL[ar]}) → **Home wins**")
            return 'Home', min(0.58 + (hr - ar) * 0.04, 0.74), lines
        else:
            lines.append(f"Equal balance; **{away}** ({RANK_LABEL[ar]}) outranks **{home}** ({RANK_LABEL[hr]}) → **Away wins**")
            return 'Away', min(0.58 + (ar - hr) * 0.04, 0.74), lines

    lines.append(f"Both cards provide equivalent **{ht}** balance at equal authority ({RANK_LABEL[hr]}) → **Draw**")
    return 'Draw', 0.60, lines


def predict(home, away, mf):
    pred, conf, lines = decide(home, away, mf)
    hist_pred, hist_weight, hist_notes = historical_signal(home, away, mf)

    if hist_pred is not None and hist_weight >= 0.25:
        if hist_pred == pred:
            conf = conf * 0.65 + hist_weight * 0.35
        else:
            conf *= 0.72

    return pred, round(min(conf, 0.95), 2), lines, hist_pred, hist_weight, hist_notes

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔮 Tarot Predictor V10")
st.caption(
    "Winner is the team whose card best balances the Match Force — "
    "through similarity (mirroring) or opposition (providing the deficit). "
    "When MF = a team card, the other card's relationship to the MF determines direction. "
    "See Rules below."
)

api_ok     = check_api()
cache_size = len(st.session_state.rcache)
if api_ok:
    st.success(f"✅ Semantic API active — {cache_size} resonance pair{'s' if cache_size != 1 else ''} cached to disk")
else:
    st.warning("⚠ ANTHROPIC_API_KEY not found — using TF-IDF keyword fallback. Set the key for full semantic analysis.")

st.divider()

# Tab-key navigation between the three card dropdowns
st.components.v1.html("""
<script>
(function() {
  function hookTab() {
    var boxes = Array.from(
      window.parent.document.querySelectorAll(
        'div[data-testid="stSelectbox"] input'
      )
    ).slice(0, 3);   // only the three card selectors
    if (boxes.length < 3) { setTimeout(hookTab, 300); return; }
    boxes.forEach(function(box, idx) {
      box.addEventListener("keydown", function(e) {
        if (e.key !== "Tab") return;
        e.preventDefault();
        var next = e.shiftKey
          ? (idx - 1 + boxes.length) % boxes.length
          : (idx + 1) % boxes.length;
        boxes[next].focus();
      });
    });
  }
  hookTab();
})();
</script>
""", height=0)

c1, c2, c3 = st.columns(3)
with c1:
    home = st.selectbox("🏠 Home Card", cards)
with c2:
    away = st.selectbox("✈ Away Card", cards)
with c3:
    mf   = st.selectbox("⚡ Match Force", cards)

comp = get_complement(mf)
if comp and comp in meaning_map:
    st.caption(
        f"**MF complement** (deficit — what the Match Force creates a need for): "
        f"_{comp}_ — _{meaning_map[comp][:90]}…_"
    )

if st.button("Predict", type="primary", use_container_width=True):
    with st.spinner("Analysing resonance…"):
        pred, conf, explanation, hist_pred, hist_weight, hist_notes = predict(home, away, mf)

    icons = {'Home':'🟢', 'Away':'🔵', 'Draw':'🟡'}
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## {icons.get(pred,'⚪')} Prediction: **{pred}**")
    with col2:
        st.metric("Confidence", f"{conf:.0%}")

    st.progress(conf)

    if conf >= 0.80:   st.caption("High confidence — a clear rule applies.")
    elif conf >= 0.65: st.caption("Good confidence — methodology points clearly.")
    elif conf >= 0.55: st.caption("Moderate confidence — hierarchy or close scores involved.")
    else:              st.caption("Lower confidence — treat as a pointer; exercise judgement.")

    st.divider()
    st.markdown("#### Reasoning")
    for i, line in enumerate(explanation):
        prefix = "→" if i < 2 else "⟹"
        st.markdown(f"{prefix} {line}")

    # ── Historical signal block ──────────────────────────────────────────────
    if hist_notes:
        st.divider()
        st.markdown("#### 📊 Historical card-MF relationships")
        st.caption("Upright and reversed versions of a card are treated as the same energy.")

        scored_notes   = [n for n in hist_notes if not n.get('_display_only')]
        display_notes  = [n for n in hist_notes if n.get('_display_only')]

        for n in scored_notes:
            st.markdown(f"**{n['label']}**")
            st.markdown(f"→ {n['summary']}")
            if n['matches']:
                with st.expander("Previous matches"):
                    for ml in n['matches']:
                        st.markdown(ml)
            st.markdown("")

        if hist_pred is not None and hist_weight >= 0.25:
            if hist_pred == pred:
                st.success(f"✅ Historical data agrees with **{pred}** — confidence boosted")
            else:
                st.warning(
                    f"⚠ Historical data points to **{hist_pred}** (weight {hist_weight:.0%}) "
                    f"— conflicts with resonance prediction, flag for review"
                )
        elif scored_notes:
            st.info("Historical data found but signal is too weak or balanced to shift prediction.")

        # Co-occurrence notes — display only, always shown at the bottom
        if display_notes:
            st.divider()
            st.markdown("#### 🔎 Broad co-occurrence (informational)")
            st.caption(
                "Any past match where the MF energy and either team card energy "
                "appeared together in any position (home, away, or MF). "
                "Not factored into confidence — use for pattern research."
            )
            for n in display_notes:
                st.markdown(f"**{n['label']}**")
                st.markdown(f"→ {n['summary']}")
                if n['matches']:
                    with st.expander("Matching past matches"):
                        for ml in n['matches']:
                            st.markdown(ml)
                st.markdown("")

    with st.expander("Card meanings"):
        for label, c in [("Home", home), ("Match Force", mf), ("Away", away)]:
            st.markdown(f"**{label}: {c}**")
            st.write(meaning_map.get(c, '—')[:280])
            cp = get_complement(c)
            if cp and cp in meaning_map:
                st.markdown(f"*Complement — {cp}:* {meaning_map[cp][:150]}")
            st.divider()

    st.caption(
        f"Cache: {len(st.session_state.rcache)} pairs stored. "
        "Each unique (card, MF) pair is queried once and saved permanently to resonance_cache.json."
    )

st.divider()
with st.expander("Decision rules (in priority order)"):
    st.markdown("""
**Rule 0 — MF-identical pattern (highest priority):**
When the MF card is the same card as one of the team cards, compare the MF meaning with the OTHER team's card:
- OPP (opposite meaning) → other team wins (MF energy flows toward what resolves/completes it, not its mirror)
- SIM (shared meaning) → matching team wins (MF reinforces its own card)
- NONE → fall through to standard resonance logic below
*Empirically validated: 11 OPP cases → other team won ✓, 8 SIM cases → matching team won ✓*

**Resonance rules (primary):**
1. Only one card balances the MF → that team wins
2. Both balance, different types → OPP/deficit beats SIM
3. Both balance, same type, clear score gap (≥ 0.15) → higher score wins
4. Both balance, same type, similar scores → card hierarchy (Major Arcana > King > Queen > Knight > Page > pip)
5. Equal everything → Draw
6. Neither balances → Draw

**Historical signal (secondary):**
Four position-independent checks, all rx-normalised (upright and reversed treated as same energy):
- Has the home card appeared as a team card when this MF was active?
- Has the away card appeared as a team card when this MF was active?
- Has the current MF card appeared as a team card when the home card was the MF? (energy resonance)
- Has the current MF card appeared as a team card when the away card was the MF? (energy resonance)
- **No-match background** (fallback — only shown when no direct team-card history exists): how did this MF perform historically when it did NOT match either team card? Provides a weak directional signal when data is sparse.

Each check shows the win/loss/draw record and the specific past matches for context.
If history agrees with resonance → confidence boosted. If it conflicts → flagged for review.
""")
