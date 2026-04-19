"""
Tarot Predictor V9
Methodology: homeostatic balancing via semantic resonance.
Uses the Anthropic API to classify whether each card's energy is
similar to (SIM) or opposing/completing (OPP) the Match Force.
Results are cached to resonance_cache.json for fast re-use.

Requirements:
  pip install streamlit pandas openpyxl anthropic scikit-learn

Run:
  streamlit run app.py
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

os.environ["ANTHROPIC_API_KEY"] = ""   # ← paste key here temporarily

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
        d = cached
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
        resp  = client.messages.create(
            model=API_MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        raw   = resp.content[0].text.strip()
        raw   = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
        d     = json.loads(raw)
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
# Core index: card_team_record[(card, mf)]
#   = {'Win': n, 'Loss': n, 'Draw': n}
#
# Tracks: when `card` appeared as a team card (home OR away) in any match
# where `mf` was the Match Force, how did `card`'s team fare?
# Position (home/away) is irrelevant — only the card-MF relationship matters.
#
# Four lookups per prediction:
#   1. card_team_record[(home, mf)]
#      → home card's win rate when this MF was active
#
#   2. card_team_record[(away, mf)]
#      → away card's win rate when this MF was active
#
#   3. card_team_record[(mf, home)]
#      → when home card was the MF in past matches, how did the current MF
#        card perform as a team card? High win rate = these two energies
#        have a strong historical resonance that favours the home side.
#
#   4. card_team_record[(mf, away)]
#      → same, but for away card as the historical MF context.
#
# Plus exact triplet for direct matches.
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

    card_team_record = defaultdict(lambda: defaultdict(int))
    exact            = defaultdict(lambda: defaultdict(int))

    for _, r in _s1.iterrows():
        outcome = nm(r.get('Outcome', ''))
        if not outcome: continue
        h  = cl(r.get('Home', ''))
        a  = cl(r.get('Away', ''))
        mf = cl(r.get('Match Force', ''))

        exact[(h, a, mf)][outcome] += 1

        # Record win/loss/draw from each team card's perspective
        if outcome == 'Home':
            card_team_record[(h, mf)]['Win']  += 1
            card_team_record[(a, mf)]['Loss'] += 1
        elif outcome == 'Away':
            card_team_record[(h, mf)]['Loss'] += 1
            card_team_record[(a, mf)]['Win']  += 1
        else:
            card_team_record[(h, mf)]['Draw'] += 1
            card_team_record[(a, mf)]['Draw'] += 1

    return {
        'card_team_record': {k: dict(v) for k, v in card_team_record.items()},
        'exact':            {k: dict(v) for k, v in exact.items()},
    }

_history = build_history(sheet1, meaning_map)


def _wld_summary(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0, ''
    w = counts.get('Win', 0)
    l = counts.get('Loss', 0)
    d = counts.get('Draw', 0)
    wr = w / total
    return wr, total, f"W{w} L{l} D{d} ({wr:.0%} win rate, {total} match{'es' if total > 1 else ''})"


def historical_notes(home, away, mf):
    """
    Returns list of (label, summary, home_pts, away_pts).
    home_pts / away_pts contribute to the composite directional signal.
    """
    ctr   = _history['card_team_record']
    notes = []
    MIN   = 2

    # 1. Home card as team card under current MF
    rec = ctr.get((home, mf), {})
    if sum(rec.values()) >= MIN:
        wr, total, summary = _wld_summary(rec)
        # High win rate → supports Home; low win rate → supports Away
        hp = round(wr, 2)
        ap = round(1 - wr - rec.get('Draw', 0) / total, 2) if total else 0
        notes.append((
            f"**{home}** as team card when **{mf}** was MF",
            summary,
            hp, ap
        ))

    # 2. Away card as team card under current MF
    rec = ctr.get((away, mf), {})
    if sum(rec.values()) >= MIN:
        wr, total, summary = _wld_summary(rec)
        ap = round(wr, 2)
        hp = round(1 - wr - rec.get('Draw', 0) / total, 2) if total else 0
        notes.append((
            f"**{away}** as team card when **{mf}** was MF",
            summary,
            hp, ap
        ))

    # 3. Current MF card as team card when HOME card was the MF
    # High win rate here means: MF energy historically wins under home card's
    # context — these two energies resonate, boosting Home.
    rec = ctr.get((mf, home), {})
    if sum(rec.values()) >= MIN:
        wr, total, summary = _wld_summary(rec)
        hp = round(wr, 2)
        ap = round(1 - wr - rec.get('Draw', 0) / total, 2) if total else 0
        notes.append((
            f"**{mf}** (current MF) as team card when **{home}** was MF",
            summary + " — historical resonance between these energies",
            hp, ap
        ))

    # 4. Current MF card as team card when AWAY card was the MF
    rec = ctr.get((mf, away), {})
    if sum(rec.values()) >= MIN:
        wr, total, summary = _wld_summary(rec)
        ap = round(wr, 2)
        hp = round(1 - wr - rec.get('Draw', 0) / total, 2) if total else 0
        notes.append((
            f"**{mf}** (current MF) as team card when **{away}** was MF",
            summary + " — historical resonance between these energies",
            hp, ap
        ))

    # 5. Exact triplet
    rec = _history['exact'].get((home, away, mf), {})
    if sum(rec.values()) >= MIN:
        total = sum(rec.values())
        best  = max(rec, key=rec.get)
        parts = ', '.join(f"{o} {rec[o]}×" for o in ('Home','Away','Draw') if o in rec)
        hp    = rec.get('Home', 0) / total
        ap    = rec.get('Away', 0) / total
        notes.append((
            f"Exact triplet **{home}** vs **{away}** / MF **{mf}**",
            f"{parts} ({total} direct matches)",
            hp * 2, ap * 2   # double weight for exact match
        ))

    return notes


def historical_signal(home, away, mf):
    notes = historical_notes(home, away, mf)
    if not notes:
        return None, 0.0, []

    home_score = sum(n[2] for n in notes)
    away_score = sum(n[3] for n in notes)
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
# DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def decide(home, away, mf):
    hs, ht, hkw  = resonance(home, mf)
    as_, at, akw = resonance(away, mf)
    hr, ar       = card_rank(home), card_rank(away)
    lines        = []

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
            lines.append(
                f"**{home}** provides the deficit energy (OPP) while **{away}** only mirrors it (SIM) → **Home wins**"
            )
            return 'Home', min(0.68 + abs(hs - as_) * 0.3, 0.88), lines
        else:
            lines.append(
                f"**{away}** provides the deficit energy (OPP) while **{home}** only mirrors it (SIM) → **Away wins**"
            )
            return 'Away', min(0.68 + abs(as_ - hs) * 0.3, 0.88), lines

    diff = abs(hs - as_)
    if diff >= ASPECTS_DELTA:
        if hs > as_:
            lines.append(
                f"Both provide **{ht}** balance; **{home}** engages more strongly ({hs:.2f} vs {as_:.2f}) → **Home wins**"
            )
            return 'Home', min(0.62 + diff * 0.35, 0.82), lines
        else:
            lines.append(
                f"Both provide **{at}** balance; **{away}** engages more strongly ({as_:.2f} vs {hs:.2f}) → **Away wins**"
            )
            return 'Away', min(0.62 + diff * 0.35, 0.82), lines

    if hr != ar:
        if hr > ar:
            lines.append(
                f"Both provide equivalent **{ht}** balance; **{home}** ({RANK_LABEL[hr]}) outranks **{away}** ({RANK_LABEL[ar]}) → **Home wins**"
            )
            return 'Home', min(0.58 + (hr - ar) * 0.04, 0.74), lines
        else:
            lines.append(
                f"Both provide equivalent **{at}** balance; **{away}** ({RANK_LABEL[ar]}) outranks **{home}** ({RANK_LABEL[hr]}) → **Away wins**"
            )
            return 'Away', min(0.58 + (ar - hr) * 0.04, 0.74), lines

    lines.append(f"Both cards provide equivalent **{ht}** balance at equal authority ({RANK_LABEL[hr]}) → **Draw**")
    return 'Draw', 0.60, lines


def predict(home, away, mf):
    pred, conf, lines = decide(home, away, mf)

    hist_pred, hist_weight, hist_notes = historical_signal(home, away, mf)

    if hist_notes:
        lines.append('---')
        lines.append('**📊 Historical card-MF relationships:**')
        for label, summary, _, _ in hist_notes:
            lines.append(f"• {label}: {summary}")

    if hist_pred is not None and hist_weight >= 0.25:
        if hist_pred == pred:
            conf = conf * 0.65 + hist_weight * 0.35
            lines.append(f"✅ Historical data **agrees** with **{pred}** — confidence boosted")
        else:
            conf *= 0.72
            lines.append(
                f"⚠ Historical data points to **{hist_pred}** (weight {hist_weight:.0%}) "
                f"— conflicts with resonance prediction, flag for review"
            )

    return pred, round(min(conf, 0.95), 2), lines

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔮 Tarot Predictor V9")
st.caption(
    "Winner is the team whose card best balances the Match Force — "
    "through similarity (mirroring) or opposition (providing the deficit). "
    "See Rules below."
)

api_ok     = check_api()
cache_size = len(st.session_state.rcache)
if api_ok:
    st.success(f"✅ Semantic API active — {cache_size} resonance pair{'s' if cache_size != 1 else ''} cached to disk")
else:
    st.warning("⚠ ANTHROPIC_API_KEY not found — using TF-IDF keyword fallback. Set the key for full semantic analysis.")

st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    home = st.selectbox("🏠 Home Card", cards)
with c2:
    mf   = st.selectbox("⚡ Match Force", cards)
with c3:
    away = st.selectbox("✈ Away Card", cards)

comp = get_complement(mf)
if comp and comp in meaning_map:
    st.caption(
        f"**MF complement** (deficit — what the Match Force creates a need for): "
        f"_{comp}_ — _{meaning_map[comp][:90]}…_"
    )

if st.button("Predict", type="primary", use_container_width=True):
    with st.spinner("Analysing resonance…"):
        pred, conf, explanation = predict(home, away, mf)

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
        if line == '---':
            st.divider()
        else:
            prefix = "→" if i < 2 else "⟹"
            st.markdown(f"{prefix} {line}")

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
**Resonance rules (primary):**
1. Only one card balances the MF → that team wins
2. Both balance, different types → OPP/deficit beats SIM
3. Both balance, same type, clear score gap (≥ 0.15) → higher score wins
4. Both balance, same type, similar scores → card hierarchy (Major Arcana > King > Queen > Knight > Page > pip)
5. Equal everything → Draw
6. Neither balances → Draw

**Historical signal (secondary — four checks, all position-independent):**
- Has the home card appeared (as any team card) in matches where this MF was active? Win rate favours Home.
- Has the away card appeared (as any team card) in matches where this MF was active? Win rate favours Away.
- Has the current MF card appeared as a team card when the home card was the MF? Shows historical resonance between these energies.
- Has the current MF card appeared as a team card when the away card was the MF? Same for away.

If history agrees with the resonance prediction → confidence boosted.
If history conflicts → confidence reduced and flagged for review.
""")
