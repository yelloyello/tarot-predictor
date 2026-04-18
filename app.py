"""
Tarot Predictor V9
Methodology: homeostatic balancing via semantic resonance.
Uses the Anthropic API to classify whether each card's energy is
similar to (SIM) or opposing/completing (OPP) the Match Force.
Results are cached to resonance_cache.json for fast re-use.

Requirements:
  pip install streamlit pandas openpyxl anthropic scikit-learn
  export ANTHROPIC_API_KEY="sk-ant-..."

Run:
  streamlit run tarot_predictor_v9.py
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

# Minimum resonance score to count a card as "balancing" the MF
BALANCE_MIN   = 0.35
# Minimum score gap to invoke "more aspects wins"
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
    sim  = _tsim(card, mf) * 5   # scale up
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
        # PASTE YOUR KEY ON THE LINE BELOW:
        os.environ["ANTHROPIC_API_KEY"] = "" 
        _API_READY = bool(os.environ.get('ANTHROPIC_API_KEY'))
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
# HISTORICAL MEMORY (exact triplet)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def build_history(_s1, _mm):
    exact = defaultdict(lambda: defaultdict(int))
    lc    = {k.lower(): k for k in _mm}
    def cl(x): return lc.get(str(x).strip().lower(), str(x).strip())
    def nm(x):
        x = str(x).strip()
        if x == '1': return 'Home'
        if x == '2': return 'Away'
        if x.upper().startswith('X'): return 'Draw'
        return None
    for _, r in _s1.iterrows():
        res = nm(r.get('Outcome',''))
        if not res: continue
        exact[(cl(r.get('Home','')), cl(r.get('Away','')), cl(r.get('Match Force','')))][res] += 1
    return exact

_history = build_history(sheet1, meaning_map)

# ─────────────────────────────────────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def decide(home, away, mf):
    hs, ht, hkw = resonance(home, mf)
    as_, at, akw = resonance(away, mf)
    hr, ar       = card_rank(home), card_rank(away)
    lines        = []

    def desc(card, score, rtype, kw):
        rs  = RANK_LABEL[card_rank(card)]
        if rtype is None:
            return f"**{card}** ({rs}) — no meaningful resonance with Match Force (strength {score:.2f})"
        verb = "mirrors the energy of" if rtype == 'SIM' else "provides what's missing in"
        kws  = ', '.join(f'*{k}*' for k in kw[:3]) if kw else 'thematic overlap'
        return f"**{card}** ({rs}) — {verb} **{mf}** via {kws}  [{rtype} · {score:.2f}]"

    lines.append(desc(home, hs, ht, hkw))
    lines.append(desc(away, as_, at, akw))

    # Rule 1: neither balances
    if ht is None and at is None:
        lines.append("Neither card achieves homeostatic balance with the Match Force → **Draw**")
        return 'Draw', 0.55, lines

    # Rule 2: only one balances
    if ht is not None and at is None:
        lines.append(f"Only **{home}** balances the Match Force — **{away}** provides no resonance → **Home wins**")
        return 'Home', min(0.70 + hs * 0.20, 0.90), lines
    if at is not None and ht is None:
        lines.append(f"Only **{away}** balances the Match Force — **{home}** provides no resonance → **Away wins**")
        return 'Away', min(0.70 + as_ * 0.20, 0.90), lines

    # Both balance — apply ordered rules

    # Rule 3: OPP beats SIM
    if ht != at:
        if ht == 'OPP':
            lines.append(
                f"**{home}** provides the deficit energy (OPP) while **{away}** only mirrors it (SIM) — "
                f"deficit takes precedence → **Home wins**"
            )
            return 'Home', min(0.68 + abs(hs - as_) * 0.3, 0.88), lines
        else:
            lines.append(
                f"**{away}** provides the deficit energy (OPP) while **{home}** only mirrors it (SIM) — "
                f"deficit takes precedence → **Away wins**"
            )
            return 'Away', min(0.68 + abs(as_ - hs) * 0.3, 0.88), lines

    # Rule 4: same type — more aspects (higher score) wins
    diff = abs(hs - as_)
    if diff >= ASPECTS_DELTA:
        if hs > as_:
            lines.append(
                f"Both provide **{ht}** balance; **{home}** engages more strongly "
                f"({hs:.2f} vs {as_:.2f}) — more matching aspects → **Home wins**"
            )
            return 'Home', min(0.62 + diff * 0.35, 0.82), lines
        else:
            lines.append(
                f"Both provide **{at}** balance; **{away}** engages more strongly "
                f"({as_:.2f} vs {hs:.2f}) — more matching aspects → **Away wins**"
            )
            return 'Away', min(0.62 + diff * 0.35, 0.82), lines

    # Rule 5: same type, similar score — card hierarchy
    if hr != ar:
        if hr > ar:
            lines.append(
                f"Both provide equivalent **{ht}** balance; "
                f"**{home}** ({RANK_LABEL[hr]}) outranks **{away}** ({RANK_LABEL[ar]}) → **Home wins**"
            )
            return 'Home', min(0.58 + (hr - ar) * 0.04, 0.74), lines
        else:
            lines.append(
                f"Both provide equivalent **{at}** balance; "
                f"**{away}** ({RANK_LABEL[ar]}) outranks **{home}** ({RANK_LABEL[hr]}) → **Away wins**"
            )
            return 'Away', min(0.58 + (ar - hr) * 0.04, 0.74), lines

    # Rule 6: equal everything — Draw
    lines.append(
        f"Both cards provide equivalent **{ht}** balance at equal authority "
        f"({RANK_LABEL[hr]}) → **Draw**"
    )
    return 'Draw', 0.60, lines


def predict(home, away, mf):
    pred, conf, lines = decide(home, away, mf)

    # Historical calibration
    hist  = _history.get((home, away, mf), {})
    total = sum(hist.values())
    if total >= 2:
        hp    = max(hist, key=hist.get)
        hr    = hist[hp] / total
        if hp == pred:
            conf = conf * 0.7 + hr * 0.3
            lines.append(f"Historical record: **{hp}** in {hist[hp]}/{total} prior appearances ({hr:.0%})")
        else:
            conf *= 0.75
            lines.append(f"⚠ Historical data ({total} matches) shows **{hp}** ({hr:.0%}) — flag for review")

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

# API status
api_ok     = check_api()
cache_size = len(st.session_state.rcache)
if api_ok:
    st.success(f"✅ Semantic API active — {cache_size} resonance pair{'s' if cache_size != 1 else ''} cached to disk")
else:
    st.warning("⚠ ANTHROPIC_API_KEY not found — using TF-IDF keyword fallback (less accurate). Set the key for full semantic analysis.")

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
1. **Only one card balances** → that team wins (clear)
2. **Both balance, different types** → OPP/deficit beats SIM (the card providing what's missing wins)
3. **Both balance, same type, clear score gap (≥ 0.15)** → higher score wins (more aspects engaged)
4. **Both balance, same type, similar scores** → Major Arcana > King > Queen > Knight > Page > pip
5. **Both balance, same type, same rank** → Draw
6. **Neither balances** → Draw
""")
