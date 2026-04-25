"""
Holistic match analysis engine — single comprehensive API call.
"""
import json, re, os

ANALYSIS_PREFIX = "ANALYSIS||"

def _akey(home, away, mf):
    return f"{ANALYSIS_PREFIX}{home}||{away}||{mf}"

def gather_history_text(home, away, mf, history, meaning_map, mf_lookup_fn, base_card_fn):
    bh = base_card_fn(home); ba = base_card_fn(away); bmf = base_card_fn(mf)
    rec = history.get('record', {}); mf_index = history.get('mf_index', {})
    exact = history.get('exact', {}); co_occur = history.get('co_occur', {})
    lines = []

    # ── MF DRAW PRIOR (empirical draw rate per MF) ───────────────────
    mf_matches_all = mf_index.get(bmf, [])
    if len(mf_matches_all) >= 5:
        draws_n  = sum(1 for m in mf_matches_all if m['outcome'] == 'Draw')
        rate     = draws_n / len(mf_matches_all)
        rate_pct = int(round(rate * 100))
        if rate >= 0.35:
            lines.append(f"** DRAW PRIOR for {mf}: {draws_n}/{len(mf_matches_all)} matches were draws ({rate_pct}%) -- HIGH DRAW MF.")
            lines.append(f"   This MF represents static/stuck/suspended energy that often does not drive a decisive outcome.")
            lines.append(f"   Lean toward DRAW if neither team shows a clearly stronger connection. Avoid forcing a winner from weak signal.")
        elif rate <= 0.15:
            lines.append(f"** DRAW PRIOR for {mf}: {draws_n}/{len(mf_matches_all)} matches were draws ({rate_pct}%) -- LOW DRAW MF.")
            lines.append(f"   This MF represents decisive/active energy. Draws are rare. Only call DRAW if the cards are truly indistinguishable.")
        else:
            lines.append(f"** DRAW PRIOR for {mf}: {draws_n}/{len(mf_matches_all)} matches were draws ({rate_pct}%) -- normal draw range.")
        lines.append("")

    mf_matches = mf_matches_all
    if mf_matches:
        winners = {}
        for m in mf_matches:
            o = m['outcome']
            w = m['home'] if o == 'Home' else (m['away'] if o == 'Away' else None)
            if w: winners[w] = winners.get(w, 0) + 1
        top = sorted(winners.items(), key=lambda x: -x[1])[:10]
        draws = sum(1 for m in mf_matches if m['outcome'] == 'Draw')
        lines.append(f"MF ({mf}) appeared in {len(mf_matches)} past matches as MF")
        lines.append(f"Cards that WON with this MF: {', '.join(f'{c}(x{n})' for c,n in top)}")
        if draws: lines.append(f"Draws when this was MF: {draws}")
    else:
        lines.append(f"MF ({mf}): no historical matches as MF")
    lines.append("")

    h_rec = rec.get((bh, bmf), {}); h_total = h_rec.get('Win',0)+h_rec.get('Loss',0)+h_rec.get('Draw',0)
    if h_total > 0:
        lines.append(f"Home ({home}) with MF ({mf}): W{h_rec['Win']} L{h_rec['Loss']} D{h_rec['Draw']} ({h_total} matches)")
        for m in h_rec.get('matches', [])[-5:]: lines.append(f"  {m['home']} vs {m['away']} / MF {m['mf']} -> {m['outcome']}")
    else:
        lines.append(f"Home ({home}) with MF ({mf}): no direct matches")
    lines.append("")

    a_rec = rec.get((ba, bmf), {}); a_total = a_rec.get('Win',0)+a_rec.get('Loss',0)+a_rec.get('Draw',0)
    if a_total > 0:
        lines.append(f"Away ({away}) with MF ({mf}): W{a_rec['Win']} L{a_rec['Loss']} D{a_rec['Draw']} ({a_total} matches)")
        for m in a_rec.get('matches', [])[-5:]: lines.append(f"  {m['home']} vs {m['away']} / MF {m['mf']} -> {m['outcome']}")
    else:
        lines.append(f"Away ({away}) with MF ({mf}): no direct matches")
    lines.append("")

    co = [m for m in co_occur.get(bh, []) if base_card_fn(m['home'])==ba or base_card_fn(m['away'])==ba]
    if co:
        lines.append(f"{home} + {away} appeared together in {len(co)} past matches:")
        for m in co[-6:]: lines.append(f"  {m['home']} vs {m['away']} / MF {m['mf']} -> {m['outcome']}")
    else:
        lines.append(f"{home} + {away} have not appeared together before")
    lines.append("")

    et = exact.get((bh, ba, bmf), {}); et_total = et.get('Home',0)+et.get('Away',0)+et.get('Draw',0)
    if et_total > 0:
        lines.append(f"Exact triplet ({home} vs {away} / MF {mf}): Home {et['Home']}x Away {et['Away']}x Draw {et['Draw']}x")
        for m in et.get('matches', []): lines.append(f"  {m['home']} vs {m['away']} / MF {m['mf']} -> {m['outcome']}")
    else:
        lines.append(f"Exact triplet ({home} vs {away} / MF {mf}): no previous match")

    entry = mf_lookup_fn(mf)
    if entry is not None:
        lines.append("")
        cell=str(entry.get('Match Force','')); wins=str(entry.get('Winners','')); themes=str(entry.get('Themes',''))
        lines.append(f"MF Lookup for {mf}:")
        if cell and cell!='nan': lines.append(f"  Context: {cell[:250]}")
        if wins and wins!='nan': lines.append(f"  Recorded winners: {wins[:350]}")
        if themes and themes!='nan': lines.append(f"  Themes: {themes[:200]}")

    return '\n'.join(lines)


_PROMPT = """You are a tarot analyst for a sports prediction system based on homeostatic balancing.

Determine which team card better balances the Match Force (MF). Use ONLY the meaning text provided, not general tarot knowledge. The winner is the card whose energy most specifically and directly balances the MF through SIM (mirrors the MF quality) or OPP (provides the exact opposite deficit the MF calls for).

PROCESS:

1. MF READING: Read the full MF meaning. Identify its domain (e.g. power/control, inner wisdom, communication, transformation, emotional state) and core quality. Pick 3-5 key phrases from the text.

2. FOR EACH TEAM CARD:
   a) Read the FULL meaning. Derive its core sense in 4-6 words.
   b) Does it operate in the SAME domain as the MF?
   c) SIM = same quality as MF. OPP = provides what MF lacks or embodies what MF overcomes.
   d) Find EXACT phrases from the provided meanings that make the connection. Quote them directly.
   e) TIED WORDS: if the same significant word appears in BOTH Home AND Away meanings in relation to the MF, it is NEUTRAL. Put it in tied_words. Look beyond it for what differentiates the cards.

3. HISTORY:
   a) DRAW PRIOR: check if the MF has a HIGH DRAW or LOW DRAW notation in the history block. This is empirical, not optional. High-draw MFs (>=35%) represent static/stuck energies that often do not produce decisive outcomes — weight DRAW seriously when signals are balanced. Low-draw MFs (<=15%) almost always pick a winner — avoid calling Draw unless cards are truly indistinguishable.
   b) What TYPE of card wins with this MF historically? What trait hierarchy does this reveal?
   c) Has Home or Away appeared with this MF before?
   d) Have Home and Away appeared together before? If they drew, what MF was it under? A draw under a different MF means they were equal on THAT MF's traits, not necessarily these.
   e) Use history to confirm, challenge, or break ties in meaning analysis.

4. CALL: 
   - OPP beats SIM (deficit energy is stronger).
   - Both same type: specificity wins.
   - History breaks genuine ties.
   - Apply the draw prior: if MF is HIGH DRAW and signals are balanced/weak, lean Draw. If MF is LOW DRAW, do not call Draw without strong evidence of equivalence.
   - Draw only if truly indistinguishable OR if the MF strongly favors draws and signals are balanced.

---
Match Force: {mf}
MF meaning: {mf_meaning}

MF Complement ({comp_card}):
{comp_meaning}

Home: {home}
Meaning: {home_meaning}

Away: {away}
Meaning: {away_meaning}

HISTORY:
{history_text}
---

Respond with ONLY valid JSON, no markdown:
{{"mf_domain":"domain","mf_quality":"core quality 6-8 words","mf_key_phrases":["phrase","phrase","phrase"],"home_sense":"4-6 words","home_domain_match":true,"home_type":"OPP","home_phrases":["exact home phrase","MF/complement phrase it connects to"],"home_tied_words":["neutral word also in away"],"home_note":"one sentence","away_sense":"4-6 words","away_domain_match":true,"away_type":"SIM","away_phrases":["exact away phrase","MF phrase it mirrors"],"away_tied_words":["neutral word also in home"],"away_note":"one sentence","history_mf_pattern":"one sentence on what type wins with this MF","history_direct":"one sentence on direct history or none","history_co":"one sentence on co-history or none","history_confirms":"Home","draw_signal":"how the MF draw prior affected this call (e.g. 'high draw MF but signals clearly favor home' or 'low draw MF so resisted draw despite balance' or 'no draw signal')","call":"Home","confidence":"high","reason":"one sentence citing specific phrases and/or history"}}"""


_CONF_MAP = {'high': 0.83, 'good': 0.70, 'moderate': 0.58, 'low': 0.48}


def analyse_match_full(home, away, mf, meaning_map, get_complement_fn,
                       history, mf_lookup_fn, base_card_fn,
                       rcache, save_cache_fn, api_model):
    key = _akey(home, away, mf)
    if key in rcache:
        cached = rcache[key]
        if isinstance(cached, dict) and 'call' in cached:
            return cached

    try:
        import anthropic
    except ImportError:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None

    comp = get_complement_fn(mf)
    comp_mean = meaning_map.get(comp, 'No complement found') if comp else 'No complement found'

    history_text = gather_history_text(
        home, away, mf, history, meaning_map, mf_lookup_fn, base_card_fn
    )

    prompt = _PROMPT.format(
        mf=mf, mf_meaning=meaning_map.get(mf, ''),
        comp_card=comp or 'none', comp_meaning=comp_mean,
        home=home, home_meaning=meaning_map.get(home, ''),
        away=away, away_meaning=meaning_map.get(away, ''),
        history_text=history_text,
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp   = client.messages.create(model=api_model, max_tokens=1200,
                                        messages=[{"role":"user","content":prompt}])
        raw    = resp.content[0].text.strip()
        raw    = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
        result = json.loads(raw)
    except Exception as e:
        return {'_error': str(e)}

    rcache[key] = result
    save_cache_fn(rcache)
    return result


def analysis_to_prediction(result):
    if not result or '_error' in result or 'call' not in result:
        return None, 0.5
    return result['call'], _CONF_MAP.get(result.get('confidence', 'moderate'), 0.58)
