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

4. CALL — APPLY THESE RULES IN ORDER:

   STEP 1 — Establish energy classification first.
     Each team card gets one of: SIM (same energy as MF), OPP (opposite/deficit
     energy to MF), or NONE (no meaningful energy connection).

   STEP 2 — If neither card has SIM or OPP (both NONE) -> Draw.

   STEP 3 — Check for the duplicate-card exception FIRST (highest priority).
     A duplicate exists when MF base card == Home base card, OR
     MF base card == Away base card. (Ignore orientation — 'Death' and 'Death rx'
     are the same base.)
     In a duplicate case, hierarchy DOES NOT APPLY. The non-duplicate card decides:
       - Non-duplicate card has OPP -> non-duplicate team WINS
       - Non-duplicate card has SIM -> matching team (the duplicate side) WINS
       - Non-duplicate card has NONE -> Draw
     Skip the rest of the rules in duplicate cases.

   STEP 4 — Check the all-Majors exception.
     If Home, Away AND MF are ALL Majors AND one team is SIM and the other is OPP
     -> Draw. (When all are majors and one mirrors / one opposes, they cancel.)
     If all are majors but both have the same type (both SIM or both OPP), continue
     to step 5.

   STEP 5 — Apply the HIERARCHY RULE (Major > Court > pip).
     Major Arcana = Major. King/Queen/Knight/Page = Court. Numbered cards = pip.
     - If one card is Major and the other is not, MAJOR WINS regardless of whether
       it is SIM or OPP, unless the Major has NONE (no energy connection at all).
     - If neither is Major: if one is Court and the other is pip, COURT WINS the
       same way (regardless of SIM/OPP), unless court is NONE.
     - If both cards are at the same hierarchy level (both pips, both courts,
       both majors), then OPP beats SIM. If both same type, lean to specificity
       and history; if no clear winner -> Draw.

   STEP 6 — If after the above steps a card with NONE would "win" by hierarchy
   over a card with SIM or OPP, reverse it: a card with no energy connection
   cannot beat a card that does have one. The card with the actual energy
   connection wins instead.

   STEP 7 — Draw priors as a final calibration:
     If you've concluded a winner but the MF is HIGH DRAW (>=35%) AND the energy
     signals are balanced/weak, lean Draw. If LOW DRAW (<=15%), avoid Draw unless
     truly indistinguishable.

NOTE on hierarchy override: Yes, this means a Major SIM beats a pip OPP. The
historical rule "OPP beats SIM" only applies when cards are at the SAME hierarchy
level. The user has empirically validated that hierarchy takes precedence.

---
Match Force: {mf}  [hierarchy: {mf_rank}]
MF meaning: {mf_meaning}

MF Complement ({comp_card}):
{comp_meaning}

Home: {home}  [hierarchy: {home_rank}]
Meaning: {home_meaning}

Away: {away}  [hierarchy: {away_rank}]
Meaning: {away_meaning}

DUPLICATE CHECK (auto-detected): {duplicate_note}
ALL-MAJORS CHECK (auto-detected): {all_majors_note}

HISTORY:
{history_text}
---

Respond with ONLY valid JSON, no markdown:
{{"mf_domain":"domain","mf_quality":"core quality 6-8 words","mf_key_phrases":["phrase","phrase","phrase"],"home_sense":"4-6 words","home_domain_match":true,"home_type":"OPP","home_phrases":["exact home phrase","MF/complement phrase it connects to"],"home_tied_words":["neutral word also in away"],"home_note":"one sentence","away_sense":"4-6 words","away_domain_match":true,"away_type":"SIM","away_phrases":["exact away phrase","MF phrase it mirrors"],"away_tied_words":["neutral word also in home"],"away_note":"one sentence","history_mf_pattern":"one sentence on what type wins with this MF","history_direct":"one sentence on direct history or none","history_co":"one sentence on co-history or none","history_confirms":"Home","draw_signal":"how the MF draw prior affected this call (e.g. 'high draw MF but signals clearly favor home' or 'low draw MF so resisted draw despite balance' or 'no draw signal')","call":"Home","confidence":"high","reason":"one sentence citing specific phrases and/or history"}}"""


_CONF_MAP = {'high': 0.83, 'good': 0.70, 'moderate': 0.58, 'low': 0.48}


_MAJOR_BASES = {
    'chariot','death','devil','empress','emperor','fool',
    'hanged man','heirophant','hermit','high priestess','judgement',
    'justice','lovers','magician','moon','star','strength','sun',
    'temperance','tower','wheel','world'
}

def _rank_label(card):
    """Return 'Major', 'Court' or 'pip' for the AI prompt."""
    c = card.lower().replace(' rx', '').replace('rx', '').strip()
    if c in _MAJOR_BASES: return 'Major'
    if c.startswith('k') and not c.startswith('kn'): return 'Court (King)'
    if c.startswith('q'): return 'Court (Queen)'
    if c.startswith('kn'): return 'Court (Knight)'
    if c.startswith('p') and len(c) == 2: return 'Court (Page)'
    return 'pip'


def analyse_match_full(home, away, mf, meaning_map, get_complement_fn,
                       history, mf_lookup_fn, base_card_fn,
                       rcache, save_cache_fn, api_model):
    key = _akey(home, away, mf)
    cached_raw = rcache.get(key) if isinstance(rcache.get(key), dict) else None
    # Only use cache if it has a 'rules_v2' marker (post-rule-layer) — otherwise
    # we want to re-run so the new rule layer applies.
    if cached_raw and 'call' in cached_raw and cached_raw.get('rules_v2'):
        return cached_raw

    # If we have a cached AI response (without rules_v2), reuse the AI's
    # classifications and re-run the rule layer instead of paying for a new API call
    if cached_raw and 'home_type' in cached_raw and 'away_type' in cached_raw:
        result = dict(cached_raw)  # copy
    else:
        try:
            import anthropic
        except ImportError:
            return None

        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return None

        result = None  # will be filled by API call below

    comp = get_complement_fn(mf)
    comp_mean = meaning_map.get(comp, 'No complement found') if comp else 'No complement found'

    history_text = gather_history_text(
        home, away, mf, history, meaning_map, mf_lookup_fn, base_card_fn
    )

    # Pre-compute structural signals for the prompt
    h_rank   = _rank_label(home)
    a_rank   = _rank_label(away)
    mf_rank  = _rank_label(mf)
    bh = base_card_fn(home).lower()
    ba = base_card_fn(away).lower()
    bmf = base_card_fn(mf).lower()

    if bh == bmf and ba == bmf:
        dup_note = (f"All three cards share the same base ({bh}). "
                    f"Apply the duplicate rule across both sides — energetic "
                    f"resolution from the side that has any meaningful difference "
                    f"in orientation, otherwise Draw.")
    elif bh == bmf:
        dup_note = (f"DUPLICATE: Home base ({bh}) == MF base. "
                    f"Hierarchy does NOT apply. Away card decides: "
                    f"Away OPP -> Away wins; Away SIM -> Home wins; Away NONE -> Draw.")
    elif ba == bmf:
        dup_note = (f"DUPLICATE: Away base ({ba}) == MF base. "
                    f"Hierarchy does NOT apply. Home card decides: "
                    f"Home OPP -> Home wins; Home SIM -> Away wins; Home NONE -> Draw.")
    else:
        dup_note = "no duplicate — apply hierarchy rules normally"

    if h_rank == 'Major' and a_rank == 'Major' and mf_rank == 'Major':
        all_maj_note = ("ALL THREE ARE MAJORS. If one team is SIM and the other is OPP, "
                        "result is Draw (the all-majors exception). If both same type, "
                        "specificity decides; if neither has clear connection -> Draw.")
    else:
        all_maj_note = "not all majors — exception does not apply"

    prompt = _PROMPT.format(
        mf=mf, mf_meaning=meaning_map.get(mf, ''),
        comp_card=comp or 'none', comp_meaning=comp_mean,
        home=home, home_meaning=meaning_map.get(home, ''),
        away=away, away_meaning=meaning_map.get(away, ''),
        history_text=history_text,
        home_rank=h_rank, away_rank=a_rank, mf_rank=mf_rank,
        duplicate_note=dup_note, all_majors_note=all_maj_note,
    )

    if result is None:
        # No cached AI classifications — call the API
        try:
            client = anthropic.Anthropic(api_key=api_key)
            resp   = client.messages.create(model=api_model, max_tokens=1200,
                                            messages=[{"role":"user","content":prompt}])
            raw    = resp.content[0].text.strip()
            raw    = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
            result = json.loads(raw)
        except Exception as e:
            return {'_error': str(e)}

    # ── DETERMINISTIC RULE LAYER ─────────────────────────────────────────
    # Apply the encoded ruleset over the AI's SIM/OPP/NONE classifications.
    # The AI is good at reading meanings; the rules engine is good at applying
    # the hierarchy + duplicate + all-majors logic deterministically.
    rule_call, rule_reason, rule_path = _apply_rules(
        home_type=result.get('home_type', 'NONE'),
        away_type=result.get('away_type', 'NONE'),
        home_rank=h_rank, away_rank=a_rank, mf_rank=mf_rank,
        home=home, away=away, mf=mf,
        bh=bh, ba=ba, bmf=bmf,
    )
    if rule_call is not None:
        result['_ai_call']     = result.get('call')   # preserve AI's original call
        result['_ai_reason']   = result.get('reason')
        result['call']         = rule_call
        result['reason']       = rule_reason
        result['rule_path']    = rule_path
    result['rules_v2'] = True

    rcache[key] = result
    save_cache_fn(rcache)
    return result


def _apply_rules(home_type, away_type, home_rank, away_rank, mf_rank,
                 home, away, mf, bh, ba, bmf):
    """
    Apply the encoded prediction rules deterministically over the AI's SIM/OPP/NONE
    classifications. Returns (call, reason, rule_path) or (None, None, None) if
    rules don't apply.
    """
    h, a = home_type, away_type
    h_lvl = 'Major' if home_rank == 'Major' else ('Court' if home_rank.startswith('Court') else 'pip')
    a_lvl = 'Major' if away_rank == 'Major' else ('Court' if away_rank.startswith('Court') else 'pip')
    LVL_ORDER = {'Major': 3, 'Court': 2, 'pip': 1}

    # STEP 2: Both NONE -> Draw
    if h == 'NONE' and a == 'NONE':
        return ('Draw',
                'Neither card has a meaningful energy connection to the MF.',
                'both-NONE -> Draw')

    # STEP 3: Duplicate exception (highest priority over hierarchy)
    if bh == bmf and ba == bmf:
        # All three same — degenerate edge case; defer to AI
        return (None, None, None)
    if bh == bmf:
        # Home == MF, hierarchy does not apply, away decides
        if a == 'OPP':
            return ('Away',
                    f'Duplicate: Home={mf}=MF, hierarchy suspended. Away ({away}) '
                    f'has OPP energy -> Away wins.',
                    'duplicate (Home=MF) + Away OPP')
        if a == 'SIM':
            return ('Home',
                    f'Duplicate: Home={mf}=MF, hierarchy suspended. Away ({away}) '
                    f'has SIM energy, reinforcing the matching team -> Home wins.',
                    'duplicate (Home=MF) + Away SIM')
        return ('Draw',
                f'Duplicate: Home={mf}=MF, hierarchy suspended. Away ({away}) has '
                f'no energy connection -> Draw.',
                'duplicate (Home=MF) + Away NONE')
    if ba == bmf:
        if h == 'OPP':
            return ('Home',
                    f'Duplicate: Away={mf}=MF, hierarchy suspended. Home ({home}) '
                    f'has OPP energy -> Home wins.',
                    'duplicate (Away=MF) + Home OPP')
        if h == 'SIM':
            return ('Away',
                    f'Duplicate: Away={mf}=MF, hierarchy suspended. Home ({home}) '
                    f'has SIM energy, reinforcing the matching team -> Away wins.',
                    'duplicate (Away=MF) + Home SIM')
        return ('Draw',
                f'Duplicate: Away={mf}=MF, hierarchy suspended. Home ({home}) has '
                f'no energy connection -> Draw.',
                'duplicate (Away=MF) + Home NONE')

    # STEP 4: All-majors exception
    if h_lvl == 'Major' and a_lvl == 'Major' and mf_rank == 'Major':
        if {h, a} == {'SIM', 'OPP'}:
            return ('Draw',
                    'All three cards are Majors with one SIM and one OPP — they '
                    'cancel under the all-majors exception -> Draw.',
                    'all-Majors + mixed SIM/OPP -> Draw')
        # Otherwise fall through

    # STEP 5: Hierarchy override
    # If only one card has an energy connection, that card wins regardless of rank.
    if h != 'NONE' and a == 'NONE':
        return ('Home',
                f'Only Home ({home}) has an energy connection to MF ({h}).',
                'only-Home-engages')
    if a != 'NONE' and h == 'NONE':
        return ('Away',
                f'Only Away ({away}) has an energy connection to MF ({a}).',
                'only-Away-engages')

    # Both have energy connections. Apply hierarchy if levels differ.
    if h_lvl != a_lvl:
        if LVL_ORDER[h_lvl] > LVL_ORDER[a_lvl]:
            return ('Home',
                    f'{home} ({h_lvl}, {h}) outranks {away} ({a_lvl}, {a}). '
                    f'Hierarchy: {h_lvl} > {a_lvl}.',
                    f'hierarchy: {h_lvl} > {a_lvl}')
        else:
            return ('Away',
                    f'{away} ({a_lvl}, {a}) outranks {home} ({h_lvl}, {h}). '
                    f'Hierarchy: {a_lvl} > {h_lvl}.',
                    f'hierarchy: {a_lvl} > {h_lvl}')

    # Both same hierarchy level: OPP beats SIM
    if h == 'OPP' and a == 'SIM':
        return ('Home',
                f'Same hierarchy level ({h_lvl}). Home is OPP, Away is SIM. '
                f'OPP beats SIM.',
                f'same-level ({h_lvl}) + OPP beats SIM')
    if a == 'OPP' and h == 'SIM':
        return ('Away',
                f'Same hierarchy level ({h_lvl}). Away is OPP, Home is SIM. '
                f'OPP beats SIM.',
                f'same-level ({h_lvl}) + OPP beats SIM')

    # Both same level AND same type — defer to AI's specificity / history call
    return (None, None, None)


def analysis_to_prediction(result):
    if not result or '_error' in result or 'call' not in result:
        return None, 0.5
    return result['call'], _CONF_MAP.get(result.get('confidence', 'moderate'), 0.58)
