"""
Holistic match analysis engine — v3 ruleset.

CORE METHODOLOGY (new):
A team card "controls" the Match Force (MF) by either:
  1. DESTABILISING (OPP)  — possessing an equal, opposing energy / characteristic
  2. MIRRORING (SIM)      — possessing an equal mirror of the MF's energy / characteristic

PRIMARY RULES
  • If one card destabilises (OPP) and the other mirrors (SIM)  → destabiliser wins
  • If only one card has any connection (other is NONE)         → that team wins
  • If both cards mirror (both SIM)                              → Draw
  • If neither card has any connection (both NONE)               → Draw

HIERARCHY (Major > King > Queen > Knight > Page > pip)
  Used only as a tie-breaker / for the exception below.

HIERARCHY EXCEPTION
  When one card MIRRORS the MF at a HIGHER rank, and the other DESTABILISES
  but at a LOWER rank → Draw. (The destabilising force is too weak to overpower
  the higher-rank mirror, but the mirror can't actively control either.)

PIP-PIP EXCEPTION
  When both team cards are pips, AND one is SIM and one is OPP → Draw.
  (Pips lack the rank-weight to decisively control the MF.)

SHARED-THEME DRAW
  When all three cards share a single dominant energetic theme (e.g. all about
  "lack of clarity", or all about "control") → Draw. Neither team has unique
  purchase on the MF energy.

DUPLICATE-CARD RULES (highest priority — override hierarchy entirely)
  Triggered when a team card has the SAME base card as the MF (e.g. Home=Death,
  MF=Death rx — they are the same base).
  • Remaining card is SIM  → duplicate-side team wins  (MF "matches with" its duplicate)
  • Remaining card is OPP  → remaining team wins        (opposing force takes control)
  • Remaining card is NONE → duplicate-side team wins   (duplicate is the only connection)

SPECIFICITY DIRECTIVE
  The AI must compare at the TRAIT level — not at broad themes. Each card has
  many distinct traits. Home and Away may connect to DIFFERENT aspects of the MF.
  Even lesser-used, niche traits count: a card with "domestic goddess" is an exact
  destabiliser to a card with "neglected home", even if those are minor traits in
  each card's full meaning.
"""
import json, re, os

ANALYSIS_PREFIX = "ANALYSIS||"
RULES_VERSION = "rules_v5"   # bump when prompt or rule layer changes meaningfully


def _extract_json(text):
    """
    Pull a JSON object out of a model response. Robust to:
      - markdown fences (```json ... ``` or just ``` ... ```)
      - leading/trailing prose around the JSON
      - extra whitespace
    Returns the parsed dict, or None if no valid JSON object can be recovered.
    """
    if not text:
        return None
    t = text.strip()
    # Strip surrounding markdown fences if present
    if t.startswith('```'):
        t = re.sub(r'^```[a-zA-Z0-9_-]*\n?', '', t)
        if t.endswith('```'):
            t = t[:-3]
        t = t.strip()
    # Try a direct parse first
    try:
        return json.loads(t)
    except Exception:
        pass
    # Fall back: find the first {...} object and parse it (greedy to last brace)
    first = t.find('{')
    last  = t.rfind('}')
    if first != -1 and last != -1 and last > first:
        candidate = t[first:last+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def _akey(home, away, mf):
    return f"{ANALYSIS_PREFIX}{home}||{away}||{mf}"


# ════════════════════════════════════════════════════════════════════════
# HISTORY TEXT — passed to the AI as context only (rule layer ignores it)
# ════════════════════════════════════════════════════════════════════════
def gather_history_text(home, away, mf, history, meaning_map, mf_lookup_fn, base_card_fn):
    bh = base_card_fn(home); ba = base_card_fn(away); bmf = base_card_fn(mf)
    rec = history.get('record', {}); mf_index = history.get('mf_index', {})
    exact = history.get('exact', {}); co_occur = history.get('co_occur', {})
    lines = []

    # ── MF DRAW PRIOR ────────────────────────────────────────────────
    mf_matches_all = mf_index.get(bmf, [])
    if len(mf_matches_all) >= 5:
        draws_n  = sum(1 for m in mf_matches_all if m['outcome'] == 'Draw')
        rate     = draws_n / len(mf_matches_all)
        rate_pct = int(round(rate * 100))
        if rate >= 0.35:
            lines.append(f"** DRAW PRIOR for {mf}: {draws_n}/{len(mf_matches_all)} matches were draws ({rate_pct}%) -- HIGH DRAW MF.")
            lines.append(f"   This MF often does not produce a decisive outcome.")
        elif rate <= 0.15:
            lines.append(f"** DRAW PRIOR for {mf}: {draws_n}/{len(mf_matches_all)} matches were draws ({rate_pct}%) -- LOW DRAW MF.")
            lines.append(f"   This MF tends to produce decisive outcomes.")
        else:
            lines.append(f"** DRAW PRIOR for {mf}: {draws_n}/{len(mf_matches_all)} matches were draws ({rate_pct}%) -- normal range.")
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


# ════════════════════════════════════════════════════════════════════════
# THE PROMPT — emphasises trait-level matching, not broad-energy generalising
# ════════════════════════════════════════════════════════════════════════
_PROMPT = """You are a tarot analyst for a sports prediction system. Determine how each team card relates to the Match Force (MF) at the TRAIT level.

═══════════════════════════════════════════════════════════════════════
CORE METHODOLOGY
═══════════════════════════════════════════════════════════════════════
A team card "controls" the MF in one of two ways:
  • SIM (mirror)      — the team card possesses an EQUAL, MIRRORING energy or characteristic to the MF
  • OPP (destabilise) — the team card possesses an EQUAL but OPPOSING energy or characteristic to the MF
Otherwise: NONE (no meaningful connection).

═══════════════════════════════════════════════════════════════════════
CRITICAL: COMPARE AT THE TRAIT LEVEL, NOT THE BROAD-THEME LEVEL
═══════════════════════════════════════════════════════════════════════
Each card has MANY distinct traits. The Chariot is NOT just "self-control" —
it is self-control AND ambition AND confidence AND momentum AND triumph AND
direction AND hard work AND succor (assistance in distress) AND travel.

Do NOT pick one main theme of the MF and try to find that theme in both team
cards. Instead:

  1. Enumerate 6-10 DISTINCT trait clusters from the MF meaning text.
  2. Enumerate 6-10 DISTINCT trait clusters from each team card.
  3. For EACH MF trait, scan EVERY trait of the team card for a SIM (direct
     mirror) or OPP (direct opposite) match.
  4. Home and Away may connect to DIFFERENT MF traits — that is normal and
     expected. E.g. Home might mirror the MF's "self-control" aspect, while
     Away opposes the MF's "confidence" aspect. Both connections are valid.

PAY SPECIAL ATTENTION TO LESSER-USED, NICHE TRAITS.
A card with the niche trait "domestic goddess" is an EXACT destabiliser to a
card with the niche trait "neglected home" — even though those traits are
minor within each card's overall meaning. Such lesser-used trait matches have
historically been decisive. Do NOT overlook them by focusing only on each
card's "headline" themes.

A SIM/OPP connection requires SPECIFIC trait language — a literal mirror or
literal opposite of a phrase in the meaning text. Vague resonance does not
qualify. Quote the EXACT phrase from each side that makes the connection.

═══════════════════════════════════════════════════════════════════════
SHARED-THEME CHECK (all three cards)
═══════════════════════════════════════════════════════════════════════
Check whether ALL THREE cards (Home, Away, MF) share a single dominant
energetic theme (e.g. all relate to "lack of clarity", all relate to
"control struggles", all relate to "endings"). If yes — flag this. When
all three cards crowd the same thematic space, no team has unique
purchase on the MF energy, and the rule layer will resolve to Draw.

═══════════════════════════════════════════════════════════════════════
SAME-TYPE PRINCIPLE — CLASSIFY DELIBERATELY (CRITICAL)
═══════════════════════════════════════════════════════════════════════
The rule layer treats both-OPP as Draw and both-SIM as Draw — regardless
of rank, regardless of whether the cards engage the same MF trait. So
your SIM/OPP/NONE classification directly determines this outcome.

Be DELIBERATE about classification. A team card's classification reflects
its DOMINANT relationship to the MF — not the first/most-obvious one.

  WORKED EXAMPLE (do NOT repeat this mistake)
  MF: 8Wrx — multiple traits including "stopping of momentum",
    "unfinished business / things stuck", "lethargy", "impulsiveness",
    "domestic disputes", etc.
  6S has the following candidate connections:
    SIM: "pulling back, creating distance" ≈ MF "stopping"
    SIM: "exhaustion, lethargy, weary" ≈ MF "sluggishness"
    OPP: "moving on, putting past behind" ↔ MF "unfinished business / stuck"
    OPP: "regaining balance and control" ↔ MF "out of control"
    OPP: "stability, escaping to calmer waters" ↔ MF "chaos, panic, drama"
    OPP: "healing, recovery" ↔ MF "deteriorating"
  → The OPP relationships are MORE NUMEROUS and AS LITERAL as the SIM
    ones. Classifying as SIM here would be wrong; the card's dominant
    posture toward this MF is destabilising — OPP.

WHEN A CARD HAS BOTH SIM AND OPP CANDIDATES:
- Count the meaningful connections of each type
- Weigh how literal/specific each is
- Pick the type that has the BROADER and MORE LITERAL set of engagements
- Tie-break in favour of OPP (destabilising is the more active form of control)

═══════════════════════════════════════════════════════════════════════
CONTESTED-TRAIT BACKUP CHECK (lower priority)
═══════════════════════════════════════════════════════════════════════
If, after applying the same-type principle above, you DO classify one team
as SIM and the other as OPP, still walk the MF traits and check: does any
single MF trait have meaningful connections from BOTH teams (one SIM, one
OPP via different bullets)? List those in contested_mf_traits.

This is a safety net for cases where SIM/OPP genuinely apply but the cards
also share thematic engagement with one specific MF trait — the rule layer
will treat that as a deadlock and Draw.

═══════════════════════════════════════════════════════════════════════
OUTPUT — SIM / OPP / NONE FOR EACH TEAM CARD
═══════════════════════════════════════════════════════════════════════
You do NOT decide the final Home/Away/Draw call. The rule layer applies
all the rules (hierarchy, pip-pip, duplicate, same-type, shared-theme,
contested-trait) deterministically over your classifications.

LIST ALL MEANINGFUL CONNECTIONS in home_connections and away_connections,
not just the single strongest. If 6S has FOUR meaningful trait connections
to MF aspects, list ALL FOUR — even if one is the headline. The classifier
needs to see the full picture to decide between SIM and OPP.

═══════════════════════════════════════════════════════════════════════
INPUT
═══════════════════════════════════════════════════════════════════════
Match Force: {mf}  [rank: {mf_rank}]
MF meaning (read in full, including all lesser-used phrases):
{mf_meaning}

MF complement ({comp_card}) — provided as opposite-vocabulary reference only:
{comp_meaning}

Home: {home}  [rank: {home_rank}]
Home meaning (read in full):
{home_meaning}

Away: {away}  [rank: {away_rank}]
Away meaning (read in full):
{away_meaning}

DUPLICATE STATUS (auto-detected): {duplicate_note}

HISTORY (informational only — do not use to determine SIM/OPP):
{history_text}

═══════════════════════════════════════════════════════════════════════
RESPOND WITH ONLY VALID JSON, NO MARKDOWN FENCES
═══════════════════════════════════════════════════════════════════════
{{
  "mf_traits": ["distinct trait cluster 1", "trait 2", "trait 3", "trait 4", "trait 5", "trait 6"],
  "home_traits": ["home trait 1", "home trait 2", "home trait 3", "home trait 4", "home trait 5", "home trait 6"],
  "away_traits": ["away trait 1", "away trait 2", "away trait 3", "away trait 4", "away trait 5", "away trait 6"],

  "home_type": "SIM | OPP | NONE",
  "home_connections": [
    {{"mf_trait": "exact MF trait", "home_trait": "exact home trait", "kind": "SIM or OPP", "note": "why this is a literal mirror or literal opposite"}}
  ],
  "home_note": "one-sentence summary of Home's strongest connection",

  "away_type": "SIM | OPP | NONE",
  "away_connections": [
    {{"mf_trait": "exact MF trait", "away_trait": "exact away trait", "kind": "SIM or OPP", "note": "why this is a literal mirror or literal opposite"}}
  ],
  "away_note": "one-sentence summary of Away's strongest connection",

  "shared_theme": false,
  "shared_theme_note": "if true, name the theme all three cards share; otherwise empty string",

  "contested_mf_traits": ["MF trait that BOTH home and away meaningfully engage with — list each contested trait separately, empty array if none"],
  "contested_note": "if any contested traits, briefly explain how each side engages the same trait; otherwise empty string",

  "history_signal": "one sentence on whether history meaningfully supports a side or is neutral",

  "confidence": "high | good | moderate | low",
  "reasoning": "2-3 sentences justifying the SIM/OPP/NONE classifications, citing specific trait language"
}}"""


_CONF_MAP = {'high': 0.83, 'good': 0.70, 'moderate': 0.58, 'low': 0.48}


# ════════════════════════════════════════════════════════════════════════
# RANK LABELS — full hierarchy (Major > King > Queen > Knight > Page > pip)
# ════════════════════════════════════════════════════════════════════════
_MAJOR_BASES = {
    'chariot','death','devil','empress','emperor','fool',
    'hanged man','heirophant','hermit','high priestess','judgement',
    'justice','lovers','magician','moon','star','strength','sun',
    'temperance','tower','wheel','world',
    # also tolerate the alternate spelling
    'hierophant',
}

RANK_ORDER = {'Major': 6, 'King': 5, 'Queen': 4, 'Knight': 3, 'Page': 2, 'pip': 1}


def _rank_label(card):
    """Return one of 'Major', 'King', 'Queen', 'Knight', 'Page', 'pip'."""
    c = card.lower().replace(' rx', '').replace('rx', '').strip()
    if c in _MAJOR_BASES: return 'Major'
    if c.startswith('kn'): return 'Knight'
    if c.startswith('k'):  return 'King'
    if c.startswith('q'):  return 'Queen'
    if c.startswith('p') and len(c) == 2: return 'Page'
    return 'pip'


# ════════════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ════════════════════════════════════════════════════════════════════════
def analyse_match_full(home, away, mf, meaning_map, get_complement_fn,
                       history, mf_lookup_fn, base_card_fn,
                       rcache, save_cache_fn, api_model):
    key = _akey(home, away, mf)
    cached_raw = rcache.get(key) if isinstance(rcache.get(key), dict) else None

    # Only honour cache entries built under the current rule version.
    if cached_raw and 'call' in cached_raw and cached_raw.get(RULES_VERSION):
        return cached_raw

    # If we have an AI-only cache from a prior version (it has home_type/away_type
    # set), we *can* reuse those classifications and just re-run the rule layer.
    # But because the prompt changed materially in v3, prefer fresh AI calls.
    # We re-use only when v3 AI output is already there.
    reuse_ai = (cached_raw
                and 'home_type' in cached_raw
                and 'away_type' in cached_raw
                and 'home_connections' in cached_raw)  # v3 schema marker

    if reuse_ai:
        result = dict(cached_raw)
    else:
        try:
            import anthropic
        except ImportError:
            return None
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return None
        result = None  # filled by API call below

    comp = get_complement_fn(mf)
    comp_mean = meaning_map.get(comp, 'No complement found') if comp else 'No complement found'

    history_text = gather_history_text(
        home, away, mf, history, meaning_map, mf_lookup_fn, base_card_fn
    )

    # Pre-compute structural signals
    h_rank   = _rank_label(home)
    a_rank   = _rank_label(away)
    mf_rank  = _rank_label(mf)
    bh = base_card_fn(home).lower()
    ba = base_card_fn(away).lower()
    bmf = base_card_fn(mf).lower()

    # Duplicate note (the AI is told only to classify SIM/OPP/NONE; the duplicate
    # rule resolution happens in the rule layer below).
    if bh == bmf and ba == bmf:
        dup_note = (f"All three cards share the same base ({bh}). Apply your classification "
                    f"to each team card normally — the rule layer handles the resolution.")
    elif bh == bmf:
        dup_note = (f"DUPLICATE: Home base ({bh}) == MF base. Classify Away normally; "
                    f"the rule layer will apply the duplicate-card rule.")
    elif ba == bmf:
        dup_note = (f"DUPLICATE: Away base ({ba}) == MF base. Classify Home normally; "
                    f"the rule layer will apply the duplicate-card rule.")
    else:
        dup_note = "no duplicate — standard rules apply"

    prompt = _PROMPT.format(
        mf=mf, mf_meaning=meaning_map.get(mf, ''),
        comp_card=comp or 'none', comp_meaning=comp_mean,
        home=home, home_meaning=meaning_map.get(home, ''),
        away=away, away_meaning=meaning_map.get(away, ''),
        history_text=history_text,
        home_rank=h_rank, away_rank=a_rank, mf_rank=mf_rank,
        duplicate_note=dup_note,
    )

    if result is None:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            resp   = client.messages.create(model=api_model, max_tokens=3000,
                                            messages=[{"role":"user","content":prompt}])
            raw    = resp.content[0].text.strip()
            result = _extract_json(raw)
            if result is None:
                # Couldn't recover JSON — surface the raw text so the user can see
                preview = raw[:600] + ('…' if len(raw) > 600 else '')
                return {'_error': f'Could not parse JSON from model response. Raw text:\n{preview}'}
        except Exception as e:
            return {'_error': f'{type(e).__name__}: {e}'}

    # ── DETERMINISTIC RULE LAYER ────────────────────────────────────────
    home_type = result.get('home_type', 'NONE')
    away_type = result.get('away_type', 'NONE')
    shared_theme = bool(result.get('shared_theme', False))
    contested_mf_traits = result.get('contested_mf_traits', []) or []
    # Defensive: ensure it's a list of strings
    if not isinstance(contested_mf_traits, list):
        contested_mf_traits = []

    rule_call, rule_reason, rule_path = _apply_rules(
        home_type=home_type, away_type=away_type,
        home_rank=h_rank, away_rank=a_rank, mf_rank=mf_rank,
        home=home, away=away, mf=mf,
        bh=bh, ba=ba, bmf=bmf,
        shared_theme=shared_theme,
        contested_mf_traits=contested_mf_traits,
    )
    if rule_call is not None:
        result['call']      = rule_call
        result['reason']    = rule_reason
        result['rule_path'] = rule_path

    # Mark as v3-processed so the cache layer will reuse it.
    result[RULES_VERSION] = True

    rcache[key] = result
    save_cache_fn(rcache)
    return result


# ════════════════════════════════════════════════════════════════════════
# RULE LAYER — deterministic application of the v3 ruleset
# ════════════════════════════════════════════════════════════════════════
def _apply_rules(home_type, away_type, home_rank, away_rank, mf_rank,
                 home, away, mf, bh, ba, bmf, shared_theme,
                 contested_mf_traits=None):
    """
    Apply the v4 ruleset over the AI's SIM/OPP/NONE classifications.
    Returns (call, reason, rule_path).

    v4 adds: contested-trait deadlock. When the AI flags one or more MF
    traits that BOTH team cards meaningfully engage with (regardless of
    SIM/OPP polarity), the cards cancel thematically → Draw, irrespective
    of rank.
    """
    h, a = home_type, away_type
    h_rn = RANK_ORDER.get(home_rank, 1)
    a_rn = RANK_ORDER.get(away_rank, 1)
    h_is_pip = home_rank == 'pip'
    a_is_pip = away_rank == 'pip'
    contested_mf_traits = contested_mf_traits or []

    # ── STEP 1: DUPLICATE-CARD RULES (highest priority — override hierarchy) ──
    if bh == bmf and ba == bmf:
        # All three same base — degenerate edge case. Defer if no signal.
        if h == 'NONE' and a == 'NONE':
            return ('Draw',
                    'All three cards share the same base and neither team shows a '
                    'differentiating connection — Draw.',
                    'all-three-duplicate -> Draw')
        # Fall through to general rules if there is differential signal
    elif bh == bmf:
        # Home duplicates MF — hierarchy does NOT apply. Away decides:
        if a == 'OPP':
            return ('Away',
                    f'Duplicate: Home ({home}) shares base with MF ({mf}); '
                    f'hierarchy suspended. Away ({away}) opposes the MF — '
                    f'the opposing force takes control → Away wins.',
                    'duplicate (Home=MF) + Away OPP -> Away')
        if a == 'SIM':
            return ('Home',
                    f'Duplicate: Home ({home}) shares base with MF ({mf}); '
                    f'hierarchy suspended. Away ({away}) mirrors the MF — '
                    f'the MF matches with its duplicate → Home wins.',
                    'duplicate (Home=MF) + Away SIM -> Home')
        # Away NONE
        return ('Home',
                f'Duplicate: Home ({home}) shares base with MF ({mf}); '
                f'hierarchy suspended. Away ({away}) has no connection — '
                f'the duplicated card wins → Home wins.',
                'duplicate (Home=MF) + Away NONE -> Home')
    elif ba == bmf:
        if h == 'OPP':
            return ('Home',
                    f'Duplicate: Away ({away}) shares base with MF ({mf}); '
                    f'hierarchy suspended. Home ({home}) opposes the MF — '
                    f'the opposing force takes control → Home wins.',
                    'duplicate (Away=MF) + Home OPP -> Home')
        if h == 'SIM':
            return ('Away',
                    f'Duplicate: Away ({away}) shares base with MF ({mf}); '
                    f'hierarchy suspended. Home ({home}) mirrors the MF — '
                    f'the MF matches with its duplicate → Away wins.',
                    'duplicate (Away=MF) + Home SIM -> Away')
        return ('Away',
                f'Duplicate: Away ({away}) shares base with MF ({mf}); '
                f'hierarchy suspended. Home ({home}) has no connection — '
                f'the duplicated card wins → Away wins.',
                'duplicate (Away=MF) + Home NONE -> Away')

    # ── STEP 2: BOTH NONE → Draw ───────────────────────────────────────
    if h == 'NONE' and a == 'NONE':
        return ('Draw',
                'Neither team has a meaningful trait-level connection to the MF — '
                'neither can exert control → Draw.',
                'both-NONE -> Draw')

    # ── STEP 3: SHARED-THEME → Draw ────────────────────────────────────
    if shared_theme:
        return ('Draw',
                'All three cards share a single dominant energetic theme — '
                'neither team has unique purchase on the MF energy → Draw.',
                'shared-theme -> Draw')

    # ── STEP 4: ONE ENGAGES, ONE NONE → engaging side wins ─────────────
    # "If there is no destabiliser, the mirror wins" — and analogously, if
    # there is no mirror, the destabiliser wins. Either way, the engaging
    # side wins when the other side has no connection.
    if h != 'NONE' and a == 'NONE':
        kind = 'mirrors' if h == 'SIM' else 'destabilises'
        return ('Home',
                f'Only Home ({home}) connects to the MF — it {kind} ({h}). '
                f'Away ({away}) has no connection → Home controls the MF.',
                f'only-Home-engages ({h}) -> Home')
    if a != 'NONE' and h == 'NONE':
        kind = 'mirrors' if a == 'SIM' else 'destabilises'
        return ('Away',
                f'Only Away ({away}) connects to the MF — it {kind} ({a}). '
                f'Home ({home}) has no connection → Away controls the MF.',
                f'only-Away-engages ({a}) -> Away')

    # ── STEP 5: BOTH SAME TYPE → Draw (regardless of rank, regardless of
    # whether they engage the same or different MF traits) ─────────────
    # The principle: when both teams provide an equal MIRRORING force, neither
    # destabilises so neither controls → Draw. When both teams provide an equal
    # OPPOSING force, both equally destabilise so neither uniquely controls →
    # Draw. Home and Away can engage DIFFERENT MF traits and this still holds:
    # what matters is the TYPE of relationship, not which trait it lands on.
    if h == 'SIM' and a == 'SIM':
        return ('Draw',
                'Both teams mirror the MF — no destabilising force to take '
                'control → Draw.',
                'both-SIM -> Draw')
    if h == 'OPP' and a == 'OPP':
        return ('Draw',
                f'Both teams destabilise the MF (Home via its OPP traits, Away '
                f'via its OPP traits — possibly different MF aspects). Both '
                f'apply equal opposing force, so neither uniquely controls → Draw.',
                'both-OPP -> Draw')

    # ── STEP 5b: CONTESTED-TRAIT SAFETY NET ────────────────────────────
    # If the AI returned a SIM/OPP mismatch but ALSO flagged that both teams
    # engage the same MF trait, that's the trait deadlock case from the
    # Judgement/6S/8Wrx example — both engage the "stuck/unfinished" trait
    # via different lenses. Treat as Draw rather than letting OPP win.
    if contested_mf_traits:
        contested_preview = ', '.join(f'"{t}"' for t in contested_mf_traits[:2])
        more = f' (+{len(contested_mf_traits)-2} more)' if len(contested_mf_traits) > 2 else ''
        return ('Draw',
                f'Both teams engage the same MF trait{"s" if len(contested_mf_traits) > 1 else ""} — '
                f'{contested_preview}{more} — even though their AI classifications '
                f'differ, they cancel on the contested trait → Draw.',
                f'contested-trait deadlock -> Draw')

    # ── STEP 7: ONE SIM, ONE OPP — apply exceptions then standard rule ─
    # PIP-PIP EXCEPTION: both pips, one SIM one OPP → Draw
    if h_is_pip and a_is_pip:
        return ('Draw',
                f'Both team cards are pips, one mirrors and one destabilises — '
                f'pips lack the rank-weight to decisively control the MF → Draw '
                f'(pip-pip exception).',
                'pip-pip OPP/SIM -> Draw')

    # HIERARCHY EXCEPTION: SIM is HIGHER rank, OPP is LOWER rank → Draw
    if h == 'SIM' and a == 'OPP' and h_rn > a_rn:
        return ('Draw',
                f'Home ({home}, {home_rank}) mirrors the MF at higher rank; '
                f'Away ({away}, {away_rank}) destabilises but at lower rank — '
                f'they cancel → Draw (hierarchy exception).',
                'hierarchy exception: SIM-higher vs OPP-lower -> Draw')
    if a == 'SIM' and h == 'OPP' and a_rn > h_rn:
        return ('Draw',
                f'Away ({away}, {away_rank}) mirrors the MF at higher rank; '
                f'Home ({home}, {home_rank}) destabilises but at lower rank — '
                f'they cancel → Draw (hierarchy exception).',
                'hierarchy exception: SIM-higher vs OPP-lower -> Draw')

    # STANDARD RULE: destabiliser (OPP) wins
    if h == 'OPP' and a == 'SIM':
        rank_note = (f' Home outranks Away ({home_rank} vs {away_rank}).' if h_rn > a_rn
                     else (f' Away outranks Home ({away_rank} vs {home_rank}), but the '
                           f'destabiliser still controls.' if a_rn > h_rn
                           else ' (equal rank)'))
        return ('Home',
                f'Home ({home}) destabilises the MF; Away ({away}) mirrors. '
                f'The destabiliser controls →  Home wins.{rank_note}',
                f'OPP beats SIM -> Home')
    if a == 'OPP' and h == 'SIM':
        rank_note = (f' Away outranks Home ({away_rank} vs {home_rank}).' if a_rn > h_rn
                     else (f' Home outranks Away ({home_rank} vs {away_rank}), but the '
                           f'destabiliser still controls.' if h_rn > a_rn
                           else ' (equal rank)'))
        return ('Away',
                f'Away ({away}) destabilises the MF; Home ({home}) mirrors. '
                f'The destabiliser controls →  Away wins.{rank_note}',
                f'OPP beats SIM -> Away')

    # Fallback — shouldn't be reached, but return Draw safely
    return ('Draw',
            'Rule layer fell through — defaulting to Draw.',
            'fallback -> Draw')


# ════════════════════════════════════════════════════════════════════════
# PUBLIC API — call → confidence
# ════════════════════════════════════════════════════════════════════════
def analysis_to_prediction(result):
    if not result or '_error' in result or 'call' not in result:
        return None, 0.5
    return result['call'], _CONF_MAP.get(result.get('confidence', 'moderate'), 0.58)
