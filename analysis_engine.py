"""
Holistic match analysis engine — INTERIM ruleset (rules_v6_interim).

WHAT'S NEW vs v5
================
- Hierarchy logic removed entirely (no rank, no pip-pip exception, no hierarchy
  exception). Rank no longer influences the rule layer.
- SIM/OPP precedence FLIPPED: the mirror (SIM) wins over the destabiliser (OPP).
  Previous engine treated OPP as the more active form of control; the refined
  reading rule is that the MF prefers an aligning partner first, with opposition
  as the fallback when no alignment is present.
- Contested-trait deadlock removed from the rule layer. It was firing too often
  and was the main cause of "everything is labelled a draw". Shared-theme draw
  (all three cards share one dominant theme) is preserved.
- Prompt revised: removed rank/pip/hierarchy/contested mentions, flipped the
  tie-break advice to SIM-preferred, added explicit guidance on reading
  reversed cards with dual/opposing meanings (use the other cards to
  disambiguate which polarity is active).

CORE METHODOLOGY (unchanged in spirit)
======================================
A team card "connects" to the Match Force (MF) by either:
  • SIM (mirror)      — possesses an EQUAL, MIRRORING energy / characteristic
  • OPP (destabilise) — possesses an EQUAL but OPPOSING energy / characteristic
  • NONE              — no meaningful connection

The AI emits SIM / OPP / NONE for each team card. The rule layer below is
deterministic.

PRIMARY RULES (interim)
=======================
  • Only one card connects (other is NONE)  → the engaging team wins
  • One SIM, one OPP                        → SIM wins (mirror controls; opposition is fallback)
  • Both SIM                                → Draw (alignments compete, neither uniquely controls)
  • Both OPP                                → Draw (both block the MF, neither uniquely controls)
  • Both NONE                               → Draw (no team can exert control)
  • Shared-theme (all three cards)          → Draw

DUPLICATE-CARD RULES (highest priority)
=======================================
Triggered when a team card has the same base card as the MF.
  • Remaining card is SIM  → duplicate-side team wins (MF matches with its duplicate)
  • Remaining card is OPP  → remaining team wins      (opposing force takes control)
  • Remaining card is NONE → duplicate-side team wins (duplicate is the only connection)

SPECIFICITY DIRECTIVE (unchanged)
=================================
The AI must compare at the TRAIT level — not at broad themes. Each card has
many distinct traits. Home and Away may connect to DIFFERENT aspects of the MF.
Lesser-used niche traits count.
"""
import json, re, os

ANALYSIS_PREFIX = "ANALYSIS||"
RULES_VERSION = "rules_v6_interim"   # bump invalidates older cached results

# When True, reuse SIM/OPP classifications from older cached entries (saves API
# costs). When False, every prediction re-calls the API under the new prompt.
# Old v5 cached AI outputs were generated under the OPP-preferred prompt;
# reusing them works fine because the new rule layer just applies different
# logic to the same classifications. Flip to False (or delete
# resonance_cache.json) if you want fully fresh AI calls.
REUSE_OLDER_AI_OUTPUT = True


def _extract_json(text):
    """
    Pull a JSON object out of a model response. Robust to:
      - markdown fences (```json ... ``` or just ``` ... ```)
      - leading/trailing prose around the JSON
      - extra whitespace
    """
    if not text:
        return None
    t = text.strip()
    if t.startswith('```'):
        t = re.sub(r'^```[a-zA-Z0-9_-]*\n?', '', t)
        if t.endswith('```'):
            t = t[:-3]
        t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
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
# THE PROMPT — SIM-preferred tie-break, no rank/hierarchy, dual-meaning guidance
# ════════════════════════════════════════════════════════════════════════
_PROMPT = """You are a tarot analyst for a sports prediction system. Determine how each team card relates to the Match Force (MF) at the TRAIT level.

═══════════════════════════════════════════════════════════════════════
CORE METHODOLOGY
═══════════════════════════════════════════════════════════════════════
A team card connects to the MF in one of two ways:
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
A card with the niche trait "domestic goddess" is an EXACT mirror to a
card with the niche trait "homemaker" — and an EXACT opposite to a card with
"neglected home" — even though those traits are minor within each card's
overall meaning. Such lesser-used trait matches have historically been
decisive. Do NOT overlook them by focusing only on each card's "headline"
themes.

A SIM/OPP connection requires SPECIFIC trait language — a literal mirror or
literal opposite of a phrase in the meaning text. Vague resonance does not
qualify. Quote the EXACT phrase from each side that makes the connection.

═══════════════════════════════════════════════════════════════════════
HANDLING REVERSED CARDS WITH DUAL / OPPOSING MEANINGS
═══════════════════════════════════════════════════════════════════════
Reversed cards often carry BOTH polarities of a theme — e.g. AceWrx says
"failure, blockage, dead ends" AND "a certain clouded joy"; Death rx says
"fear of beginnings, repeating negative patterns" AND "inertia, sleep,
lethargy"; 4Prx says "excessive hoarding and control" AND "finally releasing
what you've been gripping too tightly, generosity, letting go".

Which polarity is ACTIVE in a given match is determined by the OTHER CARDS
in play. Read all three meanings together and pick the polarity of each
reversed card that makes the energetic story coherent. Default to the
stronger/more established polarity when no clear contextual cue exists.

This applies whether the reversed card is the MF or a team card. If the MF
is reversed and has dual meanings, let the team cards' traits disambiguate
which MF polarity is operative for this match.

═══════════════════════════════════════════════════════════════════════
SHARED-THEME CHECK (all three cards)
═══════════════════════════════════════════════════════════════════════
Check whether ALL THREE cards (Home, Away, MF) share a single dominant
energetic theme (e.g. all relate to "lack of clarity", all to "endings",
all to "withdrawal"). If yes — flag this. When all three cards crowd the
same thematic space, the rule layer will resolve to Draw.

═══════════════════════════════════════════════════════════════════════
SAME-TYPE PRINCIPLE — CLASSIFY DELIBERATELY
═══════════════════════════════════════════════════════════════════════
The rule layer treats both-SIM as Draw and both-OPP as Draw. So your
SIM/OPP/NONE classification directly determines whether the match resolves
to a winner or a draw.

When a team card has BOTH SIM and OPP candidate connections to the MF:
  • Count the meaningful connections of each type
  • Weigh how literal/specific each is
  • Pick the type with the BROADER and MORE LITERAL set of engagements
  • Tie-break in favour of SIM. The MF prefers an aligning partner first;
    opposition is the fallback mode of control. When the alignment case
    and the opposition case are roughly equal in strength, classify as SIM.

═══════════════════════════════════════════════════════════════════════
OUTPUT — SIM / OPP / NONE FOR EACH TEAM CARD
═══════════════════════════════════════════════════════════════════════
You do NOT decide the final Home/Away/Draw call. The rule layer applies
the rules (duplicate, both-same-type, shared-theme, SIM-beats-OPP)
deterministically over your classifications.

LIST ALL MEANINGFUL CONNECTIONS in home_connections and away_connections,
not just the single strongest. If a team card has FOUR meaningful trait
connections to MF aspects, list ALL FOUR — even if one is the headline.

═══════════════════════════════════════════════════════════════════════
INPUT
═══════════════════════════════════════════════════════════════════════
Match Force: {mf}
MF meaning (read in full, including all lesser-used phrases):
{mf_meaning}

MF complement ({comp_card}) — provided as opposite-vocabulary reference only:
{comp_meaning}

Home: {home}
Home meaning (read in full):
{home_meaning}

Away: {away}
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

  "history_signal": "one sentence on whether history meaningfully supports a side or is neutral",

  "confidence": "high | good | moderate | low",
  "reasoning": "2-3 sentences justifying the SIM/OPP/NONE classifications, citing specific trait language"
}}"""


_CONF_MAP = {'high': 0.83, 'good': 0.70, 'moderate': 0.58, 'low': 0.48}


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

    # Optionally reuse the AI's SIM/OPP/NONE classifications from a prior
    # version's cache, and just re-run the deterministic rule layer below.
    reuse_ai = (REUSE_OLDER_AI_OUTPUT
                and cached_raw
                and 'home_type' in cached_raw
                and 'away_type' in cached_raw
                and 'home_connections' in cached_raw)

    if reuse_ai:
        result = dict(cached_raw)
        # Strip stale call/reason/rule_path so the rule layer can rewrite them
        for k in ('call', 'reason', 'rule_path'):
            result.pop(k, None)
        # Also clear older RULES_VERSION flags so only the current one applies
        for old_flag in ('rules_v3', 'rules_v4', 'rules_v5'):
            result.pop(old_flag, None)
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

    bh = base_card_fn(home).lower()
    ba = base_card_fn(away).lower()
    bmf = base_card_fn(mf).lower()

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
                preview = raw[:600] + ('…' if len(raw) > 600 else '')
                return {'_error': f'Could not parse JSON from model response. Raw text:\n{preview}'}
        except Exception as e:
            return {'_error': f'{type(e).__name__}: {e}'}

    # ── DETERMINISTIC RULE LAYER ────────────────────────────────────────
    home_type = result.get('home_type', 'NONE')
    away_type = result.get('away_type', 'NONE')
    shared_theme = bool(result.get('shared_theme', False))

    rule_call, rule_reason, rule_path = _apply_rules(
        home_type=home_type, away_type=away_type,
        home=home, away=away, mf=mf,
        bh=bh, ba=ba, bmf=bmf,
        shared_theme=shared_theme,
    )
    if rule_call is not None:
        result['call']      = rule_call
        result['reason']    = rule_reason
        result['rule_path'] = rule_path

    # Mark as current-version-processed so the cache layer will reuse it.
    result[RULES_VERSION] = True

    rcache[key] = result
    save_cache_fn(rcache)
    return result


# ════════════════════════════════════════════════════════════════════════
# RULE LAYER — INTERIM v6 (no hierarchy, SIM > OPP, duplicates preserved)
# ════════════════════════════════════════════════════════════════════════
def _apply_rules(home_type, away_type, home, away, mf, bh, ba, bmf, shared_theme):
    """
    Apply the interim ruleset over the AI's SIM/OPP/NONE classifications.
    Returns (call, reason, rule_path).

    Rule precedence (top to bottom):
      1. Duplicate-card rules           (override everything else)
      2. Both NONE                       → Draw
      3. Shared-theme                    → Draw
      4. One engages, one NONE           → engaging side wins
      5. Both same type (SIM/SIM or
         OPP/OPP)                        → Draw
      6. One SIM, one OPP                → SIM wins (mirror beats destabiliser)
    """
    h, a = home_type, away_type

    # ── STEP 1: DUPLICATE-CARD RULES (highest priority) ────────────────
    if bh == bmf and ba == bmf:
        # All three same base — edge case
        if h == 'NONE' and a == 'NONE':
            return ('Draw',
                    'All three cards share the same base and neither team shows a '
                    'differentiating connection → Draw.',
                    'all-three-duplicate -> Draw')
        # Otherwise fall through to general rules
    elif bh == bmf:
        # Home duplicates MF
        if a == 'OPP':
            return ('Away',
                    f'Duplicate: Home ({home}) shares base with MF ({mf}). '
                    f'Away ({away}) destabilises the MF — the opposing force '
                    f'takes control → Away wins.',
                    'duplicate (Home=MF) + Away OPP -> Away')
        if a == 'SIM':
            return ('Home',
                    f'Duplicate: Home ({home}) shares base with MF ({mf}). '
                    f'Away ({away}) mirrors the MF — the MF matches with its '
                    f'duplicate → Home wins.',
                    'duplicate (Home=MF) + Away SIM -> Home')
        return ('Home',
                f'Duplicate: Home ({home}) shares base with MF ({mf}). '
                f'Away ({away}) has no connection — the duplicate is the only '
                f'connection → Home wins.',
                'duplicate (Home=MF) + Away NONE -> Home')
    elif ba == bmf:
        # Away duplicates MF
        if h == 'OPP':
            return ('Home',
                    f'Duplicate: Away ({away}) shares base with MF ({mf}). '
                    f'Home ({home}) destabilises the MF — the opposing force '
                    f'takes control → Home wins.',
                    'duplicate (Away=MF) + Home OPP -> Home')
        if h == 'SIM':
            return ('Away',
                    f'Duplicate: Away ({away}) shares base with MF ({mf}). '
                    f'Home ({home}) mirrors the MF — the MF matches with its '
                    f'duplicate → Away wins.',
                    'duplicate (Away=MF) + Home SIM -> Away')
        return ('Away',
                f'Duplicate: Away ({away}) shares base with MF ({mf}). '
                f'Home ({home}) has no connection — the duplicate is the only '
                f'connection → Away wins.',
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

    # ── STEP 5: BOTH SAME TYPE → Draw ──────────────────────────────────
    if h == 'SIM' and a == 'SIM':
        return ('Draw',
                'Both teams mirror the MF — alignments compete, neither '
                'uniquely controls → Draw.',
                'both-SIM -> Draw')
    if h == 'OPP' and a == 'OPP':
        return ('Draw',
                'Both teams destabilise the MF — equal opposing force, '
                'neither uniquely controls → Draw.',
                'both-OPP -> Draw')

    # ── STEP 6: ONE SIM, ONE OPP → SIM wins (mirror preferred) ────────
    if h == 'SIM' and a == 'OPP':
        return ('Home',
                f'Home ({home}) mirrors the MF; Away ({away}) destabilises. '
                f'The mirror takes precedence — the MF prefers an aligning '
                f'partner before resorting to opposition → Home wins.',
                'SIM beats OPP -> Home')
    if a == 'SIM' and h == 'OPP':
        return ('Away',
                f'Away ({away}) mirrors the MF; Home ({home}) destabilises. '
                f'The mirror takes precedence — the MF prefers an aligning '
                f'partner before resorting to opposition → Away wins.',
                'SIM beats OPP -> Away')

    # Fallback — shouldn't be reached given the cases above are exhaustive
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
