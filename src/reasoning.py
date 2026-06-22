"""
Per-candidate reasoning strings for the final top-100.

Template: [Experience] + [Evidence] + [Behavior] + [Concern]. Every clause
is built directly from a field in the candidate record or the feature
dict computed for them -- there is no free-text generation step and no
LLM call, on purpose. The brief asks for "specific, short, truthful"
reasoning and explicitly forbids hallucination; the safest way to satisfy
both is to not generate prose from a model at all, just compose template
fragments from values we already trust because we computed them
ourselves. The tradeoff is the sentences are a bit mechanical -- we accept
that over the risk of a model inventing a plausible-sounding but false
claim about a candidate.
"""


def _experience_clause(candidate, feat):
    years = candidate["profile"]["years_of_experience"]
    title = candidate["profile"]["current_title"]
    company = candidate["profile"]["current_company"]
    return f"{years:.1f} yrs experience, currently {title} at {company}."


def _evidence_clause(candidate, feat):
    req = feat["matched_required_skills"]
    pref = feat["matched_preferred_skills"]
    parts = []
    if req:
        shown = req[:4]
        parts.append(f"matches required skills: {', '.join(shown)}" +
                     (f" (+{len(req)-4} more)" if len(req) > 4 else ""))
    if pref:
        parts.append(f"{len(pref)} preferred skill(s) matched")
    history = candidate["career_history"]
    if history:
        most_recent = history[0]
        desc = most_recent.get("description", "").strip()
        if desc:
            snippet = desc if len(desc) <= 140 else desc[:137] + "..."
            parts.append(f'prior role evidence: "{snippet}"')
    if not parts:
        parts.append("no direct required/preferred skill matches found in profile")
    return "; ".join(parts) + "."


def _behavior_clause(candidate, feat):
    sig = candidate["redrob_signals"]
    bits = []
    if sig["open_to_work_flag"]:
        bits.append(f"open to work, {sig['notice_period_days']}-day notice")
    else:
        bits.append("not actively open to work")
    bits.append(f"recruiter response rate {sig['recruiter_response_rate']*100:.0f}%")
    bits.append(f"last active {sig['last_active_date']}")
    return ", ".join(bits) + "."


def _concern_clause(candidate, feat):
    # Iteration 2 (see experiments/exp_log.md for the self-judge pass against
    # the spec's own Stage-4 reasoning-quality checks): low-ranked top-100
    # candidates with thin skill overlap were getting the same flat "No
    # material concerns" closer as a rank-1 candidate -- inconsistent with
    # the spec's "rank consistency" check (a low-rank candidate with glowing
    # reasoning reads as if reasoning was generated independently of rank).
    # The experience-band and thin-skill-overlap checks below are new;
    # honeypot/negative_reasons/title_penalty_note were already here.
    concerns = []
    if feat["honeypot_probability"] > 0:
        concerns.append(f"honeypot_probability={feat['honeypot_probability']:.2f} -- flagged, not excluded outright")
    if feat["negative_reasons"]:
        concerns.append("; ".join(feat["negative_reasons"]))
    if feat["title_penalty_note"]:
        concerns.append(feat["title_penalty_note"])
    if feat["experience_band_fit"] < 0.8:
        years = feat["years_of_experience"]
        concerns.append(f"{years:.1f} yrs sits outside (or at the edge of) the JD's target experience band")
    req_n = len(feat["matched_required_skills"])
    pref_n = len(feat["matched_preferred_skills"])
    if req_n <= 1 and pref_n <= 1:
        concerns.append("thin skill overlap with the JD -- fit here leans on adjacent experience and "
                         "engagement signals more than direct skill match")
    if not concerns:
        return "No material concerns flagged by our rule set."
    return "Concern: " + "; ".join(concerns) + "."


def generate_reasoning(candidate, feat):
    return " ".join([
        _experience_clause(candidate, feat),
        _evidence_clause(candidate, feat),
        _behavior_clause(candidate, feat),
        _concern_clause(candidate, feat),
    ])
