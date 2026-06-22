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
    concerns = []
    if feat["honeypot_probability"] > 0:
        concerns.append(f"honeypot_probability={feat['honeypot_probability']:.2f} -- flagged, not excluded outright")
    if feat["negative_reasons"]:
        concerns.append("; ".join(feat["negative_reasons"]))
    if feat["title_penalty_note"]:
        concerns.append(feat["title_penalty_note"])
    if feat["experience_band_fit"] < 0.5:
        concerns.append("experience years sit outside the JD's target band")
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
