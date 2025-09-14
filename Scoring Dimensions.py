def score_instruction_following(prompt: str, response: str) -> float:
    """
    Improved heuristic:
    - If prompt has an action keyword (list, explain, compare, etc.),
      check not only exact keyword match but also structural cues in response.
    - Example: for "list", look for commas or bullets; for "explain", look for 'because', 'due to'.
    - Score ranges [0,1].
    """
    ACTIONS = ["list", "explain", "describe", "compare", "summarize",
               "calculate", "show", "give", "implement", "write", "provide"]
    p_low, r_low = prompt.lower(), response.lower()
    found = [a for a in ACTIONS if a in p_low]

    # If no explicit instruction word in prompt, fallback
    if not found:
        return 1.0 if r_low.strip() else 0.0

    score = 0.0
    for action in found:
        if action == "list":
            # check if response looks like a list (commas, newlines, or digits)
            if "," in r_low or "\n" in r_low or re.search(r"\d+\.", r_low):
                score += 1.0
        elif action == "explain" or action == "describe":
            # look for cause-effect or detail words
            if re.search(r"\b(because|since|due to|as a result|which means)\b", r_low):
                score += 1.0
        elif action == "compare":
            # check for words like 'whereas', 'while', 'both', 'difference'
            if re.search(r"\b(whereas|while|both|difference|vs\.?)\b", r_low):
                score += 1.0
        else:
            # fallback: if the action keyword itself appears in response
            if action in r_low:
                score += 1.0

    return round(score / len(found), 3)



def score_coherence(prompt: str, response: str) -> float:
    """Semantic alignment between prompt and response."""
    return round(tfidf_similarity(prompt, response), 3)


def score_hallucination(prompt: str, response: str) -> float:
    """
    Detect hallucination risk.
    Higher score = higher chance of hallucination.
    """
    p_entities = extract_entities(prompt)
    r_entities = extract_entities(response)

    new_nums = [n for n in r_entities["numbers"] if n not in p_entities["numbers"]]
    new_caps = [c for c in r_entities["caps"] if c not in p_entities["caps"]]
    new_dates = [d for d in r_entities["dates"] if d not in p_entities["dates"]]

    risk = 0.15 * len(new_nums) + 0.12 * len(new_caps) + 0.2 * len(new_dates)
    if re.search(r"\b(maybe|might|not sure|possibly)\b", response, re.I):
        risk *= 0.6  # soften if hedging language used
    return round(min(1.0, risk), 3)


def score_assumption_control(response: str) -> float:
    """
    Penalize unwarranted assumptions.
    Reward disclaimers (if, assuming, based on).
    """
    r_low = response.lower()
    overconfident = re.findall(r"\b(always|never|definitely|certainly|obviously)\b", r_low)
    disclaimers = re.findall(r"\b(if|assuming|based on|given that)\b", r_low)

    score = 1.0 - 0.15 * len(overconfident) + 0.1 * len(disclaimers)
    return round(max(0.0, min(1.0, score)), 3)


def score_reference_agreement(response: str, reference: str) -> float:
    """Agreement with reference answer, if provided."""
    if not reference:
        return None
    return round(tfidf_similarity(reference, response), 3)
