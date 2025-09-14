def evaluate_responses(data):
    records = []
    for row in data:
        pid, prompt, resp = row["id"], row["prompt"], row["response"]
        ref = row.get("reference", None)

        s_inst = score_instruction_following(prompt, resp)
        s_coh = score_coherence(prompt, resp)
        s_assum = score_assumption_control(resp)
        s_hall = score_hallucination(prompt, resp)
        s_ref = score_reference_agreement(resp, ref)

        # Composite: weighted average with hallucination penalty
        base = 0.3*s_inst + 0.3*s_coh + 0.2*s_assum
        if s_ref is not None:
            base += 0.2*s_ref
        else:
            base /= 0.8  # normalize if no reference
        composite = base * (1 - 0.5*s_hall)

        records.append({
            "id": pid,
            "prompt": prompt,
            "response": resp,
            "instruction": s_inst,
            "coherence": s_coh,
            "assumption": s_assum,
            "hallucination_risk": s_hall,
            "reference_agreement": s_ref,
            "composite_score": round(max(0, min(1, composite)), 3)
        })

    df = pd.DataFrame(records)
    leaderboard = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df, leaderboard
