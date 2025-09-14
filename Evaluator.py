class AgenticEvaluator:
    def __init__(self, dataset, ref_key=None):
        """
        dataset: list of dicts with 'id', 'prompt', 'response', (optional 'reference')
        """
        self.df = pd.DataFrame(dataset)
        if ref_key and ref_key in self.df.columns:
            self.df['reference'] = self.df[ref_key]
        else:
            self.df['reference'] = None

    def score_instruction_following(self, prompt, response):
        ACTIONS = ["list", "explain", "describe", "compare", "summarize",
                   "calculate", "show", "give", "implement", "write", "provide"]
        p_low, r_low = prompt.lower(), response.lower()
        found = [a for a in ACTIONS if a in p_low]

        if not found:
            return 1.0 if r_low.strip() else 0.0

        score = 0.0
        for action in found:
            if action == "list":
                if "," in r_low or "\n" in r_low or re.search(r"\d+\.", r_low):
                    score += 1.0
            elif action in ["explain", "describe"]:
                if re.search(r"\b(because|since|due to|as a result|which means)\b", r_low):
                    score += 1.0
            elif action == "compare":
                if re.search(r"\b(whereas|while|both|difference|vs\.?)\b", r_low):
                    score += 1.0
            else:
                if action in r_low:
                    score += 1.0

        return round(score / len(found), 3)

    def score_coherence(self, prompt, response):
        return round(tfidf_similarity(prompt, response), 3)

    def score_hallucination(self, prompt, response):
        p_entities = extract_entities(prompt)
        r_entities = extract_entities(response)

        new_nums = [n for n in r_entities["numbers"] if n not in p_entities["numbers"]]
        new_caps = [c for c in r_entities["caps"] if c not in p_entities["caps"]]
        new_dates = [d for d in r_entities["dates"] if d not in p_entities["dates"]]

        risk = 0.15 * len(new_nums) + 0.12 * len(new_caps) + 0.2 * len(new_dates)
        if re.search(r"\b(maybe|might|not sure|possibly)\b", response, re.I):
            risk *= 0.6
        return round(min(1.0, risk), 3)

    def score_assumption_control(self, response):
        r_low = response.lower()
        overconfident = re.findall(r"\b(always|never|definitely|certainly|obviously)\b", r_low)
        disclaimers = re.findall(r"\b(if|assuming|based on|given that)\b", r_low)

        score = 1.0 - 0.15 * len(overconfident) + 0.1 * len(disclaimers)
        return round(max(0.0, min(1.0, score)), 3)

    def score_reference_agreement(self, response, reference):
        if not reference:
            return None
        return round(tfidf_similarity(reference, response), 3)

    def evaluate_all(self):
        records = []
        for idx, row in self.df.iterrows():
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            reference = row.get("reference", None)

            s_inst = self.score_instruction_following(prompt, response)
            s_coh = self.score_coherence(prompt, response)
            s_assum = self.score_assumption_control(response)
            s_hall = self.score_hallucination(prompt, response)
            s_ref = self.score_reference_agreement(response, reference)

            base = 0.3*s_inst + 0.3*s_coh + 0.2*s_assum
            if s_ref is not None:
                base += 0.2*s_ref
            else:
                base /= 0.8  # normalize

            composite = base * (1 - 0.5*s_hall)
            composite = round(max(0, min(1, composite)), 3)

            records.append({
                "id": row.get("id", idx),
                "prompt": prompt,
                "response": response,
                "instruction_score": s_inst,
                "coherence_score": s_coh,
                "assumption_score": s_assum,
                "hallucination_risk": s_hall,
                "reference_agreement": s_ref,
                "composite_score": composite
            })

        report_df = pd.DataFrame(records)
        leaderboard_df = report_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        return report_df, leaderboard_df
