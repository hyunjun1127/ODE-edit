"""Request-macro factual scores, with strict NLL comparisons and explicit counts."""
import math


def mean(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("EMPTY_OR_NONFINITE_EVALUATION")
    return math.fsum(values) / len(values)


def harmonic(values):
    if any(value < 0 for value in values):
        raise ValueError("NEGATIVE_SUCCESS_RATE")
    return 0.0 if any(value == 0 for value in values) else len(values) / math.fsum(1 / x for x in values)


def counterfact(cases):
    output = {}
    for kind, label in (("rewrite", "Efficacy"), ("paraphrase", "Generalization"),
                         ("neighborhood", "Specificity")):
        rates = []
        for case in cases:
            rows = case[f"{kind}_prompts_probs"]
            scores = []
            for row in rows:
                new, true = row["target_new"], row["target_true"]
                if not math.isfinite(new) or not math.isfinite(true):
                    raise ValueError("NONFINITE_NLL")
                scores.append(float(true < new if kind == "neighborhood" else new < true))
            rates.append(mean(scores))
        output[label] = 100 * mean(rates)
    values = [output[k] for k in ("Efficacy", "Generalization", "Specificity")]
    output.update(Score=harmonic(values),
                  Score_AlphaEdit_display=harmonic([round(x, 2) for x in values]),
                  requests=len(cases))
    return output


def zsre(cases):
    output = {}
    for key, label in (("rewrite_prompts_correct", "Efficacy"),
                       ("paraphrase_prompts_correct", "Generalization"),
                       ("neighborhood_W0_agreement", "Specificity"),
                       ("neighborhood_prompts_correct", "Specificity_loc_ans")):
        output[label] = 100 * mean([mean([float(x) for x in row[key]]) for row in cases])
    output["requests"] = len(cases)
    return output
