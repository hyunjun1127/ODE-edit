"""Request-macro factual scores, with strict NLL comparisons and explicit counts."""
import math
import numpy as np


def mean(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("EMPTY_OR_NONFINITE_EVALUATION")
    return math.fsum(values) / len(values)


def harmonic(values):
    if any(value < 0 for value in values):
        raise ValueError("NEGATIVE_SUCCESS_RATE")
    return 0.0 if any(value == 0 for value in values) else len(values) / math.fsum(1 / x for x in values)


def counterfact(cases):
    output, display_values = {}, []
    for kind, label in (("rewrite", "Efficacy"), ("paraphrase", "Generalization"),
                         ("neighborhood", "Specificity")):
        rates, native_rates = [], []
        for case in cases:
            rows = case[f"{kind}_prompts_probs"]
            scores = []
            for row in rows:
                new, true = row["target_new"], row["target_true"]
                if not math.isfinite(new) or not math.isfinite(true):
                    raise ValueError("NONFINITE_NLL")
                scores.append(float(true < new if kind == "neighborhood" else new < true))
            rates.append(mean(scores))
            native_rates.append(np.mean(scores))
        output[label] = 100 * mean(rates)
        # Upstream summarizes via NumPy at both levels before np.around.
        # Preserve the fsum-based raw metrics; halfway display rounding can
        # otherwise differ despite exactly identical per-prompt success bits.
        display_values.append(float(np.around(np.mean(native_rates) * 100, 2)))
    values = [output[k] for k in ("Efficacy", "Generalization", "Specificity")]
    output.update(Score=harmonic(values),
                  Score_AlphaEdit_display=harmonic(display_values),
                  requests=len(cases))
    return output


def zsre(cases):
    """Paper Loc is loc_ans accuracy; W0 agreement is a separate auxiliary.

    Reduce saved per-token correctness within each request, then across requests.
    A missing W0 reference does not make measured loc_ans accuracy unavailable.
    Frozen historical summaries require saved-raw reduction, not key relabeling.
    """
    output = {}
    for key, label in (("rewrite_prompts_correct", "Efficacy"),
                       ("paraphrase_prompts_correct", "Generalization"),
                       ("neighborhood_prompts_correct", "Specificity")):
        output[label] = 100 * mean([mean([float(x) for x in row[key]]) for row in cases])
    # Compatibility alias for the same measured answer accuracy, never W0.
    output["Specificity_loc_ans"] = output["Specificity"]
    if all(row.get("neighborhood_W0_agreement") is not None for row in cases):
        output["W0_prediction_agreement"] = 100 * mean([
            mean([float(x) for x in row["neighborhood_W0_agreement"]]) for row in cases])
    output["requests"] = len(cases)
    return output
