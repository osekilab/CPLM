from typing import List
import pandas as pd

def get_legend_label(model_name):
    return model_name

def calc_accuracy_from_scores(scores: List[float],
                              scoring_method: str,
                              ) -> float:
    """
    Compute accuracy given scores.

    Notes:
        Odd-numbered lines contain grammatical sentences, and even-numbered lines contain ungrammatical sentences.

        Whether a pair is scored as correct depends on the scoring method:
        For MLM (masked LM) scoring, the pseudo-log-likelihood is used.
        For the holistic method (causal LM scoring), the negative log-likelihood (NLL) sum should be smaller for the correct sentence.
    """

    num_pairs = len(scores) // 2
    num_correct = 0
    for i in range(num_pairs):
        if scoring_method == 'clm' and float(scores[2 * i]) < float(scores[2 * i + 1]):
#            print(float(scores[2 * i]))
#            print(float(scores[2 * i + 1]))
            # In causal LM, lower score (negative log-likelihood) for the grammatical sentence indicates correctness.
            num_correct += 1
        elif scoring_method == 'holistic' and float(scores[2 * i]) < float(scores[2 * i + 1]):
            # Holistic scoring: smaller cross-entropy sum indicates correctness.
            num_correct += 1

    return num_correct / num_pairs * 100


def get_group_names(df: pd.DataFrame,
                    ) -> List[str]:
    df['group_name'] = df['model'].str.cat(df['corpora'], sep='+').astype('category')
    res = df['group_name'].unique().tolist()
    return res
