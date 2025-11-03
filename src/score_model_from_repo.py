import numpy as np
import sys
import os
import torch
import random
from collections import defaultdict


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a model on test paradigms.")
    parser.add_argument('--model_dir', required=True, help="Path to the local model directory.")
    parser.add_argument('--output', required=True, help="Path to the output file for results.")
    return parser.parse_args()

args = parse_arguments()
LOCAL_MODEL_PATH = args.model_dir
output_file = args.output


module_path = os.path.abspath("~/CPLM/src/transformers/src")
if module_path not in sys.path:
    sys.path.append(module_path)


from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers import GPT2Config, PreTrainedTokenizerFast
from unmasked.clm.scoring_device import clm_score_model_on_paradigm
from unmasked import configs
from unmasked.utils_clm import calc_accuracy_from_scores


grammar_mapping = {
    "agreement_determiner_noun-across_1_adjective": "D-N AGR",
    "agreement_determiner_noun-between_neighbors": "D-N AGR",
    "agreement_subject_verb-across_prepositional_phrase": "S-V AGR",
    "agreement_subject_verb-across_relative_clause": "S-V AGR",
    "agreement_subject_verb-in_question_with_aux": "S-V AGR",
    "agreement_subject_verb-in_simple_question": "S-V AGR",
    "anaphor_agreement-pronoun_gender": "ANA.AGR",
    "argument_structure-dropped_argument": "ARG.STR",
    "argument_structure-swapped_arguments": "ARG.STR",
    "argument_structure-transitive": "ARG.STR",
    "binding-principle_a": "BINDING",
    "case-subjective_pronoun": "CASE",
    "ellipsis-n_bar": "ELLIPSIS",
    "filler-gap-wh_question_object": "FILLER.GAP",
    "filler-gap-wh_question_subject": "FILLER.GAP",
    "irregular-verb": "IRREGULAR",
    "island-effects-adjunct_island": "ISLAND",
    "island-effects-coordinate_structure_constraint": "ISLAND",
    "local_attractor-in_question_with_aux": "LOCAL.ATR",
    "npi_licensing-matrix_question": "NPI",
    "npi_licensing-only_npi_licensor": "NPI",
    "quantifiers-existential_there": "QUANTIFIERS",
    "quantifiers-superlative": "QUANTIFIERS"
}


LOWER_CASE = True
TEST_SUITE_NAME = 'zorro'

if TEST_SUITE_NAME == 'blimp':
    num_expected_scores = 2000
elif TEST_SUITE_NAME == 'zorro':
    num_expected_scores = 4000
else:
    raise AttributeError('Invalid "TEST_SUITE_NAME".')

scoring_method = 'clm'
score_model_on_paradigm = clm_score_model_on_paradigm

# load from local
tokenizer = PreTrainedTokenizerFast.from_pretrained(LOCAL_MODEL_PATH, add_prefix_space=True)
config = GPT2Config.from_pretrained(LOCAL_MODEL_PATH)
config.vocab_size = tokenizer.vocab_size  
model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_PATH, config=config)
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


category_accuracies = defaultdict(list)


for path_paradigm in (configs.Dirs.test_suites / TEST_SUITE_NAME).glob('*.txt'):
    # scoring
    print(f"Scoring {path_paradigm.name:<60} with {LOCAL_MODEL_PATH:<40} and method={scoring_method}")
    scores = score_model_on_paradigm(model, tokenizer, path_paradigm, lower_case=LOWER_CASE, device=device)

    assert len(scores) == num_expected_scores

    # compute accuracy
    accuracy = calc_accuracy_from_scores(scores, scoring_method)
    print(accuracy)


    paradigm_name = path_paradigm.stem
    mid_category = grammar_mapping.get(paradigm_name, "UNKNOWN")


    category_accuracies[mid_category].append(accuracy)


overall_accuracies = []
with open(output_file, 'w') as f:
    for category, acc_list in category_accuracies.items():
        mean_accuracy = np.mean(acc_list) if acc_list else 0.0
        overall_accuracies.append(mean_accuracy)
        f.write(f'{category}, {mean_accuracy:.1f}\n')
        print(f'{category}, {mean_accuracy:.1f}')


    overall_macro_avg = np.mean(overall_accuracies) if overall_accuracies else 0.0
    f.write(f'Overall, {overall_macro_avg:.1f}\n')
    print(f'Overall, {overall_macro_avg:.1f}')
