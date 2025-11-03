from contextlib import contextmanager
import logging
import sys
import numpy as np
from typing import List
import random
import os
from pathlib import Path

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

SEED = 0

def clm_score_model_on_paradigm(model, tokenizer: GPT2Tokenizer, path_paradigm: Path, lower_case: bool, device: torch.device) -> List[float]:
    model.to(device)
    model.eval()

    # Load and preprocess corpus
    infile = path_paradigm.open('r')
    lines = infile.readlines()
    if lower_case:
        lines = [line.lower() for line in lines]

    scores = []

    for line in lines:
#        print(line)
        input_ids = tokenizer.encode(line, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            score = -loss.item()  # Negative log-likelihood for scoring
            scores.append(score)

    return scores

# Example usage:
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# path_paradigm = Path("/path/to/your/paradigm.txt")
# scores = clm_score_model_on_paradigm(model, tokenizer, path_paradigm, lower_case=True, device=device)
# print(scores)
