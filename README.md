# Critical Period-inspired Language Model (CPLM)
 This repository for our ACL2025 paper "[Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition](https://aclanthology.org/2025.acl-long.462/)"

## ChangeLog (Nov 2025)
- Released the pre-trained models used in the experiments and evaluation scripts on Zorro.
- The scripts used for model training are currently under maintenance and temporarily unavailable.

## 0. Prerequirments
 - python==3.10.12
 - uv==0.6.3
 - torch==1.12.1+cu113
 
 
 ## 1. Clone this repository
 
 ```bash
 git clone https://github.com/osekilab/CPLM.git
 cd CPLM
 ```
 ## 2. Create a virtual environment
 
 ```bash
 uv venv .venv
 source .venv/bin/activate
 uv pip sync --requirement uv.lock
 ```
 
 ## 3. Install the customized transformers in editable mode
 ```bash
 uv pip install -e ./transformers
 ```



## 4. Evaluation with the pre-trained models
For reproducibility, we release the pre-trained model used in the experiment (trained on AO-CHILDES) and report the Zorro performance for each seed.

* [All the pre-trained models with AO-CHILDES](https://drive.google.com/drive/folders/1An5KEoLAHrfjeIZ0ye9PUm2jPUaHjL2C?usp=drive_link)

Regarding the evaluation script, it was created by modifying [UnMasked](https://github.com/phueb/UnMasked/) library's `score_model_from_repo.py` to change it from MLM to CLM. 
After the initial setup of the original **Unmasked** repository is complete, replace and add the files listed below. Then, execute the subsequent command to perform a detailed evaluation of the pre-trained model **Zorro**.

* `unmasked/scripts/score_model_from_repo.py`
* `unmasked/clm/scoring_device.py`
* `unmasked/utils_clm.py`

 
 ```bash
cd UnMasked
python scripts/score_model_from_repo.py --model_dir  models/seed0/DynamicLimit-Exp --output output/
 ```



## Benchmark results on Zorro
### Seed = 0
| Grammar Items | NoLimit | StaticLimit | DynamicLimit-Linear | DynamicLimit-Exp |
| :--- | :--- | :--- | :--- | :--- |
| D-N AGR | 49.6 | 50.3 | 51.0 | 50.7 |
| S-V AGR | 50.2 | 50.3 | 50.1 | 50.2 |
| ANA.AGR | 47.2 | 47.2 | 47.4 | 47.7 |
| ARG.STR | 41.1 | 41.9 | 57.6 | 63.9 |
| BINDING | 57.8 | 56.3 | 56.9 | 54.8 |
| CASE | 35.6 | 36.3 | 65.8 | 85.6 |
| ELLIPSIS | 78.7 | 77.6 | 41.3 | 66.6 |
| FILLER.GAP | 95.3 | 91.1 | 88.5 | 90.2 |
| IRREGULAR | 57.1 | 57.9 | 53.8 | 52.1 |
| ISLAND | 63.8 | 64.2 | 57.1 | 55.7 |
| LOCAL.ATR | 59.2 | 53.7 | 51.8 | 52.4 |
| NPI | 34.9 | 34.4 | 83.6 | 83.9 |
| QUANTIFIERS | 27.9 | 32.4 | 26.8 | 26.1 |
| **Overall** | **53.7** | **53.3** | **56.3** | **57.7** |

### Seed = 64
| Grammar Items | NoLimit | StaticLimit | DynamicLimit-Linear | DynamicLimit-Exp |
| :--- | :--- | :--- | :--- | :--- |
| D-N AGR | 50.3 | 50.8 | 51.3 | 50.8 |
| S-V AGR | 48.9 | 49.4 | 48.6 | 49.3 |
| ANA.AGR | 47.2 | 47.2 | 47.2 | 47.2 |
| ARG.STR | 43.7 | 39.8 | 66.9 | 70.9 |
| BINDING | 68.1 | 67.0 | 67.6 | 63.5 |
| CASE | 91.2 | 86.0 | 100.0 | 100.0 |
| ELLIPSIS | 62.3 | 58.2 | 49.9 | 47.2 |
| FILLER.GAP | 52.7 | 54.3 | 95.7 | 97.4 |
| IRREGULAR | 52.6 | 52.6 | 53.2 | 51.2 |
| ISLAND | 64.8 | 67.2 | 57.7 | 52.2 |
| LOCAL.ATR | 38.2 | 37.8 | 39.3 | 44.0 |
| NPI | 90.3 | 92.4 | 85.0 | 87.1 |
| QUANTIFIERS | 53.4 | 60.2 | 82.4 | 83.6 |
| **Overall** | **58.7** | **58.7** | **65.0** | **65.0** |

### Seed = 128
| Grammar Items | NoLimit | StaticLimit | DynamicLimit-Linear | DynamicLimit-Exp |
| :--- | :--- | :--- | :--- | :--- |
| D-N AGR | 49.5 | 49.4 | 50.8 | 50.8 |
| S-V AGR | 50.1 | 50.0 | 50.2 | 50.4 |
| ANA.AGR | 55.3 | 55.0 | 53.9 | 53.8 |
| ARG.STR | 49.6 | 51.5 | 68.4 | 68.3 |
| BINDING | 59.6 | 58.1 | 56.5 | 57.8 |
| CASE | 85.7 | 88.6 | 100.0 | 100.0 |
| ELLIPSIS | 78.8 | 78.3 | 52.0 | 45.6 |
| FILLER.GAP | 68.3 | 78.8 | 88.2 | 93.2 |
| IRREGULAR | 45.4 | 46.1 | 52.0 | 53.4 |
| ISLAND | 56.5 | 57.2 | 56.2 | 52.9 |
| LOCAL.ATR | 43.8 | 44.5 | 52.7 | 57.6 |
| NPI | 36.6 | 36.5 | 84.3 | 83.9 |
| QUANTIFIERS | 62.4 | 64.4 | 61.1 | 63.2 |
| **Overall** | **57.1** | **58.3** | **63.6** | **63.9** |



# Citation
If you use our code for your work, please cite:

```
@inproceedings{mita-etal-2025-developmentally,
    title = "Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition",
    author = "Mita, Masato  and
      Yoshida, Ryo  and
      Oseki, Yohei",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.462/",
    pages = "9386--9399",
    ISBN = "979-8-89176-251-0",
    abstract = "Large language models possess general linguistic abilities but acquire language less efficiently than humans. This study proposes a method for integrating the developmental characteristics of working memory during the critical period, a stage when human language acquisition is particularly efficient, into the training process of language models. The proposed method introduces a mechanism that initially constrains working memory during the early stages of training and gradually relaxes this constraint in an exponential manner as learning progresses. Targeted syntactic evaluation shows that the proposed method outperforms conventional methods without memory constraints or with static memory constraints. These findings not only provide new directions for designing data-efficient language models but also offer indirect evidence supporting the role of the developmental characteristics of working memory as the underlying mechanism of the critical period in language acquisition."
}
```

