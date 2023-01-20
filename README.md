# Lightweight Spanish Language Models

[ALBETO and DistilBETO: Lightweight Spanish Language Models](https://arxiv.org/abs/2204.09145)

**ALBETO** and **DistilBETO** are versions of [ALBERT](https://github.com/google-research/albert) and [DistilBERT](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) pre-trained exclusively on [Spanish corpora](https://github.com/josecannete/spanish-corpora). We train several versions of ALBETO ranging from 5M to 223M parameters and one of DistilBETO with 67M parameters.

All models (pre-trained and fine-tuned) can be found on [our organization on the HuggingFace Hub](https://huggingface.co/dccuchile).

The following tables show the results of every model in different evaluation tasks with links to the model on the HuggingFace Hub.

# Pre-trained models

| Model          | Parameters | Evaluation Average | Size  | Performance |
|----------------|------------|--------------------|-------|-------------|
| [BETO *uncased*](https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased)   | 110M       | 77.48                  | 1x    | 0.95x           |
| [BETO *cased*](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)     | 110M       | 81.02                  | 1x    | 1x           |
| [DistilBETO](https://huggingface.co/CenIA/distillbert-base-spanish-uncased)     | 67M        | 73.22                  | 1.64x | 0.90x           |
| [ALBETO *tiny*](https://huggingface.co/CenIA/albert-tiny-spanish)    | 5M         | 70.86                  | 22x   | 0.87x           |
| [ALBETO *base*](https://huggingface.co/CenIA/albert-base-spanish)    | 12M        | 79.35                  | 9.16x | 0.97x           |
| [ALBETO *large*](https://huggingface.co/CenIA/albert-large-spanish)   | 18M        | 78.12                  | 6.11x | 0.96x           |
| [ALBETO *xlarge*](https://huggingface.co/CenIA/albert-xlarge-spanish)  | 59M        | 80.20                  | 1.86x | 0.98x           |
| [ALBETO *xxlarge*](https://huggingface.co/CenIA/albert-xxlarge-spanish) | 223M       | 81.34                  | 0.49x | 1x           |

# Fine-tuned models

## POS / NER

|                | POS       | NER       |
|----------------|-----------|-----------|
| BETO *uncased*   | [97.70](https://huggingface.co/CenIA/bert-base-spanish-wwm-uncased-finetuned-pos)     | [83.76](https://huggingface.co/CenIA/bert-base-spanish-wwm-uncased-finetuned-ner)     |
| BETO *cased*     | [**98.84**](https://huggingface.co/CenIA/bert-base-spanish-wwm-cased-finetuned-pos) | [**88.24**](https://huggingface.co/CenIA/bert-base-spanish-wwm-cased-finetuned-ner) |
| DistilBETO     | [97.50](https://huggingface.co/CenIA/distillbert-base-spanish-uncased-finetuned-pos)     | [81.19](https://huggingface.co/CenIA/distillbert-base-spanish-uncased-finetuned-ner)     |
| ALBETO *tiny*    | [97.04](https://huggingface.co/CenIA/albert-tiny-spanish-finetuned-pos)     | [75.11](https://huggingface.co/CenIA/albert-tiny-spanish-finetuned-ner)     |
| ALBETO *base*    | [98.08](https://huggingface.co/CenIA/albert-base-spanish-finetuned-pos)     | [83.35](https://huggingface.co/CenIA/albert-base-spanish-finetuned-ner)     |
| ALBETO *large*   | [97.87](https://huggingface.co/CenIA/albert-large-spanish-finetuned-pos)     | [83.72](https://huggingface.co/CenIA/albert-large-spanish-finetuned-ner)     |
| ALBETO *xlarge*  | [98.06](https://huggingface.co/CenIA/albert-xlarge-spanish-finetuned-pos)     | [82.30](https://huggingface.co/CenIA/albert-xlarge-spanish-finetuned-ner)     |
| ALBETO *xxlarge* | [98.35](https://huggingface.co/CenIA/albert-xxlarge-spanish-finetuned-pos)     | [84.36](https://huggingface.co/CenIA/albert-xxlarge-spanish-finetuned-ner)     |


## MLDoc / PAWS-X / XNLI

|                | MLDoc     | PAWS-X    | XNLI      |
|----------------|-----------|-----------|-----------|
| BETO *uncased*   | [96.38](https://huggingface.co/CenIA/bert-base-spanish-wwm-uncased-finetuned-mldoc)     | [84.25](https://huggingface.co/CenIA/bert-base-spanish-wwm-uncased-finetuned-pawsx)     | [77.76](https://huggingface.co/CenIA/bert-base-spanish-wwm-uncased-finetuned-xnli)     |
| BETO *cased*     | [96.65](https://huggingface.co/CenIA/bert-base-spanish-wwm-cased-finetuned-mldoc)     | [89.80](https://huggingface.co/CenIA/bert-base-spanish-wwm-cased-finetuned-pawsx)     | [81.98](https://huggingface.co/CenIA/bert-base-spanish-wwm-cased-finetuned-xnli)     |
| DistilBETO     | [96.35](https://huggingface.co/CenIA/distillbert-base-spanish-uncased-finetuned-mldoc)     | [75.80](https://huggingface.co/CenIA/distillbert-base-spanish-uncased-finetuned-pawsx)     | [76.59](https://huggingface.co/CenIA/distillbert-base-spanish-uncased-finetuned-xnli)     |
| ALBETO *tiny*    | [95.82](https://huggingface.co/CenIA/albert-tiny-spanish-finetuned-mldoc)     | [80.20](https://huggingface.co/CenIA/albert-tiny-spanish-finetuned-pawsx)     | [73.43](https://huggingface.co/CenIA/albert-tiny-spanish-finetuned-xnli)     |
| ALBETO *base*    | [96.07](https://huggingface.co/CenIA/albert-base-spanish-finetuned-mldoc)     | [87.95](https://huggingface.co/CenIA/albert-base-spanish-finetuned-pawsx)     | [79.88](https://huggingface.co/CenIA/albert-base-spanish-finetuned-xnli)     |
| ALBETO *large*   | [92.22](https://huggingface.co/CenIA/albert-large-spanish-finetuned-mldoc)     | [86.05](https://huggingface.co/CenIA/albert-large-spanish-finetuned-pawsx)     | [78.94](https://huggingface.co/CenIA/albert-large-spanish-finetuned-xnli)     |
| ALBETO *xlarge*  | [95.70](https://huggingface.co/CenIA/albert-xlarge-spanish-finetuned-mldoc)     | [89.05](https://huggingface.co/CenIA/albert-xlarge-spanish-finetuned-pawsx)     | [81.68](https://huggingface.co/CenIA/albert-xlarge-spanish-finetuned-xnli)     |
| ALBETO *xxlarge* | [**96.85**](https://huggingface.co/CenIA/albert-xxlarge-spanish-finetuned-mldoc) | [**89.85**](https://huggingface.co/CenIA/albert-xxlarge-spanish-finetuned-pawsx) | [**82.42**](https://huggingface.co/CenIA/albert-xxlarge-spanish-finetuned-xnli) |

## QA

| Model          | MLQA          | SQAC          | TAR, XQuAD    |
|----------------|---------------|---------------|---------------|
| BETO *uncased*   | [64.12 / 40.83](https://huggingface.co/CenIA/bert-base-spanish-wwm-uncased-finetuned-qa-mlqa) | [72.22 / 53.45](https://huggingface.co/CenIA/bert-base-spanish-wwm-uncased-finetuned-qa-sqac) | [74.81 / 54.62](https://huggingface.co/CenIA/bert-base-spanish-wwm-uncased-finetuned-qa-tar) |
| BETO *cased*     | [67.65 / 43.38](https://huggingface.co/CenIA/bert-base-spanish-wwm-cased-finetuned-qa-mlqa) | [78.65 / 60.94](https://huggingface.co/CenIA/bert-base-spanish-wwm-cased-finetuned-qa-sqac) | [77.81 / 56.97](https://huggingface.co/CenIA/bert-base-spanish-wwm-cased-finetuned-qa-tar) |
| DistilBETO     | [57.97 / 35.50](https://huggingface.co/CenIA/distillbert-base-spanish-uncased-finetuned-qa-mlqa) | [64.41 / 45.34](https://huggingface.co/CenIA/distillbert-base-spanish-uncased-finetuned-qa-sqac) | [66.97 / 46.55](https://huggingface.co/CenIA/distillbert-base-spanish-uncased-finetuned-qa-tar) |
| ALBETO *tiny*    | [51.84 / 28.28](https://huggingface.co/CenIA/albert-tiny-spanish-finetuned-qa-mlqa) | [59.28 / 39.16](https://huggingface.co/CenIA/albert-tiny-spanish-finetuned-qa-sqac) | [66.43 / 45.71](https://huggingface.co/CenIA/albert-tiny-spanish-finetuned-qa-tar) |
| ALBETO *base*    | [66.12 / 41.10](https://huggingface.co/CenIA/albert-base-spanish-finetuned-qa-mlqa) | [77.71 / 59.84](https://huggingface.co/CenIA/albert-base-spanish-finetuned-qa-sqac) | [77.18 / 57.05](https://huggingface.co/CenIA/albert-base-spanish-finetuned-qa-tar) |
| ALBETO *large*   | [65.56 / 40.98](https://huggingface.co/CenIA/albert-large-spanish-finetuned-qa-mlqa) | [76.36 / 56.54](https://huggingface.co/CenIA/albert-large-spanish-finetuned-qa-sqac) | [76.72 / 56.21](https://huggingface.co/CenIA/albert-large-spanish-finetuned-qa-tar) |
| ALBETO *xlarge*  | [68.26 / 43.76](https://huggingface.co/CenIA/albert-xlarge-spanish-finetuned-qa-mlqa) | [78.64 / 59.26](https://huggingface.co/CenIA/albert-xlarge-spanish-finetuned-qa-sqac) | [**80.15** / **59.66**](https://huggingface.co/CenIA/albert-xlarge-spanish-finetuned-qa-tar) |
| ALBETO *xxlarge* | [**70.17** / **45.99**](https://huggingface.co/CenIA/albert-xxlarge-spanish-finetuned-qa-mlqa) | [**81.49** / **62.67**](https://huggingface.co/CenIA/albert-xxlarge-spanish-finetuned-qa-sqac) | [79.13 / 58.40](https://huggingface.co/CenIA/albert-xxlarge-spanish-finetuned-qa-tar) |

# Citation

[ALBETO and DistilBETO: Lightweight Spanish Language Models](https://arxiv.org/abs/2204.09145)

To cite this resource in a publication please use the following:

```
 @inproceedings{canete2022albeto,
   title="ALBETO and DistilBETO: Lightweight Spanish Language Models",
   author="Ca{\~n}ete, Jos{\'e} and Donoso, Sebasti{\'a}n and Bravo-Marquez, Felipe and Carvallo, Andr{\'e}s and Araujo, Vladimir",
   booktitle = "Proceedings of the 13th Language Resources and Evaluation Conference",
   year = "2022",
   address = "Marseille, France",
   publisher = "European Language Resources Association",
 }
 ```
 
