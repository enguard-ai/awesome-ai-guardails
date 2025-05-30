# Awesome AI Guardrails

A curated list of awesome AI guardrails.

## Categories

### Risk Categories

- `all`
- `hallucination`
- `nsfw`
- `factuality`
- `bias`
- `language-detection`/`language-translation`
- `pii`
- `deepfake-detection`
- `miscellaneous`

### Model Categories

- `all`
- `text-classification`
- `text-to-text-generation`
- `image-classification`
- `token-classification`

## Models

| Name | Size | Task | Category | Description |
|------|------|------|----------|-------------|
| [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) | `0.1B` | `image-classification` | `nsfw` | A programming language for controlling large language models |
| [Marqo/nsfw-image-detection-384](https://huggingface.co/Marqo/nsfw-image-detection-384) | `0.006B` | `image-classification` | `nsfw` |  lightweight image classification model designed to identify NSFW images. |
| [Freepik/nsfw_image_detector](https://huggingface.co/Freepik/nsfw_image_detector?not-for-all-audiences=true) | `0.086B` | `image-classification` | `nsfw` | This model is a vision transformer based on the EVA architecture, fine-tuned for NSFW content classification |
| [Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector) | `0.086B` | `image-classification` | `deepfake-detection` | Classifies images as real or deepfake with a focus on generated images by SDXL model, fine-tuned for deepfake detection |
| [prithivMLmods/Deep-Fake-Detector-v2-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) | `0.086B` | `image-classification` | `deepfake-detection` | Classifies images as real or deepfake with a focus on generated images by SDXL model, fine-tuned for deepfake detection |
| [bespokelabs/Bespoke-MiniCheck-7B](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B) | `7B` | `text-classification` | `factuality` | Bespoke-MiniCheck is a 7B parameter model that can detect hallucinations in text. |
| [lytang/MiniCheck-Flan-T5-Large](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large) | `0.8B` | `text-classification` | `factuality` | MiniCheck-Flan-T5-Large is a 0.8B parameter model that can detect hallucinations in text. |
| [lytang/MiniCheck-RoBERTa-Large](https://huggingface.co/lytang/MiniCheck-RoBERTa-Large) | `0.4B` | `text-classification` | `factuality` | MiniCheck-RoBERTa-large is a 0.4B parameter model that can detect hallucinations in text. |
| [eliasalbouzidi/distilbert-nsfw-text-classifier](https://huggingface.co/eliasalbouzidi/distilbert-nsfw-text-classifier) | `0.0068B` | `text-classification` | `nsfw` | DistilBERT-NSFW-Text-Classifier is a 0.0068B parameter model that can detect NSFW text. |
| [d4data/bias-detection-model](https://huggingface.co/d4data/bias-detection-model) | `0.3B` | `text-classification` | `bias` | Bias Detection Model is a 268M parameter model that can detect bias in text.|
| [MoritzLaurer/ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) | `0.4B` | `text-classification` | `miscellaneous` | ModernBERT-large-zeroshot-v2.0 is a 0.4B parameter model that can classify text into a random set of categories. |
| [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | `0.6B` | `text-to-text-generation` | `language-detection` | Language Detection and generation model is a 600M parameter model that can detect language in text and generate text in the expected language.|
| [ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii](https://huggingface.co/ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii) | `0.15B` | `token-classification` | `pii` | Llama-ai4privacy-multilingual-categorical-anonymiser-openpii is a 7B parameter model that can anonymize PII in text. |
| [gliner-community/gliner_small-v2.5](https://huggingface.co/gliner-community/gliner_small-v2.5) | `0.7B` | `token-classification` | `miscellaneous` | gliner_small-v2.5 is a 0.7B parameter model that can detect any tokens in text. |
| [osmosis-ai/Osmosis-Structure-0.6B](https://huggingface.co/osmosis-ai/Osmosis-Structure-0.6B) | `0.6B` | `token-classification` | `miscellaneous` | Osmosis-Structure-0.6B is a 0.6B parameter model that can extract any tokens from text. |

## Libraries

| Name | Category | Description |
|------|----------|-------------|
| [guardrails](https://github.com/guardrails-ai/guardrails) | `all` | Adding guardrails to large language models. |
| [NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | `all` | NeMo Guardrails is an open-source toolkit for easily adding programmable guardrails to LLM-based conversational systems. |
| [uqlm](https://github.com/cvs-health/uqlm) | `hallucination` | UQLM: Uncertainty Quantification for Language Models, is a Python package for UQ-based LLM hallucination detection. |

## Datasets

| Name | Category | Description |
|------|----------|-------------|
| [lytang/LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact) | `factuality` | Bias in Bios is a dataset of 100000 bios of people with different biases. |
| [Entreprise PII Masking](https://huggingface.co/collections/ai4privacy/entreprise-pii-masking-68255aab0ad327ba71f3210f) | `pii` | Entreprise PII Masking are datasets for enterprise PII masking focused on location, work, health, digital and financial information. |
| [prithivMLmods/OpenDeepfake-Preview](https://huggingface.co/datasets/prithivMLmods/OpenDeepfake-Preview) | `deepfake-detection` | OpenDeepfake-Preview is a dataset of 20K deepfake images. |

## Papers

| Name | Category | Description |
|------|----------|-------------|
| [Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers](https://arxiv.org/abs/2504.19254) | `hallucination` | Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers |
| [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://arxiv.org/pdf/2401.00396) | `factuality` | RAGTruth is a dataset of 100000 bios of people with different biases. |
| [MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents](https://arxiv.org/pdf/2404.10774) | `factuality` | how to build small fact-checking models that have GPT-4-level performance but for 400x lower cost. |
