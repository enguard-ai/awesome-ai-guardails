# Awesome AI Guardrails

A curated list of awesome AI guardrails.

## Categories

### Main Categories

| Name | Description | Sub Categories |
|------|-------------|----------------|
| `security-and-privacy` | Security and privacy guardrails ensure content remains safe, ethical, and devoid of offensive material | inappropriate-content, offensive-language, prompt-injection, sensitive-content, deepfake-detection, pii |
| `response-and-relevance` | Ensures model responses are accurate, focused, and aligned with user intent | relevance, prompt-address, url-validation, factuality, refusal |
| `language-quality` | Ensures high standards of readability, coherence, and clarity | quality, translation-accuracy, duplicate-elimination, readability |
| `content-validation` | Ensures factual correctness and logical coherence of content | competitor-blocking, price-validation, source-verification, gibberish-filter |
| `logic-validation` | Ensures logical and functional correctness of generated code and data | sql-validation, api-validation, json-validation, logical-consistency |

### Sub Categories

| Category | Sub Category | Description |
|--------------|--------------|-------------|
| `security-and-privacy | inappropriate-content` | Detects and filters inappropriate or explicit content |
| `security-and-privacy | offensive-language` | Identifies and filters profane or offensive language |
| `security-and-privacy | prompt-injection` | Prevents manipulation attempts through malicious prompts |
| `security-and-privacy | sensitive-content` | Flags culturally, politically, or socially sensitive topics |
| `security-and-privacy | deepfake-detection` | Detects and filters deepfake content |
| `security-and-privacy | pii` | Identifies and filters personally identifiable information |
| `response-and-relevance | relevance` | Validates semantic relevance between input and output |
| `response-and-relevance | prompt-address` | Confirms response correctly addresses user's prompt |
| `response-and-relevance | url-validation` | Verifies validity of generated URLs |
| `response-and-relevance | factuality` | Cross-references content with external knowledge sources |
| `response-and-relevance | refusal` | Refuses to answer questions that are not appropriate or relevant |
| `language-quality | quality` | Assesses structure, relevance, and coherence of output |
| `language-quality | translation-accuracy` | Ensures contextually correct and linguistically accurate translations |
| `language-quality | duplicate-elimination` | Detects and removes redundant content |
| `language-quality | readability` | Evaluates text complexity for target audience |
| `content-validation | competitor-blocking` | Screens for mentions of rival brands or companies |
| `content-validation | price-validation` | Validates price-related data against verified sources |
| `content-validation | source-verification` | Verifies accuracy of external quotes and references |
| `content-validation | gibberish-filter` | Identifies and filters nonsensical or incoherent outputs |
| `logic-validation | sql-validation` | Validates SQL queries for syntax and security |
| `logic-validation | api-validation` | Ensures API calls conform to OpenAPI standards |
| `logic-validation | json-validation` | Validates JSON structure and schema |
| `logic-validation | logical-consistency` | Checks for contradictory or illogical statements |

## Models

| Name | Size | Task | Category | Sub Category | Description |
|------|------|------|----------|--------------|-------------|
| [s-nlp/mdistilbert-base-formality-ranker](https://huggingface.co/s-nlp/mdistilbert-base-formality-ranker) | `0.142B` | `text-classification` | `['content-validation']` | `['quality']` | s-nlp/mdistilbert-base-formality-ranker is a 0.142B parameter model that can detect the quality of the response. |
| [d4data/bias-detection-model](https://huggingface.co/d4data/bias-detection-model) | `0.3B` | `text-classification` | `['content-validation']` | `['bias']` | d4data/bias-detection-model is a 268M parameter model that can detect bias in text. |
| [NousResearch/Minos-v1](https://huggingface.co/NousResearch/Minos-v1) | `0.4B` | `text-classification` | `['content-validation']` | `['refusal']` | NousResearch/Minos-v1 is a 0.4B parameter model that can detect inappropriate-content text. |
| [osmosis-ai/Osmosis-Structure-0.6B](https://huggingface.co/osmosis-ai/Osmosis-Structure-0.6B) | `0.6B` | `token-classification` | `['content-validation', 'security-and-privacy']` | `['pii', 'competitor-blocking']` | osmosis-ai/Osmosis-Structure-0.6B is a 0.6B parameter model that can extract any tokens from text. |
| [gliner-community/gliner_small-v2.5](https://huggingface.co/gliner-community/gliner_small-v2.5) | `0.7B` | `token-classification` | `['content-validation', 'security-and-privacy']` | `['pii', 'competitor-blocking']` | gliner-community/gliner_small-v2.5 is a 0.7B parameter model that can detect any tokens in text. |
| [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | `0.6B` | `text-to-text-generation` | `['language-quality']` | `['translation-accuracy']` | facebook/nllb-200-distilled-600M is a 600M parameter model that can detect language in text and generate text in the expected language. |
| [protectai/distilroberta-base-rejection-v1](https://huggingface.co/protectai/distilroberta-base-rejection-v1) | `0.0821B` | `text-classification` | `['response-and-relevance']` | `['rejection']` | protectai/distilroberta-base-rejection-v1 is a 0.0821B parameter model that can detect rejection in text. |
| [s-nlp/E5-EverGreen-Multilingual-Small](https://huggingface.co/s-nlp/E5-EverGreen-Multilingual-Small) | `0.118B` | `text-classification` | `['response-and-relevance']` | `['factuality']` | s-nlp/E5-EverGreen-Multilingual-Small is a 0.118B parameter model that can detect temporal risks in text. |
| [lytang/MiniCheck-RoBERTa-Large](https://huggingface.co/lytang/MiniCheck-RoBERTa-Large) | `0.4B` | `text-classification` | `['response-and-relevance']` | `['factuality', 'logical-consistency', 'relevance']` | lytang/MiniCheck-RoBERTa-Large is a 0.4B parameter model that can detect hallucinations. |
| [lytang/MiniCheck-Flan-T5-Large](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large) | `0.8B` | `text-classification` | `['response-and-relevance']` | `['factuality', 'logical-consistency', 'relevance']` | lytang/MiniCheck-Flan-T5-Large is a 0.8B parameter model that can detect hallucinations. |
| [ibm-granite/granite-guardian-3.1-2b](https://huggingface.co/ibm-granite/granite-guardian-3.1-2b) | `2B` | `text-classification` | `['response-and-relevance']` | `['factuality', 'logical-consistency', 'relevance']` | ibm-granite/granite-guardian-3.1-2b is a 2B parameter model that can detect hallucinations in text. |
| [bespokelabs/Bespoke-MiniCheck-7B](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B) | `7B` | `text-classification` | `['response-and-relevance']` | `['factuality', 'logical-consistency', 'relevance']` | bespokelabs/Bespoke-MiniCheck-7B is a 7B parameter model that can detect hallucinations in text. |
| [nvidia/prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) | `0.184B` | `text-classification` | `['response-and-relevance', 'language-quality']` | `['relevance', 'quality']` | nvidia/prompt-task-and-complexity-classifier is a 0.184B parameter model that can detect hallucinations in text. |
| [PatronusAI/glider](https://huggingface.co/PatronusAI/glider) | `3.8B` | `text-classification` | `['response-and-relevance', 'language-quality']` | `['factuality', 'logical-consistency', 'relevance', 'quality']` | PatronusAI/glider is a 3.8B parameter model that can judge and evaluate the quality of the response. |
| [flowaicom/Flow-Judge-v0.1](https://huggingface.co/flowaicom/Flow-Judge-v0.1) | `3.8B` | `text-classification` | `['response-and-relevance', 'language-quality']` | `['factuality', 'logical-consistency', 'relevance', 'quality']` | flowaicom/Flow-Judge-v0.1 is a 3.8B parameter model that can judge and evaluate the quality of the response. |
| [Marqo/nsfw-image-detection-384](https://huggingface.co/Marqo/nsfw-image-detection-384) | `0.006B` | `image-classification` | `['security-and-privacy']` | `['inappropriate-content']` | Marqo/nsfw-image-detection-384 is a 0.006B parameter model that can detect NSFW images. |
| [Freepik/nsfw_image_detector](https://huggingface.co/Freepik/nsfw_image_detector?not-for-all-audiences=true) | `0.086B` | `image-classification` | `['security-and-privacy']` | `['inappropriate-content']` | Freepik/nsfw_image_detector is a 0.086B parameter model that can detect NSFW images. |
| [Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector) | `0.086B` | `image-classification` | `['security-and-privacy']` | `['deepfake-detection']` | Organika/sdxl-detector is a 0.086B parameter model that can detect deepfake images. |
| [prithivMLmods/Deep-Fake-Detector-v2-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) | `0.086B` | `image-classification` | `['security-and-privacy']` | `['deepfake-detection']` | prithivMLmods/Deep-Fake-Detector-v2-Model is a 0.086B parameter model that can detect deepfake images. |
| [TostAI/nsfw-image-detection-large](https://huggingface.co/TostAI/nsfw-image-detection-large) | `0.0871B` | `image-classification` | `['security-and-privacy']` | `['inappropriate-content']` | TostAI/nsfw-image-detection-large is a 0.0871B parameter model that can detect NSFW images. |
| [Ateeqq/nsfw-image-detection](https://huggingface.co/Ateeqq/nsfw-image-detection) | `0.092B` | `image-classification` | `['security-and-privacy']` | `['inappropriate-content']` | Ateeqq/nsfw-image-detection is a 0.092B parameter model that can detect NSFW images. |
| [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) | `0.1B` | `image-classification` | `['security-and-privacy']` | `['inappropriate-content']` | Falconsai/nsfw_image_detection is a 0.1B parameter model that can detect NSFW images. |
| [OpenSafetyLab/ImageGuard](https://huggingface.co/OpenSafetyLab/ImageGuard) | `na` | `image-classification` | `['security-and-privacy']` | `['inappropriate-content']` | OpenSafetyLab/ImageGuard is a model that can detect NSFW images. |
| [meta-llama/Llama-Guard-4-12B](https://huggingface.co/meta-llama/Llama-Guard-4-12B) | `12B` | `image-text-to-text` | `['security-and-privacy']` | `['inappropriate-content', 'offensive-language', 'prompt-injection', 'sensitive-content']` | meta-llama/Llama-Guard-4-12B is a 12B parameter model that can detect harmful content. |
| [meta-llama/Llama-Prompt-Guard-2-22M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M) | `0.022B` | `text-classification` | `['security-and-privacy']` | `['prompt-injection', 'jailbreaks']` | meta-llama/Llama-Prompt-Guard-2-22M is a 0.022B parameter model that can detect prompt injection in text. |
| [eliasalbouzidi/distilbert-nsfw-text-classifier](https://huggingface.co/eliasalbouzidi/distilbert-nsfw-text-classifier) | `0.068B` | `text-classification` | `['security-and-privacy']` | `['inappropriate-content']` | eliasalbouzidi/distilbert-nsfw-text-classifier is a 0.068B parameter model that can detect NSFW text. |
| [meta-llama/Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | `0.086B` | `text-classification` | `['security-and-privacy']` | `['prompt-injection', 'jailbreaks']` | meta-llama/Llama-Prompt-Guard-2-86M is a 0.086B parameter model that can detect prompt injection in text. |
| [ibm-granite/granite-guardian-hap-125m](https://huggingface.co/ibm-granite/granite-guardian-hap-125m) | `0.125B` | `text-classification` | `['security-and-privacy']` | `['toxicity', 'hallucination']` | ibm-granite/granite-guardian-hap-125m is a 0.125B parameter model that can detect toxicity in text. |
| [ibm-granite/granite-guardian-hap-125m](https://huggingface.co/ibm-granite/granite-guardian-hap-125m) | `0.125B` | `text-classification` | `['security-and-privacy']` | `['toxicity', 'hallucination']` | ibm-granite/granite-guardian-hap-125m is a 0.125B parameter model that can detect toxicity in text. |
| [protectai/deberta-v3-small-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-small-prompt-injection-v2) | `0.142B` | `text-classification` | `['security-and-privacy']` | `['prompt-injection']` | protectai/deberta-v3-small-prompt-injection-v2 is a 0.184B parameter model that can detect prompt injection in text. |
| [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | `0.182B` | `text-classification` | `['security-and-privacy']` | `['prompt-injection']` | protectai/deberta-v3-base-prompt-injection-v2 is a 0.182B parameter model that can detect prompt injection in text. |
| [TostAI/nsfw-text-detection-large](https://huggingface.co/TostAI/nsfw-text-detection-large) | `0.355B` | `text-classification` | `['security-and-privacy']` | `['inappropriate-content']` | TostAI/nsfw-text-detection-large is a 0.355B parameter model that can detect NSFW text. |
| [MoritzLaurer/ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) | `0.4B` | `text-classification` | `['security-and-privacy']` | `['inappropriate-content', 'offensive-language', 'prompt-injection', 'sensitive-content']` | MoritzLaurer/ModernBERT-large-zeroshot-v2.0 is a 0.4B parameter model that can classify text into a random set of categories. |
| [madhurjindal/Jailbreak-Detector-2-XL](https://huggingface.co/madhurjindal/Jailbreak-Detector-2-XL) | `0.5B` | `text-classification` | `['security-and-privacy']` | `['jailbreaks']` | madhurjindal/Jailbreak-Detector-2-XL is a 0.5B adapter that can detect jailbreak in text. |
| [google/shieldgemma-2b](https://huggingface.co/google/shieldgemma-2b) | `2B` | `text-classification` | `['security-and-privacy']` | `['inappropriate-content', 'offensive-language', 'prompt-injection', 'sensitive-content']` | google/shieldgemma-2b is a 2B parameter model that can detect any risks you provide |
| [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | `1B` | `text-to-text-generation` | `['security-and-privacy']` | `['inappropriate-content', 'offensive-language', 'prompt-injection', 'sensitive-content']` | meta-llama/Llama-3.2-1B-Instruct is a 1B parameter model that can detect harmful content. |
| [ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii](https://huggingface.co/ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii) | `0.15B` | `token-classification` | `['security-and-privacy']` | `['pii']` | ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii is a 7B parameter model that can anonymize PII in text. |

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
| [eliasalbouzidi/NSFW-Safe-Dataset](https://huggingface.co/datasets/eliasalbouzidi/NSFW-Safe-Dataset?not-for-all-audiences=true) | `nsfw` | NSFW-Safe-Dataset is a dataset for NSFW content detection. |
| [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) | `toxic-chat` | Toxic-Chat is a dataset for toxic chat detection. |

## Papers

| Name | Category | Description |
|------|----------|-------------|
| [Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers](https://arxiv.org/abs/2504.19254) | `hallucination` | Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers |
| [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://arxiv.org/pdf/2401.00396) | `factuality` | RAGTruth is a dataset of 100000 bios of people with different biases. |
| [MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents](https://arxiv.org/pdf/2404.10774) | `factuality` | how to build small fact-checking models that have GPT-4-level performance but for 400x lower cost. |
| [A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions](https://arxiv.org/pdf/2311.05232) | `hallucination` | A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions |
| [Granite Guardian: A Guardrail Framework for Large Language Models](https://arxiv.org/abs/2412.07724) | `all` | Granite Guardian is a guardrail framework for large language models. |
| ["Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models](https://arxiv.org/abs/2308.03825) | `prompt-injection` | "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models |
| ["Tiny-Toxic-Detector: A compact transformer-based model for toxic content detection](https://arxiv.org/abs/2409.02114) | `toxic-chat` | "Tiny-Toxic-Detector: A compact transformer-based model for toxic content detection |
| [T2ISafety: Benchmark for Assessing Fairness, Toxicity, and Privacy in Image Generation](https://arxiv.org/abs/2501.12612) | `toxic-chat` | T2ISafety is a benchmark for assessing fairness, toxicity, and privacy in image generation. |