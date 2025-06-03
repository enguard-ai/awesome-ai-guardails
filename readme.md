# 🛡️ Awesome AI Guardrails 🛡️

![Awesome](https://awesome.re/badge.svg) ![MIT License](https://img.shields.io/badge/license-MIT-brightgreen)

A curated list of awesome AI guardrails.

If you find this list helpful, give it a ⭐ on GitHub, share it, and contribute by submitting a pull request or issue!

## Categories

### Main Categories

| Name | Description |
|------|-------------|
| `security-and-privacy` | Security and privacy guardrails ensure content remains safe, ethical, and devoid of offensive material |
| `response-and-relevance` | Ensures model responses are accurate, focused, and aligned with user intent |
| `language-quality` | Ensures high standards of readability, coherence, and clarity |
| `content-validation` | Ensures factual correctness and logical coherence of content |
| `logic-validation` | Ensures logical and functional correctness of generated code and data |

### Sub Categories


### security-and-privacy Sub-Categories

| Sub Category | Description |
|--------------|-------------|
| `inappropriate-content` | Detects and filters inappropriate or explicit content |
| `offensive-language` | Identifies and filters profane or offensive language |
| `prompt-injection` | Prevents manipulation attempts through malicious prompts |
| `sensitive-content` | Flags culturally, politically, or socially sensitive topics |
| `deepfake-detection` | Detects and filters deepfake content |
| `pii` | Identifies and filters personally identifiable information |

<details>
<summary>Models in security-and-privacy</summary>

| Name | Size | Task |
|------|------|------|
| [osmosis-ai/Osmosis-Structure-0.6B](https://huggingface.co/osmosis-ai/Osmosis-Structure-0.6B) | `0.6B` | `token-classification` |
| [gliner-community/gliner_small-v2.5](https://huggingface.co/gliner-community/gliner_small-v2.5) | `0.7B` | `token-classification` |
| [Marqo/nsfw-image-detection-384](https://huggingface.co/Marqo/nsfw-image-detection-384) | `0.006B` | `image-classification` |
| [Freepik/nsfw_image_detector](https://huggingface.co/Freepik/nsfw_image_detector?not-for-all-audiences=true) | `0.086B` | `image-classification` |
| [Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector) | `0.086B` | `image-classification` |
| [prithivMLmods/Deep-Fake-Detector-v2-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) | `0.086B` | `image-classification` |
| [TostAI/nsfw-image-detection-large](https://huggingface.co/TostAI/nsfw-image-detection-large) | `0.0871B` | `image-classification` |
| [Ateeqq/nsfw-image-detection](https://huggingface.co/Ateeqq/nsfw-image-detection) | `0.092B` | `image-classification` |
| [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) | `0.1B` | `image-classification` |
| [OpenSafetyLab/ImageGuard](https://huggingface.co/OpenSafetyLab/ImageGuard) | `na` | `image-classification` |
| [meta-llama/Llama-Guard-4-12B](https://huggingface.co/meta-llama/Llama-Guard-4-12B) | `12B` | `image-text-to-text` |
| [meta-llama/Llama-Prompt-Guard-2-22M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M) | `0.022B` | `text-classification` |
| [eliasalbouzidi/distilbert-nsfw-text-classifier](https://huggingface.co/eliasalbouzidi/distilbert-nsfw-text-classifier) | `0.068B` | `text-classification` |
| [meta-llama/Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | `0.086B` | `text-classification` |
| [ibm-granite/granite-guardian-hap-125m](https://huggingface.co/ibm-granite/granite-guardian-hap-125m) | `0.125B` | `text-classification` |
| [ibm-granite/granite-guardian-hap-125m](https://huggingface.co/ibm-granite/granite-guardian-hap-125m) | `0.125B` | `text-classification` |
| [protectai/deberta-v3-small-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-small-prompt-injection-v2) | `0.142B` | `text-classification` |
| [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | `0.182B` | `text-classification` |
| [TostAI/nsfw-text-detection-large](https://huggingface.co/TostAI/nsfw-text-detection-large) | `0.355B` | `text-classification` |
| [MoritzLaurer/ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) | `0.4B` | `text-classification` |
| [madhurjindal/Jailbreak-Detector-2-XL](https://huggingface.co/madhurjindal/Jailbreak-Detector-2-XL) | `0.5B` | `text-classification` |
| [google/shieldgemma-2b](https://huggingface.co/google/shieldgemma-2b) | `2B` | `text-classification` |
| [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | `1B` | `text-to-text-generation` |
| [ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii](https://huggingface.co/ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii) | `0.15B` | `token-classification` |
</details>


### response-and-relevance Sub-Categories

| Sub Category | Description |
|--------------|-------------|
| `relevance` | Validates semantic relevance between input and output |
| `prompt-address` | Confirms response correctly addresses user's prompt |
| `url-validation` | Verifies validity of generated URLs |
| `factuality` | Cross-references content with external knowledge sources |
| `refusal` | Refuses to answer questions that are not appropriate or relevant |

<details>
<summary>Models in response-and-relevance</summary>

| Name | Size | Task |
|------|------|------|
| [protectai/distilroberta-base-rejection-v1](https://huggingface.co/protectai/distilroberta-base-rejection-v1) | `0.0821B` | `text-classification` |
| [s-nlp/E5-EverGreen-Multilingual-Small](https://huggingface.co/s-nlp/E5-EverGreen-Multilingual-Small) | `0.118B` | `text-classification` |
| [lytang/MiniCheck-RoBERTa-Large](https://huggingface.co/lytang/MiniCheck-RoBERTa-Large) | `0.4B` | `text-classification` |
| [lytang/MiniCheck-Flan-T5-Large](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large) | `0.8B` | `text-classification` |
| [ibm-granite/granite-guardian-3.1-2b](https://huggingface.co/ibm-granite/granite-guardian-3.1-2b) | `2B` | `text-classification` |
| [bespokelabs/Bespoke-MiniCheck-7B](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B) | `7B` | `text-classification` |
| [nvidia/prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) | `0.184B` | `text-classification` |
| [PatronusAI/glider](https://huggingface.co/PatronusAI/glider) | `3.8B` | `text-classification` |
| [flowaicom/Flow-Judge-v0.1](https://huggingface.co/flowaicom/Flow-Judge-v0.1) | `3.8B` | `text-classification` |
</details>


### language-quality Sub-Categories

| Sub Category | Description |
|--------------|-------------|
| `quality` | Assesses structure, relevance, and coherence of output |
| `translation-accuracy` | Ensures contextually correct and linguistically accurate translations |
| `duplicate-elimination` | Detects and removes redundant content |
| `readability` | Evaluates text complexity for target audience |

<details>
<summary>Models in language-quality</summary>

| Name | Size | Task |
|------|------|------|
| [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) | `0.109B` | `text-classification` |
| [nvidia/quality-classifier-deberta](https://huggingface.co/nvidia/quality-classifier-deberta) | `0.184B` | `text-classification` |
| [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | `0.6B` | `text-to-text-generation` |
| [nvidia/prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) | `0.184B` | `text-classification` |
| [PatronusAI/glider](https://huggingface.co/PatronusAI/glider) | `3.8B` | `text-classification` |
| [flowaicom/Flow-Judge-v0.1](https://huggingface.co/flowaicom/Flow-Judge-v0.1) | `3.8B` | `text-classification` |
</details>


### content-validation Sub-Categories

| Sub Category | Description |
|--------------|-------------|
| `competitor-blocking` | Screens for mentions of rival brands or companies |
| `price-validation` | Validates price-related data against verified sources |
| `source-verification` | Verifies accuracy of external quotes and references |
| `gibberish-filter` | Identifies and filters nonsensical or incoherent outputs |

<details>
<summary>Models in content-validation</summary>

| Name | Size | Task |
|------|------|------|
| [s-nlp/mdistilbert-base-formality-ranker](https://huggingface.co/s-nlp/mdistilbert-base-formality-ranker) | `0.142B` | `text-classification` |
| [d4data/bias-detection-model](https://huggingface.co/d4data/bias-detection-model) | `0.3B` | `text-classification` |
| [NousResearch/Minos-v1](https://huggingface.co/NousResearch/Minos-v1) | `0.4B` | `text-classification` |
| [osmosis-ai/Osmosis-Structure-0.6B](https://huggingface.co/osmosis-ai/Osmosis-Structure-0.6B) | `0.6B` | `token-classification` |
| [gliner-community/gliner_small-v2.5](https://huggingface.co/gliner-community/gliner_small-v2.5) | `0.7B` | `token-classification` |
</details>


### logic-validation Sub-Categories

| Sub Category | Description |
|--------------|-------------|
| `sql-validation` | Validates SQL queries for syntax and security |
| `api-validation` | Ensures API calls conform to OpenAPI standards |
| `json-validation` | Validates JSON structure and schema |
| `logical-consistency` | Checks for contradictory or illogical statements |

## Models


### Text-Classification Models
| Name | Size | Category | Sub Category |
|------|------|----------|--------------|
| [s-nlp/mdistilbert-base-formality-ranker](https://huggingface.co/s-nlp/mdistilbert-base-formality-ranker) | `0.142B` | `['content-validation']` | `['quality']` |
| [d4data/bias-detection-model](https://huggingface.co/d4data/bias-detection-model) | `0.3B` | `['content-validation']` | `['bias']` |
| [NousResearch/Minos-v1](https://huggingface.co/NousResearch/Minos-v1) | `0.4B` | `['content-validation']` | `['refusal']` |
| [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) | `0.109B` | `['language-quality']` | `['quality']` |
| [nvidia/quality-classifier-deberta](https://huggingface.co/nvidia/quality-classifier-deberta) | `0.184B` | `['language-quality']` | `['quality']` |
| [protectai/distilroberta-base-rejection-v1](https://huggingface.co/protectai/distilroberta-base-rejection-v1) | `0.0821B` | `['response-and-relevance']` | `['rejection']` |
| [s-nlp/E5-EverGreen-Multilingual-Small](https://huggingface.co/s-nlp/E5-EverGreen-Multilingual-Small) | `0.118B` | `['response-and-relevance']` | `['factuality']` |
| [lytang/MiniCheck-RoBERTa-Large](https://huggingface.co/lytang/MiniCheck-RoBERTa-Large) | `0.4B` | `['response-and-relevance']` | `['factuality', 'logical-consistency', 'relevance']` |
| [lytang/MiniCheck-Flan-T5-Large](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large) | `0.8B` | `['response-and-relevance']` | `['factuality', 'logical-consistency', 'relevance']` |
| [ibm-granite/granite-guardian-3.1-2b](https://huggingface.co/ibm-granite/granite-guardian-3.1-2b) | `2B` | `['response-and-relevance']` | `['factuality', 'logical-consistency', 'relevance']` |
| [bespokelabs/Bespoke-MiniCheck-7B](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B) | `7B` | `['response-and-relevance']` | `['factuality', 'logical-consistency', 'relevance']` |
| [nvidia/prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) | `0.184B` | `['response-and-relevance', 'language-quality']` | `['relevance', 'quality']` |
| [PatronusAI/glider](https://huggingface.co/PatronusAI/glider) | `3.8B` | `['response-and-relevance', 'language-quality']` | `['factuality', 'logical-consistency', 'relevance', 'quality']` |
| [flowaicom/Flow-Judge-v0.1](https://huggingface.co/flowaicom/Flow-Judge-v0.1) | `3.8B` | `['response-and-relevance', 'language-quality']` | `['factuality', 'logical-consistency', 'relevance', 'quality']` |
| [meta-llama/Llama-Prompt-Guard-2-22M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M) | `0.022B` | `['security-and-privacy']` | `['prompt-injection', 'jailbreaks']` |
| [eliasalbouzidi/distilbert-nsfw-text-classifier](https://huggingface.co/eliasalbouzidi/distilbert-nsfw-text-classifier) | `0.068B` | `['security-and-privacy']` | `['inappropriate-content']` |
| [meta-llama/Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | `0.086B` | `['security-and-privacy']` | `['prompt-injection', 'jailbreaks']` |
| [ibm-granite/granite-guardian-hap-125m](https://huggingface.co/ibm-granite/granite-guardian-hap-125m) | `0.125B` | `['security-and-privacy']` | `['toxicity', 'hallucination']` |
| [ibm-granite/granite-guardian-hap-125m](https://huggingface.co/ibm-granite/granite-guardian-hap-125m) | `0.125B` | `['security-and-privacy']` | `['toxicity', 'hallucination']` |
| [protectai/deberta-v3-small-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-small-prompt-injection-v2) | `0.142B` | `['security-and-privacy']` | `['prompt-injection']` |
| [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | `0.182B` | `['security-and-privacy']` | `['prompt-injection']` |
| [TostAI/nsfw-text-detection-large](https://huggingface.co/TostAI/nsfw-text-detection-large) | `0.355B` | `['security-and-privacy']` | `['inappropriate-content']` |
| [MoritzLaurer/ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) | `0.4B` | `['security-and-privacy']` | `['inappropriate-content', 'offensive-language', 'prompt-injection', 'sensitive-content']` |
| [madhurjindal/Jailbreak-Detector-2-XL](https://huggingface.co/madhurjindal/Jailbreak-Detector-2-XL) | `0.5B` | `['security-and-privacy']` | `['jailbreaks']` |
| [google/shieldgemma-2b](https://huggingface.co/google/shieldgemma-2b) | `2B` | `['security-and-privacy']` | `['inappropriate-content', 'offensive-language', 'prompt-injection', 'sensitive-content']` |

### Token-Classification Models
| Name | Size | Category | Sub Category |
|------|------|----------|--------------|
| [osmosis-ai/Osmosis-Structure-0.6B](https://huggingface.co/osmosis-ai/Osmosis-Structure-0.6B) | `0.6B` | `['content-validation', 'security-and-privacy']` | `['pii', 'competitor-blocking']` |
| [gliner-community/gliner_small-v2.5](https://huggingface.co/gliner-community/gliner_small-v2.5) | `0.7B` | `['content-validation', 'security-and-privacy']` | `['pii', 'competitor-blocking']` |
| [ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii](https://huggingface.co/ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii) | `0.15B` | `['security-and-privacy']` | `['pii']` |

### Text-To-Text-Generation Models
| Name | Size | Category | Sub Category |
|------|------|----------|--------------|
| [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | `0.6B` | `['language-quality']` | `['translation-accuracy']` |
| [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | `1B` | `['security-and-privacy']` | `['inappropriate-content', 'offensive-language', 'prompt-injection', 'sensitive-content']` |

### Image-Classification Models
| Name | Size | Category | Sub Category |
|------|------|----------|--------------|
| [Marqo/nsfw-image-detection-384](https://huggingface.co/Marqo/nsfw-image-detection-384) | `0.006B` | `['security-and-privacy']` | `['inappropriate-content']` |
| [Freepik/nsfw_image_detector](https://huggingface.co/Freepik/nsfw_image_detector?not-for-all-audiences=true) | `0.086B` | `['security-and-privacy']` | `['inappropriate-content']` |
| [Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector) | `0.086B` | `['security-and-privacy']` | `['deepfake-detection']` |
| [prithivMLmods/Deep-Fake-Detector-v2-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) | `0.086B` | `['security-and-privacy']` | `['deepfake-detection']` |
| [TostAI/nsfw-image-detection-large](https://huggingface.co/TostAI/nsfw-image-detection-large) | `0.0871B` | `['security-and-privacy']` | `['inappropriate-content']` |
| [Ateeqq/nsfw-image-detection](https://huggingface.co/Ateeqq/nsfw-image-detection) | `0.092B` | `['security-and-privacy']` | `['inappropriate-content']` |
| [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) | `0.1B` | `['security-and-privacy']` | `['inappropriate-content']` |
| [OpenSafetyLab/ImageGuard](https://huggingface.co/OpenSafetyLab/ImageGuard) | `na` | `['security-and-privacy']` | `['inappropriate-content']` |

### Image-Text-To-Text Models
| Name | Size | Category | Sub Category |
|------|------|----------|--------------|
| [meta-llama/Llama-Guard-4-12B](https://huggingface.co/meta-llama/Llama-Guard-4-12B) | `12B` | `['security-and-privacy']` | `['inappropriate-content', 'offensive-language', 'prompt-injection', 'sensitive-content']` |

## Organisations/Companies

### Open Source

| Name | Category | Description |
|------|----------|-------------|
| [guardrails](https://github.com/guardrails-ai/guardrails) | `all` | Adding guardrails to large language models. |
| [NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | `all` | NeMo Guardrails is an open-source toolkit for easily adding programmable guardrails to LLM-based conversational systems. |
| [uqlm](https://github.com/cvs-health/uqlm) | `hallucination` | UQLM: Uncertainty Quantification for Language Models, is a Python package for UQ-based LLM hallucination detection. |
| [llm-guard](https://github.com/protectai/llm-guard) | `all` | The Security Toolkit for LLM Interactions. |

### Closed Source

| Name | Category | Description |
|------|----------|-------------|
| [Lakera](https://www.lakera.ai/lakera-guard) | `all` | Lakera is a company that provides a range of AI services. |
| [Guardrails AI Pro](https://www.guardrailsai.com/pro) | `all` | Guardrails AI Pro is a commercial version of guardrails that provides additional features and support. |

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