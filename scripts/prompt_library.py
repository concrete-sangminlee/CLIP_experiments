"""Central prompt library for zero-shot and fine-tuning experiments."""
from __future__ import annotations

from typing import Dict

# Class-specific descriptors in English and Korean for better coverage.
CLASS_DESCRIPTORS: Dict[str, Dict[str, str]] = {
    "loosened": {
        "en": "a loosened or partially unscrewed steel bolt that needs tightening",
        "ko": "제대로 조여지지 않아 틈이 벌어진 느슨한 강철 볼트",
    },
    "missing": {
        "en": "an empty bolt hole or missing nut exposing bare metal",
        "ko": "볼트나 너트가 완전히 사라져 빈 구멍이 드러난 상태",
    },
    "fixed": {
        "en": "a properly fastened structural bolt with intact protective coating",
        "ko": "방청 코팅이 유지된 완전히 고정된 구조용 볼트",
    },
}

# Prompt templates referencing {condition} placeholder.
PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "base_en": {
        "lang": "en",
        "template": "a high-resolution industrial inspection photo of {condition}",
    },
    "materials_en": {
        "lang": "en",
        "template": "macro photograph of {condition} on a steel bridge connection",
    },
    "context_ko": {
        "lang": "ko",
        "template": "산업 현장의 구조물에서 {condition} 를 보여주는 근접 사진",
    },
    "safety_report": {
        "lang": "en",
        "template": "engineering report style image depicting {condition} under harsh lighting",
    },
}

DEFAULT_TEMPLATE_KEYS = ["base_en", "materials_en", "context_ko"]
DEFAULT_LANGUAGES = ["en", "ko"]


def build_prompts(class_name: str, template_keys=None, languages=None):
    template_keys = template_keys or DEFAULT_TEMPLATE_KEYS
    languages = languages or DEFAULT_LANGUAGES
    prompts = []
    for key in template_keys:
        entry = PROMPT_TEMPLATES.get(key)
        if not entry:
            continue
        lang = entry["lang"]
        if languages and lang not in languages:
            continue
        descriptor = CLASS_DESCRIPTORS[class_name].get(lang, CLASS_DESCRIPTORS[class_name]["en"])
        prompts.append(entry["template"].format(condition=descriptor))
    return prompts
