# src/labeling/prompt_builder.py

import re
from src.labeling.llm_client import call_qwen_chat



def clean_llm_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def label_prompting(
    *,
    emails: list[str] | None = None,
    labels: list[str] | None = None,
    summaries: list[str] | None = None,
    role: str = "email clustering",
) -> tuple[str, str]:
    """
    Generate a strict (label, summary) pair using Gemini.

    Exactly ONE of the following must be provided:
    - emails (leaf level)
    - labels + summaries (merge / parent / root level)

    Returns:
        (label, summary)
    """

    # ----------------------------
    # Prompt construction
    # ----------------------------
    prompt_parts = [
        f"You are an expert NLP system for {role}.\n",
        "Task:\n"
    ]

    if emails is not None:
        prompt_parts.append(
            "Analyze the following emails and identify their common theme.\n\n"
        )
    else:
        prompt_parts.append(
            "Consolidate the following labels and summaries into a single higher-level theme.\n\n"
        )

    prompt_parts.append(
        "Rules (must follow strictly):\n"
        "- Output EXACTLY one label and one summary.\n"
        "- Label MUST be a concise noun phrase.\n"
        "- The label MUST represent the shared abstract topic across all inputs.\n"
        "- Label MUST be 3–8 words ONLY (never exceed 8 words).\n"
        "- Do NOT use punctuation in the label.\n"
        "- Do NOT use generic words such as email, message, mail, discussion, conversation.\n"
        "- Summary MUST be 1–2 short sentences.\n"
        "- Summary must describe the shared intent, topic or theme, not topic, not individual emails or labels\n"
        "- Do NOT repeat the label in the summary.\n"
        "- Do NOT include bullet points, lists, or explanations.\n"
        "- Must provide the label and summary in the exact output format.\n\n"

        "Output format (STRICT — do not add anything else):\n"
        "Label: <label text>\n"
        "Summary: <summary text>\n\n"
    )

    if emails is not None:
        prompt_parts.append(
            "Emails:\n" +
            "\n".join(f"{i+1}. {text}" for i, text in enumerate(emails))
        )
    else:
        prompt_parts.append(
            "Input labels:\n" +
            "\n".join(f"- {lbl}" for lbl in labels) +
            "\n\nInput summaries:\n" +
            "\n".join(f"- {smr}" for smr in summaries)
        )

    prompt = "".join(prompt_parts)

    # ----------------------------
    # LLM call
    # ----------------------------
    result = call_gemini_chat(prompt)

    # ----------------------------
    # Strict parsing
    # ----------------------------
    label = None
    summary = None

    for line in result.splitlines():
        line_lower = line.lower().strip()

        if line_lower.startswith("label:"):
            label = clean_llm_text(line.split(":", 1)[1])
        elif line_lower.startswith("summary:"):
            summary = clean_llm_text(line.split(":", 1)[1])


    return label, summary
