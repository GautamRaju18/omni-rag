"""The answer prompt and its reply schema (Phase 10).

Bump ``PROMPT_VERSION`` whenever the system prompt, the user template or the schema changes:
evaluation reports record it, so results from different prompts are never mixed.

The brief's seven requirements map onto the rules below:
1. sources as the only evidence (rule 1) · 2. no unsupported claims (rules 1, 2) ·
3. document names and pages: the numbered sources carry them, code turns [n] into a citation ·
4. insufficient evidence (rule 5) · 5. direct vs inferred (``support``) ·
6. multi-document answers (rule 3: every part, any number of sources) ·
7. tables and image descriptions (rules 2 and 4).
"""

from __future__ import annotations

PROMPT_VERSION = "answer-v1"

ANSWER_SYSTEM = (
    "You answer questions about technical project documents using ONLY the numbered "
    "sources you are given.\n"
    "\n"
    "Rules:\n"
    "1. Use only facts stated in the sources. No outside knowledge, no guesses.\n"
    "2. Answer with a list of short claims, one fact each. For every claim give:\n"
    "   - text: the claim, complete enough to read on its own;\n"
    "   - sources: the numbers of the sources that support it;\n"
    "   - quote: words copied exactly from one of those sources that state the fact "
    "(one sentence or one table row, at most 40 words);\n"
    '   - support: "direct" if the quote states the claim, "inferred" if the claim '
    "combines several sources or follows from them without being stated.\n"
    "3. Answer every part of the question. Facts may come from different sources.\n"
    '4. Text marked "Image description (generated ...)" was written by a model, not '
    'by the document\'s author: claims that rely on it are "inferred".\n'
    "5. If the sources do not contain the answer, set answerable to false, give no "
    "claims, and say in missing what information is absent. If they contain only part"
    " of the answer, give that part and say in missing what is absent; otherwise "
    'missing is "".\n'
    "6. Copy numbers, units, names and identifiers exactly as the sources write them."
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "integer"}},
                    "quote": {"type": "string"},
                    "support": {"type": "string", "enum": ["direct", "inferred"]},
                },
                "required": ["text", "sources", "quote", "support"],
            },
        },
        "missing": {"type": "string"},
    },
    "required": ["answerable", "claims", "missing"],
}


def user_prompt(question: str, rendered_context: str) -> str:
    """Sources first, question last (the model reads the evidence before the task)."""
    return f"Sources:\n\n{rendered_context}\n\nQuestion: {question}"
