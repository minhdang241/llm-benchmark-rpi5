"""Token-shape prompts for S1.

The prompt is intentionally content-neutral and repeatable. The actual prompt
token count is recorded from llama.cpp's timing output, because tokenizer
families differ slightly.
"""


def s1_pp512() -> str:
    sentence = (
        "A small edge computer receives sensor readings, checks local context, "
        "summarizes recent events, and chooses a concise action for an operator."
    )
    return (
        "Read the following repeated operations log and produce a short technical "
        "answer about the likely system bottleneck.\n\n"
        + " ".join(sentence for _ in range(30))
        + "\n\nAnswer in a direct paragraph."
    )


PROMPTS = {
    "s1_pp512": s1_pp512,
}


def get_prompt(name: str) -> str:
    try:
        return PROMPTS[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(PROMPTS))
        raise ValueError(f"Unknown prompt {name!r}; available prompts: {choices}") from exc
