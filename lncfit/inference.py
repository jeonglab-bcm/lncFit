from transformers import pipeline

from lncfit.parsers import parse_log2fc


def run_chatnt_inference(prompt: str, dna_sequences: list[str]) -> float | None:
    """Load ChatNT, run inference, print raw response, return parsed log2FC."""
    pipe = pipeline(model="InstaDeepAI/ChatNT", trust_remote_code=True)
    result = pipe(inputs={"english_sequence": prompt, "dna_sequences": dna_sequences})

    raw_response = result[0]["generated_text"] if isinstance(result, list) else str(result)

    print("=== Raw ChatNT response ===")
    print(raw_response)
    print()

    value = parse_log2fc(raw_response)
    if value is not None:
        print(f"Parsed log2FC  : {value}")
    else:
        print("Parsed log2FC  : (no numeric value found in response)")
    print()
    print(
        "WARNING: lncRNA essentiality / CRISPR-screen log2FC is outside ChatNT's "
        "documented training tasks. This is an out-of-distribution exploratory estimate "
        "and requires experimental validation against known CRISPR-screen labels before use."
    )

    return value
