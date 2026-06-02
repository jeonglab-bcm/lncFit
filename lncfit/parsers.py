import re


def parse_log2fc(text: str):
    match = re.search(r"(?<![A-Za-z\d])[-+]?\d+\.?\d*", text)
    return float(match.group()) if match else None
