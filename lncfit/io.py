def read_fasta(path: str) -> list[str]:
    sequences = []
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
    if current_seq:
        sequences.append("".join(current_seq))
    return sequences
