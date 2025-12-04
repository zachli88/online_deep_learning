# file written with the help of chatgpt 5.1
def generate_dataset(
    output_json: str,
    oversample: int = 10,
    temperature: float = 0.6,
):
    """
    Create a reasoning-augmented dataset for RFT by sampling many
    CoT-style generations and retaining only those that match
    the correct numeric answer.
    """
    import json
    from pathlib import Path
    from tqdm import tqdm

    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    # ------------------------------------------------------------
    # Load the reasoning-capable model (large instruct checkpoint)
    # ------------------------------------------------------------
    print("[RFT-DATAGEN] Initializing CoT model...")
    generator = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    # Primary training split
    base_data = Dataset("train")

    # Container for RFT entries
    refined_rows = []
    accepted = 0

    print(f"[RFT-DATAGEN] Sampling {oversample} completions per example (T={temperature})")

    # ------------------------------------------------------------
    # Iterate through dataset
    # ------------------------------------------------------------
    for idx in tqdm(range(len(base_data)), desc="Collecting valid CoT rollouts"):
        prompt, target_val = base_data[idx]

        # Sample multiple reasoning traces
        # batched_generate returns list[list[str]] when num_return_sequences > 1
        sampled = generator.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature,
        )[0]

        # Identify the first valid answer
        chosen_trace = None
        for seq in sampled:
            try:
                predicted = generator.parse_answer(seq)
                if is_answer_valid(predicted, target_val):
                    chosen_trace = seq
                    break
            except Exception:
                continue

        # Append a successful sample
        if chosen_trace is not None:
            refined_rows.append([prompt, target_val, chosen_trace])
            accepted += 1

    # ------------------------------------------------------------
    # Write resulting dataset to disk
    # ------------------------------------------------------------
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fout:
        json.dump(refined_rows, fout, indent=2)

    total = len(base_data)
    rate = (accepted / total) * 100 if total > 0 else 0

    print(f"\n[RFT-DATAGEN] Completed.")
    print(f"  Accepted: {accepted} / {total} ({rate:.1f}%)")
    print(f"  Output saved to: {output_json}")


if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)
