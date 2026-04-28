import argparse
import pandas as pd
from vllm import LLM, SamplingParams


SYSTEM_PROMPT_EQUIVALENT = (
    "You are a precise paraphrasing assistant. Rewrite the user's sentence in "
    "different words while preserving the exact meaning, tone, named entities, "
    "numbers, and intent. Do not add, omit, or infer information. Output ONLY "
    "the reformulated sentence, with no preamble, no quotes, no commentary."
)

SYSTEM_PROMPT_UNEQUIVALENT = (
    "You are a sentence-rewriting assistant. Your job is to produce a sentence "
    "whose MEANING is clearly DIFFERENT from the original. The output must NOT "
    "be a paraphrase, a summary, or a shortened version of the input. Keep the "
    "topic, style, and approximate length similar, but apply at least ONE of "
    "the following transformations so the assertion changes:\n"
    "  - flip a verdict, claim, or negation (e.g. 'qualifies' -> 'does not "
    "qualify', 'is hate speech' -> 'is not hate speech', 'yes' -> 'no')\n"
    "  - swap a key entity, target group, person, or category for a different "
    "one\n"
    "  - invert the sentiment or stance (positive <-> negative, supports <-> "
    "opposes)\n"
    "  - change a key fact, number, or attribute\n"
    "Example:\n"
    "  Original: The post does not direct abuse at any identity and is a "
    "neutral statement.\n"
    "  Rewritten: The post directs abuse at a specific identity and is a "
    "hateful statement.\n"
    "A reader comparing the two sentences must immediately see they assert "
    "different things. Output ONLY the rewritten sentence, with no preamble, "
    "no quotes, no commentary."
)


def build_prompts(tokenizer, sentences, system_prompts):
    prompts = []
    for s, sys in zip(sentences, system_prompts):
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": str(s)},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return prompts


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="hatecot_final_D3 (2).csv",
        help="Path to input CSV/XLSX",
    )
    p.add_argument(
        "--output",
        default="reformulated.csv",
        help="Path to output CSV. Original columns are preserved with an added 'synthetic' column.",
    )
    p.add_argument(
        "--column",
        default="post",
        help="Column in input to reformulate",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of rows to sample. Use -1 (or 0) to reformulate the full dataset.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct-AWQ",
        help="AWQ 4-bit fits ~10GB. Use Qwen2.5-72B-Instruct on a big GPU.",
    )
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "auto"],
        help="Use float16 on V100 (no bf16 hardware).",
    )
    p.add_argument(
        "--quantization",
        default="awq",
        help="Matches the model. Pass 'none' for unquantized models.",
    )
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    if args.input.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)

    if args.column not in df.columns:
        raise ValueError(
            f"Column '{args.column}' not in {list(df.columns)}"
        )

    df = df[df[args.column].notna()]
    df = df[df[args.column].astype(str).str.strip() != ""]
    if args.n_samples and args.n_samples > 0:
        sample = df.sample(
            n=min(args.n_samples, len(df)), random_state=args.seed
        ).reset_index(drop=True)
    else:
        sample = df.reset_index(drop=True)
    originals = sample[args.column].astype(str).tolist()

    quantization = None if args.quantization.lower() == "none" else args.quantization
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        quantization=quantization,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    labels = [
        "equivalent" if i % 2 == 0 else "unequivalent"
        for i in range(len(originals))
    ]
    system_prompts = [
        SYSTEM_PROMPT_EQUIVALENT if l == "equivalent" else SYSTEM_PROMPT_UNEQUIVALENT
        for l in labels
    ]
    prompts = build_prompts(tokenizer, originals, system_prompts)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    synthetic = [o.outputs[0].text.strip() for o in outputs]

    out_df = sample.copy()
    out_df["synthetic"] = synthetic
    out_df["meaning"] = labels
    if args.output.lower().endswith((".xlsx", ".xls")):
        out_df.to_excel(args.output, index=False)
    else:
        out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
