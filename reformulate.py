import argparse
import pandas as pd
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are a precise paraphrasing assistant. Rewrite the user's sentence in "
    "different words while preserving the exact meaning, tone, named entities, "
    "numbers, and intent. Do not add, omit, or infer information. Output ONLY "
    "the reformulated sentence, with no preamble, no quotes, no commentary."
)


def build_prompts(tokenizer, sentences):
    prompts = []
    for s in sentences:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
        help="Path to output CSV with columns [original, synthetic]",
    )
    p.add_argument(
        "--column",
        default="post",
        help="Column in input to reformulate",
    )
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Swap to Qwen2.5-32B-Instruct or 7B if VRAM is tight",
    )
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
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
    sample = df.sample(
        n=min(args.n_samples, len(df)), random_state=args.seed
    ).reset_index(drop=True)
    originals = sample[args.column].astype(str).tolist()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    prompts = build_prompts(tokenizer, originals)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    synthetic = [o.outputs[0].text.strip() for o in outputs]

    out_df = pd.DataFrame({"original": originals, "synthetic": synthetic})
    if args.output.lower().endswith((".xlsx", ".xls")):
        out_df.to_excel(args.output, index=False)
    else:
        out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
