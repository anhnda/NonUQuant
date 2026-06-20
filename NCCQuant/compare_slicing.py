"""
Cross-Dataset Validation: AWQ Methods with Sliding Window Evaluation (FINAL CORRECTED)

Fixes:
1. Sliding Window Math: Now uses correct context masking (labels=-100).
2. Llama 3 BOS: Manually handles BOS to prevent "Double BOS" (PPL 15.5 -> 6.2).
3. Tokenizer: Removed fix_mistral_regex kwarg entirely — causes "multiple values"
   error on Llama-3 and some tokenizers versions. Warning is cosmetic and harmless.

Metric: loglikelihood_rolling (Standard lm-evaluation-harness methodology)
Stride: 512 tokens
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
import pickle
import warnings
from pathlib import Path


class AWQSlidingWindowValidator:
    def __init__(self, device="cuda", seed=42, stride=512, max_length=2048, cache_dir="./dataset_cache"):
        self.device = device
        self.seed = seed
        self.stride = stride
        self.max_length = max_length
        self.results = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        print("=" * 80)
        print("AWQ SLIDING WINDOW CROSS-DATASET VALIDATION (FINAL)")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Stride: {stride}")
        print(f"Max Seq Length: {max_length}")
        print(f"Cache Dir: {cache_dir}")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Dataset loaders
    # ------------------------------------------------------------------

    def load_wikitext2_test(self, n_samples=None):
        """
        Load WikiText-2 test set.
        CRITICAL: Concatenates all lines into one continuous stream.
        WikiText is a stream dataset; evaluating separate lines destroys context.
        Note: n_samples parameter is ignored - full test set is always used.
        """
        print("\n[1/3] Loading WikiText-2 test...")

        cache_file = self.cache_dir / f"wikitext2_test_seed{self.seed}.pkl"
        if cache_file.exists():
            print(f"  📦 Loading from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        full_text = "\n".join([x for x in dataset["text"] if x])
        print(f"  ✅ Loaded continuous stream ({len(full_text)} chars)")

        result = [full_text]
        print(f"  💾 Saving to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        return result

    def load_c4_validation(self, n_samples=500):
        """Load C4 validation set as continuous stream."""
        print("\n[2/3] Loading C4 validation...")

        cache_file = self.cache_dir / f"c4_validation_n{n_samples}_seed{self.seed}.pkl"
        if cache_file.exists():
            print(f"  📦 Loading from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for item in tqdm(dataset, total=n_samples, desc="  Collecting C4"):
            if len(texts) >= n_samples:
                break
            if len(item["text"].strip()) > 500:
                texts.append(item["text"])

        full_text = "\n\n".join(texts)
        print(f"  ✅ Loaded continuous stream ({len(full_text)} chars, {len(texts)} documents)")

        result = [full_text]
        print(f"  💾 Saving to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        return result

    def load_ag_news_test(self, n_samples=500):
        """Load AG News test set as continuous stream."""
        print("\n[3/3] Loading AG News test...")

        cache_file = self.cache_dir / f"ag_news_test_n{n_samples}_seed{self.seed}.pkl"
        if cache_file.exists():
            print(f"  📦 Loading from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        dataset = load_dataset("ag_news", split="test")
        texts = [item["text"] for item in dataset if len(item["text"].strip()) > 200]

        if n_samples < len(texts):
            random.seed(self.seed)
            texts = random.sample(texts, n_samples)

        full_text = "\n\n".join(texts)
        print(f"  ✅ Loaded continuous stream ({len(full_text)} chars, {len(texts)} articles)")

        result = [full_text]
        print(f"  💾 Saving to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        return result

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_sliding_window(self, model, tokenizer, texts):
        """Final Corrected Sliding Window Evaluation."""
        model.eval()
        nlls = []
        total_tokens = 0

        for text in texts:
            # Tokenize WITHOUT adding special tokens automatically
            # This prevents the [BOS][BOS] double-injection issue
            encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = encodings.input_ids

            # Manual BOS injection — Llama 3 requires ID 128000 at position 0
            if tokenizer.bos_token_id is not None:
                if input_ids.shape[1] == 0 or input_ids[0, 0].item() != tokenizer.bos_token_id:
                    bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
                    input_ids = torch.cat([bos_tensor, input_ids], dim=1)

            # Safety cap: max_length * 200 tokens (~280k for WikiText-2 full test set)
            if input_ids.size(1) > self.max_length * 200:
                input_ids = input_ids[:, : self.max_length * 200]

            input_ids = input_ids.to(self.device)
            seq_len = input_ids.size(1)

            if seq_len < 2:
                continue

            window_range = list(range(0, seq_len, self.stride))
            num_windows = len(window_range)
            print(f"  Processing {seq_len:,} tokens in {num_windows} windows...")

            prev_end_loc = 0
            pbar = tqdm(window_range, desc="  Windows", unit="win", leave=False)

            for begin_loc in pbar:
                end_loc = min(begin_loc + self.max_length, seq_len)
                trg_len = end_loc - prev_end_loc

                input_chunk = input_ids[:, begin_loc:end_loc]
                target_chunk = input_chunk.clone()

                # Mask context tokens so loss is only computed over new stride tokens
                if begin_loc > 0:
                    target_chunk[:, :-trg_len] = -100

                if target_chunk.size(1) == 0:
                    break

                with torch.no_grad():
                    outputs = model(input_chunk, labels=target_chunk)
                    # outputs.loss is mean NLL; convert back to sum for aggregation
                    neg_log_likelihood = outputs.loss * trg_len

                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc

                # Live PPL in progress bar
                if nlls:
                    current_nll = torch.stack(nlls).sum()
                    current_ppl = torch.exp(current_nll / (total_tokens + prev_end_loc)).item()
                    pbar.set_postfix({"PPL": f"{current_ppl:.4f}", "tokens": f"{total_tokens + prev_end_loc:,}"})

                if end_loc == seq_len:
                    break

            total_tokens += seq_len

        if not nlls:
            return None

        total_nll = torch.stack(nlls).sum()
        perplexity = torch.exp(total_nll / total_tokens).item()

        return {"perplexity": perplexity, "total_tokens": total_tokens}

    # ------------------------------------------------------------------
    # Tokenizer loader (fix for fix_mistral_regex conflict)
    # ------------------------------------------------------------------

    @staticmethod
    def load_tokenizer(model_path):
        """
        Load tokenizer without passing fix_mistral_regex.

        fix_mistral_regex was a tokenizers-internal kwarg used to suppress a
        deprecation warning. Passing it to AutoTokenizer on Llama-3 (or newer
        tokenizers builds) raises:
            TypeError: got multiple values for keyword argument 'fix_mistral_regex'
        The warning it suppresses is cosmetic — skipping the kwarg is safe.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
            )
        return tokenizer

    # ------------------------------------------------------------------
    # Per-model evaluation
    # ------------------------------------------------------------------

    def evaluate_model_on_dataset(self, model_path, model_name, texts, dataset_name):
        print(f"\n  Evaluating {model_name} on {dataset_name}...")

        try:
            tokenizer = self.load_tokenizer(model_path)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )

            results = self.evaluate_sliding_window(model, tokenizer, texts)

            if results:
                print(f"  ✅ Perplexity: {results['perplexity']:.4f}")
            else:
                print("  ❌ Evaluation failed (no results)")

            del model
            torch.cuda.empty_cache()
            return results

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_validation(self, heuristic_path, standard_path=None, n_samples=2000):
        print("\n" + "=" * 80)
        print("LOADING DATASETS")
        print("=" * 80)

        datasets = {
            "WikiText-2": self.load_wikitext2_test(n_samples),
            "C4": self.load_c4_validation(n_samples),
            # "AG News": self.load_ag_news_test(n_samples),
        }

        print("\n" + "=" * 80)
        print("EVALUATING MODELS")
        print("=" * 80)

        models = {"Heuristic AWQ": heuristic_path}
        if standard_path:
            models["Standard AWQ"] = standard_path

        for dataset_name, texts in datasets.items():
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*80}")

            for model_name, model_path in models.items():
                result = self.evaluate_model_on_dataset(model_path, model_name, texts, dataset_name)
                if result:
                    if dataset_name not in self.results:
                        self.results[dataset_name] = {}
                    self.results[dataset_name][model_name] = result

        return self.results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_comparison_table(self):
        """Generate formatted comparison table."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE RESULTS")
        print("=" * 80)

        has_both_models = any(
            len(self.results.get(ds, {})) == 2 for ds in self.results.keys()
        )

        if not has_both_models:
            print(f"\n{'Dataset':<15} {'Model':<20} {'Perplexity':<15} {'Total Tokens':<15}")
            print("-" * 70)

            dataset_results = []
            for dataset_name, models_data in self.results.items():
                for model_name, data in models_data.items():
                    ppl = data["perplexity"]
                    tokens = data["total_tokens"]
                    print(f"{dataset_name:<15} {model_name:<20} {ppl:<15.4f} {tokens:<15,}")
                    dataset_results.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "perplexity": ppl,
                        "total_tokens": tokens,
                    })
            return dataset_results

        print(f"\n{'Dataset':<15} {'Heuristic AWQ':<15} {'Standard AWQ':<15} {'Delta':<12} {'Winner':<10}")
        print("-" * 80)

        dataset_results = []
        for dataset_name in self.results.keys():
            if (
                "Heuristic AWQ" in self.results[dataset_name]
                and "Standard AWQ" in self.results[dataset_name]
            ):
                heur_ppl = self.results[dataset_name]["Heuristic AWQ"]["perplexity"]
                std_ppl = self.results[dataset_name]["Standard AWQ"]["perplexity"]
                delta = heur_ppl - std_ppl
                delta_pct = (delta / std_ppl) * 100
                winner = "Heuristic" if delta < -0.05 else ("Standard" if delta > 0.05 else "Tie")

                print(f"{dataset_name:<15} {heur_ppl:<15.4f} {std_ppl:<15.4f} {delta_pct:>+11.3f}%  {winner:<10}")

                dataset_results.append({
                    "dataset": dataset_name,
                    "heuristic_ppl": heur_ppl,
                    "standard_ppl": std_ppl,
                    "delta_pct": delta_pct,
                    "winner": winner,
                })

        return dataset_results

    def analyze_results(self, dataset_results):
        """Comprehensive analysis of results."""
        if not dataset_results or "heuristic_ppl" not in dataset_results[0]:
            print("\n" + "=" * 80)
            print("SINGLE MODEL EVALUATION COMPLETE")
            print("=" * 80)
            return {"mode": "single_model"}

        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        heur_wins = sum(1 for r in dataset_results if r["winner"] == "Heuristic")
        std_wins = sum(1 for r in dataset_results if r["winner"] == "Standard")
        ties = sum(1 for r in dataset_results if r["winner"] == "Tie")

        print(f"\nWin Count:")
        print(f"  Heuristic AWQ: {heur_wins}/{len(dataset_results)}")
        print(f"  Standard AWQ:  {std_wins}/{len(dataset_results)}")
        print(f"  Ties:          {ties}/{len(dataset_results)}")

        avg_heur = np.mean([r["heuristic_ppl"] for r in dataset_results])
        avg_std = np.mean([r["standard_ppl"] for r in dataset_results])
        avg_delta_pct = ((avg_heur - avg_std) / avg_std) * 100

        print(f"\nAverage Perplexity:")
        print(f"  Heuristic AWQ: {avg_heur:.4f}")
        print(f"  Standard AWQ:  {avg_std:.4f}")
        print(f"  Difference:    {avg_delta_pct:+.3f}%")

        print("\n" + "=" * 80)
        print("FINAL VERDICT")
        print("=" * 80)

        if heur_wins > std_wins:
            print(f"\n🏆 HEURISTIC AWQ is the OVERALL WINNER!")
            print(f"   Wins: {heur_wins}/{len(dataset_results)} datasets")
            print(f"   Average improvement: {abs(avg_delta_pct):.3f}%")
            winner = "Heuristic AWQ"
        elif std_wins > heur_wins:
            print(f"\n🏆 STANDARD AWQ is the OVERALL WINNER!")
            print(f"   Wins: {std_wins}/{len(dataset_results)} datasets")
            print(f"   Average improvement: {abs(avg_delta_pct):.3f}%")
            winner = "Standard AWQ"
        else:
            print(f"\n🤝 TIE - Both methods equally strong")
            winner = "Tie"

        return {
            "winner": winner,
            "heuristic_wins": heur_wins,
            "standard_wins": std_wins,
            "ties": ties,
            "avg_heuristic": avg_heur,
            "avg_standard": avg_std,
            "avg_delta_pct": avg_delta_pct,
        }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AWQ Sliding Window Cross-Dataset Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--heuristic-path", type=str, required=True,
                        help="Path to Heuristic AWQ model")
    parser.add_argument("--standard-path", type=str, default="",
                        help="Path to Standard AWQ model (optional for comparison)")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of samples per dataset")
    parser.add_argument("--cache-dir", type=str, default="./dataset_cache",
                        help="Directory to cache downloaded datasets")
    args = parser.parse_args()

    validator = AWQSlidingWindowValidator(cache_dir=args.cache_dir)

    validator.run_validation(
        args.heuristic_path,
        args.standard_path if args.standard_path else None,
        args.n_samples,
    )

    dataset_results = validator.generate_comparison_table()
    analysis = validator.analyze_results(dataset_results)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    if analysis.get("mode") != "single_model":
        print(f"\n🏆 Winner: {analysis['winner']}")
        print(f"📊 Tested: {len(dataset_results)} datasets")
        print(f"✅ Heuristic wins: {analysis['heuristic_wins']}")
        print(f"✅ Standard wins: {analysis['standard_wins']}")
        print(f"🤝 Ties: {analysis['ties']}")
    else:
        print(f"\n📊 Tested: {len(dataset_results)} datasets")
        print(f"✅ Single model evaluation complete")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()