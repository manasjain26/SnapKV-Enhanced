"""
Aggregate LongBench results across all SnapKV configs into a comparison table.

Usage:
    python compare_results.py --results_dir pred_e --model_prefix mistral-7B-instruct-v0.2

This will:
1. Find all result.json files for configs matching the model prefix
2. Print a per-task comparison table
3. Print category averages
4. Save results to a CSV file
"""

import os
import json
import argparse
from collections import defaultdict

# Task categories
CATEGORIES = {
    "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news"],
    "Few-Shot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}

# Friendly names for configs
CONFIG_NAMES = {
    "baseline_avgpool": "Baseline(avg)",
    "baseline_maxpool": "Baseline(max)",
    "enhanced_weighted_only": "+Weighted",
    "enhanced_multiwindow_only": "+MultiWin",
    "enhanced_spikes_only": "+Spikes",
    "enhanced_combined": "Combined",
}


def find_result_dirs(results_dir, model_prefix):
    """Find all directories matching model_prefix + config pattern."""
    configs = {}
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} does not exist")
        return configs

    for dirname in sorted(os.listdir(results_dir)):
        if dirname.startswith(model_prefix):
            config_suffix = dirname[len(model_prefix):]
            # Try to load result.json
            result_path = os.path.join(results_dir, dirname, "result.json")
            jsonl_dir = os.path.join(results_dir, dirname)

            if os.path.exists(result_path):
                configs[config_suffix] = {"path": result_path, "type": "result_json"}
            elif os.path.isdir(jsonl_dir):
                # Check if there are jsonl files (eval.py hasn't been run yet)
                jsonl_files = [f for f in os.listdir(jsonl_dir) if f.endswith('.jsonl')]
                if jsonl_files:
                    configs[config_suffix] = {"path": jsonl_dir, "type": "jsonl_dir"}

    return configs


def load_results(config_info):
    """Load results from result.json."""
    if config_info["type"] == "result_json":
        with open(config_info["path"], "r") as f:
            return json.load(f)
    elif config_info["type"] == "jsonl_dir":
        print(f"  ⚠ Found JSONL files but no result.json at {config_info['path']}")
        print(f"    Run: python eval.py --model <dirname>")
        return None
    return None


def print_comparison_table(all_results, configs_order):
    """Print a formatted comparison table."""
    # Collect all tasks
    all_tasks = set()
    for results in all_results.values():
        if results:
            all_tasks.update(results.keys())
    all_tasks = sorted(all_tasks)

    if not all_tasks:
        print("No results found!")
        return

    # Header
    config_labels = []
    for c in configs_order:
        label = CONFIG_NAMES.get(c, c[:12])
        config_labels.append(label)

    header = f"{'Task':<25}" + "".join(f"{label:>14}" for label in config_labels)
    print("\n" + "=" * len(header))
    print("Per-Task Scores")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Per-task rows
    for task in all_tasks:
        row = f"{task:<25}"
        scores = []
        for config in configs_order:
            results = all_results.get(config)
            if results and task in results:
                score = results[task]
                if isinstance(score, dict):
                    # Length-bucketed scores
                    score = sum(score.values()) / len(score)
                row += f"{score:>14.1f}"
                scores.append(score)
            else:
                row += f"{'—':>14}"
                scores.append(None)

        # Highlight best score
        valid_scores = [s for s in scores if s is not None]
        print(row)

    # Category averages
    print("\n" + "=" * len(header))
    print("Category Averages")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    overall_avgs = defaultdict(list)

    for cat_name, cat_tasks in CATEGORIES.items():
        row = f"{cat_name:<25}"
        for config in configs_order:
            results = all_results.get(config)
            if results:
                cat_scores = []
                for task in cat_tasks:
                    if task in results:
                        score = results[task]
                        if isinstance(score, dict):
                            score = sum(score.values()) / len(score)
                        cat_scores.append(score)
                if cat_scores:
                    avg = sum(cat_scores) / len(cat_scores)
                    row += f"{avg:>14.1f}"
                    overall_avgs[config].append(avg)
                else:
                    row += f"{'—':>14}"
            else:
                row += f"{'—':>14}"
        print(row)

    # Overall average
    print("-" * len(header))
    row = f"{'OVERALL AVERAGE':<25}"
    for config in configs_order:
        if overall_avgs[config]:
            avg = sum(overall_avgs[config]) / len(overall_avgs[config])
            row += f"{avg:>14.1f}"
        else:
            row += f"{'—':>14}"
    print(row)
    print("=" * len(header))

    # Delta table (vs baseline_avgpool)
    baseline_config = configs_order[0] if configs_order else None
    baseline_results = all_results.get(baseline_config, {})

    if baseline_results and len(configs_order) > 1:
        print("\n" + "=" * len(header))
        print(f"Delta vs {CONFIG_NAMES.get(baseline_config, baseline_config)}")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for task in all_tasks:
            row = f"{task:<25}"
            baseline_score = baseline_results.get(task)
            if isinstance(baseline_score, dict):
                baseline_score = sum(baseline_score.values()) / len(baseline_score)

            for config in configs_order:
                results = all_results.get(config)
                if results and task in results and baseline_score is not None:
                    score = results[task]
                    if isinstance(score, dict):
                        score = sum(score.values()) / len(score)
                    delta = score - baseline_score
                    sign = "+" if delta >= 0 else ""
                    row += f"{sign}{delta:>13.1f}"
                else:
                    row += f"{'—':>14}"
            print(row)
        print("=" * len(header))


def save_csv(all_results, configs_order, output_path):
    """Save results to CSV."""
    all_tasks = set()
    for results in all_results.values():
        if results:
            all_tasks.update(results.keys())
    all_tasks = sorted(all_tasks)

    config_labels = [CONFIG_NAMES.get(c, c) for c in configs_order]

    with open(output_path, "w") as f:
        f.write("Task," + ",".join(config_labels) + "\n")
        for task in all_tasks:
            scores = []
            for config in configs_order:
                results = all_results.get(config)
                if results and task in results:
                    score = results[task]
                    if isinstance(score, dict):
                        score = sum(score.values()) / len(score)
                    scores.append(f"{score:.1f}")
                else:
                    scores.append("")
            f.write(f"{task}," + ",".join(scores) + "\n")

    print(f"\n📁 CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare SnapKV LongBench results")
    parser.add_argument("--results_dir", type=str, default="pred_e",
                        help="Directory containing result subdirectories")
    parser.add_argument("--model_prefix", type=str, default="mistral-7B-instruct-v0.2",
                        help="Model name prefix used in directory names")
    parser.add_argument("--output_csv", type=str, default="comparison_results.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    print(f"Looking for results in: {args.results_dir}")
    print(f"Model prefix: {args.model_prefix}")

    # Also check H2O/results path (eval.py outputs there)
    results_dirs_to_check = [args.results_dir, "H2O/results"]

    configs = {}
    for rdir in results_dirs_to_check:
        found = find_result_dirs(rdir, args.model_prefix)
        if found:
            print(f"\nFound configs in {rdir}/:")
            configs.update(found)

    if not configs:
        # Try without prefix
        print(f"\nNo configs found with prefix '{args.model_prefix}'")
        print(f"Available directories in {args.results_dir}/:")
        if os.path.exists(args.results_dir):
            for d in sorted(os.listdir(args.results_dir)):
                print(f"  {d}")
        print(f"\nTry: python compare_results.py --model_prefix <prefix>")
        return

    for config_name, config_info in sorted(configs.items()):
        print(f"  {config_name}: {config_info['path']}")

    # Preferred order
    preferred_order = [
        "baseline_avgpool", "baseline_maxpool",
        "enhanced_weighted_only", "enhanced_multiwindow_only",
        "enhanced_spikes_only", "enhanced_combined"
    ]
    configs_order = [c for c in preferred_order if c in configs]
    # Add any remaining configs not in preferred order
    configs_order += [c for c in sorted(configs.keys()) if c not in configs_order]

    # Load results
    all_results = {}
    needs_eval = []
    for config_name in configs_order:
        results = load_results(configs[config_name])
        if results is None:
            needs_eval.append(config_name)
        else:
            all_results[config_name] = results

    if needs_eval:
        print(f"\n⚠ The following configs need eval.py to be run first:")
        for c in needs_eval:
            print(f"  python eval.py --model {args.model_prefix}{c}")
        if not all_results:
            return

    # Print comparison
    active_configs = [c for c in configs_order if c in all_results]
    print_comparison_table(all_results, active_configs)
    save_csv(all_results, active_configs, args.output_csv)


if __name__ == "__main__":
    main()
