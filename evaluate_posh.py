import json
import numpy as np
import pandas as pd
from operator import itemgetter
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter
from scipy.stats import spearmanr, kendalltau

import argparse

from posh.posh import PoSh

DOCENT_LABELS = [
    "1_much_better",
    "1_slightly_better",
    "equal",
    "2_slightly_better",
    "2_much_better",
]
DOCENT_1_BETTER = 0
DOCENT_1_2_EQUAL = 1
DOCENT_2_BETTER = 2

# if you want to evaluate this locally, please
# download the CapArena annotations from:
# https://github.com/njucckevin/CapArena?tab=readme-ov-file#reproduce-paper-results

CAPARENA_PATH = "corpora/caparena/caparena_annots_eval.json"


def load_coarse_benchmark(benchmark_name):
    generation_reference_pairs, pairwise_rankings = set(), []
    if benchmark_name == "docent":
        docent = load_dataset("amitha/docent-eval", split="coarse")
        for row in docent:
            reference = row["reference"]
            for key in ["model1", "model2"]:
                generation_reference_pairs.add((f"{row['uuid']}-{row[key]}", row[f"{key}_generation"], reference))

            pairwise_rankings.append(
                {
                    "model1_generation": row["model1_generation"],
                    "model2_generation": row["model2_generation"],
                    "reference": row["reference"],
                    "mistakes": DOCENT_LABELS.index(row["mistakes"]),
                    "omissions": DOCENT_LABELS.index(row["omissions"]),
                    "overall_quality": DOCENT_LABELS.index(row["overall_quality"]),
                }
            )
    else:
        for idx, (_, row) in enumerate(pd.read_json(CAPARENA_PATH).iterrows()):
            # we skip cases where human reference annotations were judged against
            # model generations as they are not suitable for reference based evaluation
            if "human" in {row["source1"], row["source2"]}:
                continue

            assert row["winner"] in {"equal", "skip", "bad", row["source1"], row["source2"]}, (
                f"winner {row['winner']} not in {row['source1']}, {row['source2']}"
            )

            reference = row["ref"]
            for key in ["1", "2"]:
                generation_reference_pairs.add((f"{row['img']}-{row['source' + key]}", row[f"caption{key}"], reference))

            # this mapping replicates CapArena's reported results
            winner = {
                row["source1"]: "model1",
                row["source2"]: "model2",
                "equal": "equal",
                "skip": "equal",
                "bad": "equal",
            }[row["winner"]]
            pairwise_rankings.append(
                {
                    "dataset_idx": idx,
                    "image": row["img"],
                    "model1": row["source1"],
                    "model2": row["source2"],
                    "model1_generation": row["caption1"],
                    "model2_generation": row["caption2"],
                    "reference": reference,
                    "winner": winner,
                    "cluster": row["cluster"],
                }
            )

    generation_reference_pairs = list(sorted(generation_reference_pairs))
    cache_keys = list(map(itemgetter(0), generation_reference_pairs))
    generations = list(map(itemgetter(1), generation_reference_pairs))
    references = list(map(itemgetter(2), generation_reference_pairs))

    assert len(set(cache_keys)) == len(cache_keys)

    return generations, references, cache_keys, pairwise_rankings


# adapted from compute_elo in the CapArena codebase
# https://github.com/njucckevin/CapArena/blob/master/cal_ranking.py
def calculate_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rankings = defaultdict(lambda: INIT_RATING)

    for rd, source1, source2, winner in battles[["source1", "source2", "winner"]].itertuples():
        ra = rankings[source1]
        rb = rankings[source2]

        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == source1:
            sa = 1
        elif winner == source2:
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)" or winner == "equal":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rankings[source1] += K * (sa - ea)
        rankings[source2] += K * (1 - sa - eb)

    return rankings


def calculate_bootstrap_elo(battles, n_trials=1_000):
    battles = pd.DataFrame.from_records(battles, columns=["source1", "source2", "winner"])

    rows = []
    np.random.seed(42)
    for i in range(n_trials):
        rows.append(calculate_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)

    bootstrap_elo_lu = df[df.median().sort_values(ascending=False).index]
    bootstrap_lu_median = (
        bootstrap_elo_lu.median().reset_index().set_axis(["model", "Elo rating"], axis=1)
    )
    bootstrap_lu_median["Elo rating"] = (bootstrap_lu_median["Elo rating"] + 0.5).astype(int)

    return {row["model"]: row["Elo rating"] for _, row in bootstrap_lu_median.iterrows()}


def calculate_coarse_correlations(pairwise_rankings, scores, benchmark, n_trials=1000):
    if benchmark == "docent":
        tie_counts = Counter()
        score_differences = defaultdict(list)
        for pairwise_ranking in pairwise_rankings:
            model1_generation = pairwise_ranking["model1_generation"]
            model2_generation = pairwise_ranking["model2_generation"]
            reference = pairwise_ranking["reference"]

            model1_score = scores[(model1_generation, reference)]
            model2_score = scores[(model2_generation, reference)]

            for ranking_type, metric_type in [
                ("mistakes", "precision"),
                ("omissions", "recall"),
                ("overall_quality", "f1"),
            ]:
                if pairwise_ranking[ranking_type] == DOCENT_LABELS.index("equal"):
                    tie_counts[ranking_type] += 1

                score_differences[metric_type].append(
                    abs(model1_score[metric_type] - model2_score[metric_type])
                )

        tie_thresholds_exc = {
            metric_type: list(sorted(score_differences[metric_type]))[tie_counts[ranking_type]]
            for ranking_type, metric_type in (
                [("mistakes", "precision"), ("omissions", "recall"), ("overall_quality", "f1")]
            )
        }

        acc_tie_labels, diff_labels = (
            defaultdict(list),
            defaultdict(list),
        )
        acc_tie_preds, diff_preds = (
            defaultdict(list),
            defaultdict(list),
        )
        for pairwise_ranking in pairwise_rankings:
            model1_generation = pairwise_ranking["model1_generation"]
            model2_generation = pairwise_ranking["model2_generation"]
            reference = pairwise_ranking["reference"]

            model1_score = scores[(model1_generation, reference)]
            model2_score = scores[(model2_generation, reference)]

            for ranking_type, metric_type in [
                ("mistakes", "precision"),
                ("omissions", "recall"),
                ("overall_quality", "f1"),
            ]:
                if (
                    model1_score[metric_type] - model2_score[metric_type]
                    >= tie_thresholds_exc[metric_type]
                ):
                    acc_tie_preds[ranking_type].append(DOCENT_1_BETTER)
                elif (
                    model2_score[metric_type] - model1_score[metric_type]
                    >= tie_thresholds_exc[metric_type]
                ):
                    acc_tie_preds[ranking_type].append(DOCENT_2_BETTER)
                else:
                    acc_tie_preds[ranking_type].append(DOCENT_1_2_EQUAL)
                acc_tie_labels[ranking_type].append(
                    {
                        0: DOCENT_1_BETTER,
                        1: DOCENT_1_BETTER,
                        2: DOCENT_1_2_EQUAL,
                        3: DOCENT_2_BETTER,
                        4: DOCENT_2_BETTER,
                    }[pairwise_ranking[ranking_type]]
                )

                ranking_diff = {0: 2, 1: 1, 2: 0, 3: -1, 4: -2}[pairwise_ranking[ranking_type]]
                diff_labels[ranking_type].append(ranking_diff)
                diff_preds[ranking_type].append(
                    model1_score[metric_type] - model2_score[metric_type]
                )

        correlations = {}
        for ranking_type in ["mistakes", "omissions", "overall_quality"]:
            spearman_r = spearmanr(diff_labels[ranking_type], diff_preds[ranking_type])
            kendall_tau = kendalltau(diff_labels[ranking_type], diff_preds[ranking_type])
            correlations = {
                **correlations,
                f"{ranking_type}_accuracy": accuracy_score(
                    acc_tie_labels[ranking_type], acc_tie_preds[ranking_type]
                ),
                f"{ranking_type}_spearman": spearman_r.statistic,
                f"{ranking_type}_spearman_pvalue": spearman_r.pvalue,
                f"{ranking_type}_kendall": kendall_tau.statistic,
                f"{ranking_type}_kendall_pvalue": kendall_tau.pvalue,
            }
        return correlations
    else:
        assert benchmark == "caparena"

        gold_tie_count = 0
        for pairwise_ranking in pairwise_rankings:
            if pairwise_ranking["winner"] == "equal":
                gold_tie_count += 1

        pred_score_differences = []
        for pairwise_ranking in pairwise_rankings:
            reference = pairwise_ranking["reference"]

            model1_pred_score = scores[(pairwise_ranking["model1_generation"], reference)]["f1"]
            model2_pred_score = scores[(pairwise_ranking["model2_generation"], reference)]["f1"]
            pred_score_differences.append(abs(model1_pred_score - model2_pred_score))
        pred_score_differences.sort()
        pred_tie_threshold_exc = pred_score_differences[gold_tie_count]

        gold_battles, pred_battles = [], []
        caption_level_agreed, caption_level_total = Counter(), Counter()
        for pairwise_ranking in pairwise_rankings:
            model1 = pairwise_ranking["model1"]
            model2 = pairwise_ranking["model2"]
            gold_winner = {
                "model1": model1,
                "model2": model2,
                "equal": "equal",
            }[pairwise_ranking["winner"]]

            gold_battles.append(
                (
                    model1,
                    model2,
                    gold_winner,
                )
            )

            reference = pairwise_ranking["reference"]
            model1_pred_score = scores[(pairwise_ranking["model1_generation"], reference)]["f1"]
            model2_pred_score = scores[(pairwise_ranking["model2_generation"], reference)]["f1"]

            if model1_pred_score - model2_pred_score >= pred_tie_threshold_exc:
                pred_winner = model1
            elif model2_pred_score - model1_pred_score >= pred_tie_threshold_exc:
                pred_winner = model2
            else:
                pred_winner = "equal"

            pred_battles.append((model1, model2, pred_winner))

            if gold_winner == pred_winner:
                caption_level_agreed["overall"] += 1
                caption_level_agreed[pairwise_ranking["cluster"]] += 1
            caption_level_total["overall"] += 1
            caption_level_total[pairwise_ranking["cluster"]] += 1

        gold_elo_rankings = calculate_bootstrap_elo(gold_battles, n_trials=n_trials)
        pred_elo_rankings = calculate_bootstrap_elo(pred_battles, n_trials=n_trials)

        assert gold_elo_rankings.keys() == pred_elo_rankings.keys(), (
            f"gold_elo_rankings.keys() != pred_elo_rankings.keys(): {gold_elo_rankings.keys()} != {pred_elo_rankings.keys()}"
        )

        return {
            **{
                f"caption_{level.replace(' ', '_')}": caption_level_agreed[level]
                / caption_level_total[level]
                for level in caption_level_total.keys()
            },
            "model_spearman": spearmanr(
                [gold_elo_rankings[model] for model in sorted(gold_elo_rankings.keys())],
                [pred_elo_rankings[model] for model in sorted(pred_elo_rankings.keys())],
            ).statistic,
            "model_kendall": kendalltau(
                [gold_elo_rankings[model] for model in sorted(gold_elo_rankings.keys())],
                [pred_elo_rankings[model] for model in sorted(pred_elo_rankings.keys())],
            ).statistic,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="docent", choices=["docent", "caparena"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--disable-prefix-caching", action="store_true", default=False)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--verbosity", type=str, choices=["quiet", "debug"], default="quiet")
    args = parser.parse_args()

    posh = PoSh(
        qa_gpu_memory_utilization=args.gpu_memory_utilization,
        qa_tensor_parallel_size=args.tensor_parallel_size,
        qa_enable_prefix_caching=not args.disable_prefix_caching,
        cache_dir=args.cache_dir,
        verbosity=args.verbosity
    )

    generations, references, cache_keys, pairwise_rankings = load_coarse_benchmark(args.benchmark)

    scores = {}
    for generation, reference, coarse_score in zip(
        generations,
        references,
        posh.evaluate(generations=generations, references=references, cache_keys=cache_keys),
    ):
        assert (generation, reference) not in scores
        scores[(generation, reference)] = coarse_score

    correlations = calculate_coarse_correlations(pairwise_rankings, scores, args.benchmark)

    print(json.dumps(correlations, indent=4))
