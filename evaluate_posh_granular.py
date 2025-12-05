import os
import json
import math
import spacy
import pickle
import random
import string
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from tabulate import tabulate
from operator import itemgetter
from datasets import load_dataset
from collections import defaultdict
from intervaltree import Interval, IntervalTree

from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score

from posh.posh import PoSh
from posh.graphs.text_graphs import SceneGraphExtractor

LABELED_TOKENS_TO_SKIP = set(string.punctuation) | {""}


def get_texts(docent):
    all_texts = set()
    for example in docent:
        all_texts.update([example["reference"], example["generation"]])
    return list(sorted(all_texts))


def get_parsed_texts(texts):
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    return {
        text: doc
        for text, doc in zip(
            texts, tqdm(nlp.pipe(texts), total=len(texts), desc="parsing texts")
        )
    }


def get_sgs(texts, cache_dir):
    scene_graph_extractor = SceneGraphExtractor(cache_dir=cache_dir)
    return {
        text: sg
        for text, sg in zip(
            tqdm(texts, desc="getting sgs"), scene_graph_extractor.get_graphs(texts)
        )
    }


def get_actual_components(docent, parsed_texts):
    actual_components = {}
    for item in tqdm(docent, desc="getting actual components"):
        key = f"{item['uuid']}-{item['model']}"
        assert key not in actual_components, key

        actual_components[key] = {}
        for error_type, text_type in [
            ("mistakes", "generation"),
            ("omissions", "reference"),
        ]:
            labeled_spans = IntervalTree()
            for annotation in json.loads(item[error_type]):
                labeled_spans.add(
                    Interval(annotation["start"], annotation["end"], True)
                )

            text = item[text_type]
            actual_components[key][error_type] = []
            for sentence in parsed_texts[text].sents:
                for token in sentence:
                    if token.text.strip().lower() in LABELED_TOKENS_TO_SKIP:
                        continue

                    is_error = labeled_spans.overlaps(
                        token.idx, token.idx + len(token.text)
                    )
                    actual_components[key][error_type].append(
                        {
                            "span": (token.idx, token.idx + len(token.text)),
                            "text": token.text,
                            "is_error": is_error,
                        }
                    )
    return actual_components


def get_random_predictions(actual_components):
    random.seed(42)

    predicted_component_scores = {}
    for actual_component_key in sorted(actual_components.keys()):
        predicted_component_scores[actual_component_key] = {}
        for label_type in sorted(actual_components[actual_component_key].keys()):
            predicted_component_scores[actual_component_key][label_type] = []
            for _ in range(len(actual_components[actual_component_key][label_type])):
                predicted_component_scores[actual_component_key][label_type].append(
                    random.random()
                )
    return predicted_component_scores


def get_embeddings_predictions(
    docent, actual_components, segmenter, default_value, embedding_model
):
    sentence_embedder = SentenceTransformer(embedding_model, device="cuda")

    predicted_component_scores = {}
    for item in tqdm(docent, desc="getting embedding predictions"):
        key = f"{item['uuid']}-{item['model']}"
        assert key not in predicted_component_scores, key
        predicted_component_scores[key] = {"mistakes": [], "omissions": []}

        generation_segments = segmenter(item["generation"])
        reference_segments = segmenter(item["reference"])

        generation_segment_embeddings = sentence_embedder.encode(
            list(map(itemgetter(0), generation_segments)),
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=False,
        )
        reference_segment_embeddings = sentence_embedder.encode(
            list(map(itemgetter(0), reference_segments)),
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=False,
        )

        sims = sentence_embedder.similarity(
            generation_segment_embeddings, reference_segment_embeddings
        )

        sims = (sims + 1) / 2

        assert sims.min() >= -0.001, sims.min()
        assert sims.max() <= 1.001, sims.max()

        span_scores = {"mistakes": defaultdict(list), "omissions": defaultdict(list)}
        for i, (_, generation_spans) in enumerate(generation_segments):
            for j, (_, reference_spans) in enumerate(reference_segments):
                sim = sims[i, j]
                for generation_span in generation_spans:
                    span_scores["mistakes"][generation_span].append(sim)

                for reference_span in reference_spans:
                    span_scores["omissions"][reference_span].append(sim)

        for error_type in ["mistakes", "omissions"]:
            for actual_component in actual_components[key][error_type]:
                predicted_component_scores[key][error_type].append(
                    max(
                        span_scores[error_type][actual_component["span"]],
                        default=default_value,
                    )
                )

    return predicted_component_scores


NGRAM_UNCHECKED_SCORE = 0.0


def get_n_grams(text, parsed_texts, n):
    ngrams = []
    for sentence in parsed_texts[text].sents:
        for i in range(len(sentence) - n + 1):
            ngram = sentence[i : i + n]
            ngrams.append(
                (
                    ngram.text,
                    [(token.idx, token.idx + len(token.text)) for token in ngram],
                )
            )
    return ngrams


def get_n_gram_embeddings_predictions(
    docent, parsed_texts, ngram_size, actual_components, embedding_model
):
    return get_embeddings_predictions(
        docent,
        actual_components,
        segmenter=partial(get_n_grams, parsed_texts=parsed_texts, n=ngram_size),
        default_value=NGRAM_UNCHECKED_SCORE,
        embedding_model=embedding_model,
    )


SG_UNCHECKED_SCORE = 1.0


def get_scene_graph_elements(text, parsed_texts, scene_graphs):
    spans = IntervalTree()
    for sentence in parsed_texts[text].sents:
        for token in sentence:
            spans.add(
                Interval(
                    token.idx,
                    token.idx + len(token.text),
                    (token.idx, token.idx + len(token.text)),
                )
            )

    elements = []
    for entity in scene_graphs[text]["entities"]:
        elements.append(
            (
                entity["head"],
                [span for s, e in entity["text_spans"] for _, _, span in spans[s:e]],
            )
        )
        for attribute in entity["attributes"]:
            elements.append(
                (
                    f"{attribute} {entity['head']}",
                    [
                        span
                        for s, e in entity["attribute_text_spans"][attribute]
                        for _, _, span in spans[s:e]
                    ],
                )
            )

    for relation in scene_graphs[text]["relations"]:
        relation_head = scene_graphs[text]["entities"][relation["subject"]]["head"]
        relation_tail = scene_graphs[text]["entities"][relation["object"]]["head"]
        elements.append(
            (
                f"{relation_head} {relation['relation']} {relation_tail}",
                [span for s, e in relation["text_spans"] for _, _, span in spans[s:e]],
            )
        )
    return elements


def get_scene_graph_embeddings_predictions(
    docent, parsed_texts, scene_graphs, actual_components, embedding_model
):
    return get_embeddings_predictions(
        docent,
        actual_components,
        segmenter=partial(
            get_scene_graph_elements,
            parsed_texts=parsed_texts,
            scene_graphs=scene_graphs,
        ),
        default_value=SG_UNCHECKED_SCORE,
        embedding_model=embedding_model,
    )


POSH_MISSING_SCORE = 1.0
POSH_UNCHECKED_SCORE = 5.0


def get_posh_predictions(docent, actual_components, cache_dir):
    posh = PoSh(
        cache_dir=args.cache_dir,
        verbosity="quiet",
    )

    keys, generations, references = [], [], []
    for item in docent:
        keys.append(f"{item['uuid']}-{item['model']}")
        generations.append(item["generation"])
        references.append(item["reference"])

    granular_scores = posh.evaluate(
        generations=generations,
        references=references,
        cache_keys=keys if cache_dir else None,
    )[0]

    predicted_component_scores = defaultdict(lambda: defaultdict(list))
    for key, granular_score in zip(keys, granular_scores):
        for error_type, score_type, score_target in [
            ("mistakes", "precision", "generation"),
            ("omissions", "recall", "reference"),
        ]:
            entities, attributes, relations = granular_score[score_target]
            granular_answers = granular_score["answers"][score_type]

            span_scores, seen_relations = IntervalTree(), set()
            for entity_num in range(len(entities)):
                assert entity_num in granular_answers

                entity_present = granular_answers[entity_num]["present"]
                entity_score = granular_answers[entity_num]["score"]

                for text_span in entities[entity_num]["text_spans"]:
                    span_scores.add(Interval(*text_span, entity_score))

                for attribute_num, attribute in enumerate(attributes[entity_num]):
                    if not entity_present:
                        attribute_score = POSH_MISSING_SCORE
                    else:
                        attribute_score = granular_answers[
                            ("attribute", entity_num, attribute_num)
                        ]["score"]

                    for text_span in attribute["text_spans"]:
                        span_scores.add(Interval(*text_span, attribute_score))

                for relation in relations[entity_num]:
                    if relation["original_relation_idx"] in seen_relations:
                        continue

                    seen_relations.add(relation["original_relation_idx"])
                    other_entity_num = (
                        relation["entity2"]
                        if relation["entity1"] == entity_num
                        else relation["entity1"]
                    )
                    if entity_present and granular_answers[other_entity_num]["present"]:
                        relation_score = granular_answers[
                            ("relation", relation["original_relation_idx"])
                        ]["score"]
                    else:
                        relation_score = POSH_MISSING_SCORE

                    for text_span in relation["text_spans"]:
                        span_scores.add(Interval(*text_span, relation_score))

            for actual_component in actual_components[key][error_type]:
                span = actual_component["span"]
                predicted_component_scores[key][error_type].append(
                    max(
                        map(itemgetter(-1), span_scores[span[0] : span[1]]),
                        default=POSH_UNCHECKED_SCORE,
                    )
                )

    return predicted_component_scores


def get_max_f1(predicted_component_scores, actual_components, n_thresholds=10):
    assert set(predicted_component_scores.keys()) == set(actual_components.keys())

    preds_by_error_type, labels_by_error_type = defaultdict(list), defaultdict(list)
    for actual_component_key in sorted(actual_components.keys()):
        assert (
            predicted_component_scores[actual_component_key].keys()
            == actual_components[actual_component_key].keys()
        )
        for error_type in sorted(actual_components[actual_component_key].keys()):
            assert len(
                predicted_component_scores[actual_component_key][error_type]
            ) == len(actual_components[actual_component_key][error_type])
            for i, actual_component in enumerate(
                actual_components[actual_component_key][error_type]
            ):
                preds_by_error_type[error_type].append(
                    predicted_component_scores[actual_component_key][error_type][i]
                )
                labels_by_error_type[error_type].append(
                    {False: 0, True: 1}[actual_component["is_error"]]
                )

    best_thresholds, best_f1s = {}, {}
    for error_type in sorted(preds_by_error_type.keys()):
        preds = np.array(preds_by_error_type[error_type])
        for threshold in np.linspace(preds.min(), preds.max(), n_thresholds):
            preds_thresholded = preds <= threshold
            f1 = f1_score(
                labels_by_error_type[error_type], preds_thresholded, average="macro"
            )
            if error_type not in best_f1s or f1 > best_f1s[error_type]:
                best_f1s[error_type] = f1
                best_thresholds[error_type] = threshold
    return best_thresholds, best_f1s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-embedding-model",
        type=str,
        choices=["all-mpnet-base-v2", "Qwen/Qwen3-Embedding-8B"],
        default="Qwen/Qwen3-Embedding-8B",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    docent = load_dataset("amitha/docent-eval-granular", split="test")
    docent_texts = get_texts(docent)
    docent_parsed_texts = get_parsed_texts(docent_texts)
    docent_sgs = get_sgs(
        docent_texts,
        os.path.join(args.cache_dir, "scene_graphs") if args.cache_dir else None,
    )
    actual_components = get_actual_components(docent, docent_parsed_texts)

    scores = []
    for name, predicted_component_scores in [
        ("random", get_random_predictions(actual_components)),
        (
            "ngram-embeddings-4",
            get_n_gram_embeddings_predictions(
                docent,
                docent_parsed_texts,
                ngram_size=4,
                actual_components=actual_components,
                embedding_model=args.baseline_embedding_model,
            ),
        ),
        (
            "sg-embeddings",
            get_scene_graph_embeddings_predictions(
                docent,
                docent_parsed_texts,
                docent_sgs,
                actual_components,
                embedding_model=args.baseline_embedding_model,
            ),
        ),
        ("posh", get_posh_predictions(docent, actual_components, args.cache_dir)),
    ]:
        thresholds, f1s = get_max_f1(predicted_component_scores, actual_components)
        scores.append(
            [
                name,
                thresholds["mistakes"],
                thresholds["omissions"],
                f1s["mistakes"],
                f1s["omissions"],
            ]
        )

    print(
        tabulate(
            scores,
            headers=[
                "Model",
                "Mistakes Threshold",
                "Omissions Threshold",
                "Mistakes F1",
                "Omissions F1",
            ],
            tablefmt="grid",
        )
    )
