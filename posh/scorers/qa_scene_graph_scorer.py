import os
import json
import pickle
import numpy as np
from pathlib import Path
from ordered_set import OrderedSet
from more_itertools import chunked
from collections import defaultdict
from vllm import LLM, SamplingParams

HEAD_PRONOUNS = {
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
}

PART_OF_CONTAINS_EQUIVALENTS = {
    "contains": {
        "have",
        "has",
        "had",
        "having",
        "hold",
        "holds",
        "hold",
        "holding",
        "contain",
        "contains",
        "contained",
        "containing",
        "wear",
        "wears",
        "wore",
        "wearing",
    },
    "part of": {
        "part of",
    },
}

PROMPT_FILES = {
    "precision": "prompts/precision.json",
    "recall": "prompts/recall.json",
}


class FrozenDict:
    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __eq__(self, other):
        if isinstance(other, FrozenDict):
            return self._d == other._d
        elif isinstance(other, dict):
            return self._d == other
        return False

    def __hash__(self):
        return hash(tuple(sorted(self._d.items())))

    def __str__(self):
        return str(self._d)

    def __repr__(self):
        return repr(self._d)


class QASceneGraphScorer:
    def __init__(
        self,
        model_type="Qwen/Qwen3-14B",
        presence_threshold=2,
        max_tokens=2000,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        unofficial=False,
        batch_size=None,
        cache_dir=None,
        verbosity="quiet",
        keep_alive=False,
    ):
        self.model_type = model_type
        self.presence_threshold = presence_threshold
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_prefix_caching = enable_prefix_caching
        self.unofficial = unofficial
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.verbosity = verbosity
        self.keep_alive = keep_alive

        if self.unofficial:
            print(
                "PoSh is running in unofficial mode.  While faster, these scores are not replicable!"
            )
        else:
            os.environ["VLLM_USE_V1"] = "1"
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "1"

        self.prompts = {}
        for prompt_type, prompt_file in PROMPT_FILES.items():
            with open(os.path.join(Path(__file__).parent, prompt_file), "r") as f:
                self.prompts[prompt_type] = json.load(f)

        # used for keep_alive=True

        self.model = None

    def get_cache_dir(self):
        assert self.cache_dir is not None
        cache_keys = [
            self.model_type.split("/")[-1],
            str(self.presence_threshold),
            str(self.max_tokens),
        ]
        return os.path.join(self.cache_dir, "_".join(cache_keys))

    def extract_entities_attributes_relations(self, scene_graph, text):
        entities, attributes_by_entity, relations_by_entity = (
            OrderedSet(),
            defaultdict(OrderedSet),
            defaultdict(OrderedSet),
        )

        attributes_by_text = defaultdict(lambda: defaultdict(set))
        for entity_idx, entity in enumerate(scene_graph.get("entities", [])):
            entity_text = entity["head"].strip().lower()

            if (entity["quantity"] or "").strip() != "":
                attributes_by_text[entity_text][entity_idx].add(entity["quantity"].strip().lower())

            for attribute in entity["attributes"]:
                attribute = attribute.strip().lower()
                attributes_by_text[entity_text][entity_idx].add(attribute)

        for entity_idx, entity in enumerate(scene_graph.get("entities", [])):
            entity_text = entity["head"].strip().lower()

            entities.add(
                FrozenDict(
                    {
                        "text": entity_text,
                        "heads": tuple(head.strip().lower() for head in entity["heads"]),
                        "entity_idx": entity_idx,
                        "original_entity_idx": entity_idx,
                        "sentence_idxs": tuple(sorted(entity["sentence_idxs"])),
                        "text_spans": tuple(sorted(entity.get("text_spans", []))),
                        "is_us": entity.get("is_us", entity_text in {"us", "our", "we"}),
                        "is_left": entity.get("is_left", False),
                        "is_right": entity.get("is_right", False),
                    }
                )
            )

            if (entity["quantity"] or "").strip() != "":
                attributes_by_entity[entity_idx].add(
                    FrozenDict(
                        {
                            "text": entity["quantity"].strip().lower(),
                            "entity": entity_idx,
                            "original_entity_idx": entity_idx,
                            "original_attribute_idx": "quantity",
                            "sentence_idxs": (entity["quantity_sentence_idx"],)
                            if entity.get("quantity_sentence_idx", None) is not None
                            else (),
                            "is_unique": all(
                                [
                                    entity["quantity"].strip().lower()
                                    not in attributes_by_text[entity_text][other_entity_idx]
                                    for other_entity_idx in attributes_by_text[entity_text]
                                    if other_entity_idx != entity_idx
                                ]
                            ),
                            "text_spans": tuple(
                                entity.get("attribute_text_spans", {}).get(
                                    entity["quantity"].strip().lower(), []
                                )
                            ),
                        }
                    )
                )

            seen_attributes = set()
            for attribute_num, attribute in enumerate(sorted(entity["attributes"])):
                attribute = attribute
                sentence_idxs = tuple(
                    sorted(entity.get("attribute_sentence_idxs", {}).get(attribute, []))
                )

                attribute = attribute.strip().lower()
                if attribute in seen_attributes:
                    continue

                seen_attributes.add(attribute)
                attributes_by_entity[entity_idx].add(
                    FrozenDict(
                        {
                            "text": attribute,
                            "entity": entity_idx,
                            "original_entity_idx": entity_idx,
                            "original_attribute_idx": attribute_num,
                            "sentence_idxs": sentence_idxs,
                            "is_unique": all(
                                [
                                    attribute
                                    not in attributes_by_text[entity_text][other_entity_idx]
                                    for other_entity_idx in attributes_by_text[entity_text]
                                    if other_entity_idx != entity_idx
                                ]
                            ),
                            "text_spans": tuple(
                                entity.get("attribute_text_spans", {}).get(attribute, [])
                            ),
                        }
                    )
                )

        assert len(entities) == len(scene_graph.get("entities", []))

        relations_by_entity_pair = defaultdict(set)
        for relation in scene_graph.get("relations", []):
            entity1_idx = relation["subject"]
            entity2_idx = relation["object"]
            relations_by_entity_pair[(entity1_idx, entity2_idx)].add(relation["relation"])

        seen_relations = set()
        for relation_idx, relation in enumerate(scene_graph.get("relations", [])):
            entity1_idx = relation["subject"]
            entity2_idx = relation["object"]
            relation_text = relation["relation"]

            if (entity1_idx, entity2_idx, relation_text) in seen_relations:
                continue

            is_contains = relation["relation"] in PART_OF_CONTAINS_EQUIVALENTS["contains"]
            is_part_of = relation["relation"] in PART_OF_CONTAINS_EQUIVALENTS["part of"]

            if "sentence_idx" in relation:
                sentence_idxs = (relation["sentence_idx"],)
            elif "sentence_idxs" in relation:
                sentence_idxs = tuple(sorted(relation["sentence_idxs"]))

            seen_relations.add((entity1_idx, entity2_idx, relation_text))

            relations_by_entity[entity1_idx].add(
                FrozenDict(
                    {
                        "text": relation_text,
                        "entity1": entity1_idx,
                        "entity2": entity2_idx,
                        "original_relation_idx": relation_idx,
                        "is_contains": is_contains,
                        "is_part_of": is_part_of,
                        "sentence_idxs": sentence_idxs,
                        "text_spans": tuple(relation.get("text_spans", [])),
                    }
                )
            )
            relations_by_entity[entity2_idx].add(
                FrozenDict(
                    {
                        "text": relation_text,
                        "entity1": entity1_idx,
                        "entity2": entity2_idx,
                        "original_relation_idx": relation_idx,
                        "is_contains": is_contains,
                        "is_part_of": is_part_of,
                        "sentence_idxs": sentence_idxs,
                        "text_spans": tuple(relation.get("text_spans", [])),
                    }
                )
            )
        return entities, attributes_by_entity, relations_by_entity

    def get_lowest_granular_score(self):
        return 1

    def get_highest_granular_score(self):
        return 5

    def get_parent_entity_nums(self, entity_num, relations):
        return {
            *[
                relation["entity2"]
                for relation in relations[entity_num]
                if relation["is_part_of"] and relation["entity1"] == entity_num
            ],
            *[
                relation["entity1"]
                for relation in relations[entity_num]
                if relation["is_contains"] and relation["entity2"] == entity_num
            ],
        }

    def prepare_questions(
        self,
        entities,
        attributes,
        relations,
        source_text,
        target_text,
        answers,
        question_type,
        direction,
    ):
        prompts = self.prompts[direction]

        question_manifests = []
        if question_type in {"parent_entities", "child_entities"}:
            entity_name_collisions = defaultdict(lambda: defaultdict(list))
            for entity_num, entity in enumerate(entities):
                if entity["is_us"] or entity["is_left"] or entity["is_right"]:
                    continue

                entity_name_collisions["overall"][entity["text"]].append(entity_num)

                parent_entity_nums = self.get_parent_entity_nums(entity_num, relations)
                if len(parent_entity_nums) == 0:
                    continue

                for parent_entity_num in parent_entity_nums:
                    if parent_entity_num in answers and answers[parent_entity_num]["present"]:
                        entity_name_collisions[parent_entity_num][entity["text"]].append(entity_num)

            for entity_num, entity in enumerate(entities):
                if entity["is_us"] or entity["is_left"] or entity["is_right"]:
                    question_manifests.append(
                        {
                            "type": "entity",
                            "entity_num": entity_num,
                            "is_us": entity["is_us"],
                            "is_left": entity["is_left"],
                            "is_right": entity["is_right"],
                            "entity_identifier": "the viewer"
                            if entity["is_us"]
                            else "our left"
                            if entity["is_left"]
                            else "our right",
                            "entity_identifier_type": "default",
                            "question": "Please respond with 'yes'.",
                            "hash": entity["text"],
                        }
                    )
                    continue

                parent_entity_nums = self.get_parent_entity_nums(entity_num, relations)

                if question_type == "parent_entities" and len(parent_entity_nums) > 0:
                    continue
                elif question_type == "child_entities" and len(parent_entity_nums) == 0:
                    continue

                has_no_parents = (
                    question_type == "parent_entities" and len(parent_entity_nums) == 0
                ) or (
                    question_type == "child_entities"
                    and all(
                        [
                            not (
                                parent_entity_num in answers
                                and answers[parent_entity_num]["present"]
                            )
                            for parent_entity_num in parent_entity_nums
                        ]
                    )
                )

                formatters, formatters_collisions = [], []
                if has_no_parents or len(entity_name_collisions["overall"][entity["text"]]) == 1:
                    formatters.append(prompts["entity_formatters"]["no_parents"])
                    formatters_collisions.append(entity_name_collisions["overall"][entity["text"]])

                if not has_no_parents:
                    for parent_entity_num in sorted(parent_entity_nums):
                        if parent_entity_num in answers and answers[parent_entity_num]["present"]:
                            formatters.append(
                                prompts["entity_formatters"]["has_parents"].format(
                                    parent_entity_identifier=answers[parent_entity_num][
                                        "identifier"
                                    ],
                                )
                            )
                            formatters_collisions.append(
                                entity_name_collisions[parent_entity_num][entity["text"]]
                            )

                entity_identifiers, entity_identifier_sort_keys, entity_identifier_types = (
                    [],
                    [],
                    [],
                )
                for formatter, colliding_entity_nums in zip(formatters, formatters_collisions):
                    added_entity_identifier = False
                    if len(colliding_entity_nums) == 1:
                        added_entity_identifier = True
                        entity_identifiers.append(formatter.format(entity_text=entity["text"]))
                        entity_identifier_sort_keys.append(0)
                        entity_identifier_types.append("head")

                    colliding_attributes = {
                        attribute["text"]
                        for other_entity_num in colliding_entity_nums
                        for attribute in attributes[other_entity_num]
                        if entity_num != other_entity_num
                    }
                    for attribute in sorted(
                        attributes[entity_num],
                        key=lambda _attribute: min(_attribute["sentence_idxs"]),
                    ):
                        formatted_entity = formatter.format(entity_text=entity["text"])
                        entity_identifier = f"{attribute['text']} {formatted_entity}"
                        if (
                            attribute["text"] not in colliding_attributes
                            and entity_identifier not in entity_identifiers
                        ):
                            added_entity_identifier = True
                            entity_identifiers.append(entity_identifier)
                            entity_identifier_sort_keys.append(1)
                            entity_identifier_types.append("attribute")

                    colliding_heads = {
                        head
                        for other_entity_num in colliding_entity_nums
                        for head in entities[other_entity_num]["heads"]
                        if other_entity_num != entity_num
                    }
                    for head in entities[entity_num]["heads"]:
                        entity_identifier = formatter.format(entity_text=head)
                        if (
                            head.lower() not in HEAD_PRONOUNS
                            and head not in colliding_heads
                            and entity_identifier not in entity_identifiers
                        ):
                            added_entity_identifier = True
                            entity_identifiers.append(entity_identifier)
                            entity_identifier_sort_keys.append(2)
                            entity_identifier_types.append("head")

                    if added_entity_identifier:
                        continue

                    for relation_type in ["entity1", "entity2"]:
                        other_relation_type = "entity2" if relation_type == "entity1" else "entity1"
                        colliding_relations = {
                            (relation["text"], relation[other_relation_type])
                            for other_entity_num in colliding_entity_nums
                            for relation in relations[other_entity_num]
                            if other_entity_num != entity_num
                            and relation[relation_type] == other_entity_num
                        }
                        for relation in sorted(
                            relations[entity_num],
                            key=lambda _relation: min(_relation["sentence_idxs"]),
                        ):
                            # we handle part of relations separately
                            if relation["is_part_of"]:
                                continue

                            if relation[relation_type] != entity_num:
                                continue

                            # in particular, a unique relation
                            if (
                                relation["text"],
                                relation[other_relation_type],
                            ) in colliding_relations:
                                continue

                            other_entity_num = relation[other_relation_type]
                            if other_entity_num in answers:
                                if answers[other_entity_num]["present"]:
                                    other_entity_identifier = answers[other_entity_num][
                                        "identifier"
                                    ]
                                else:
                                    continue
                            else:
                                other_entity_identifier = entities[other_entity_num]["text"]
                                if (
                                    other_entity_num in attributes
                                    and len(attributes[other_entity_num]) > 0
                                ):
                                    first_attribute = min(
                                        attributes[other_entity_num],
                                        key=lambda _attribute: min(_attribute["sentence_idxs"]),
                                    )["text"]
                                else:
                                    first_attribute = ""
                                other_entity_identifier = (
                                    f"{first_attribute} {other_entity_identifier}".strip()
                                )

                            if relation_type == "entity1":
                                relation_text = (
                                    f"{entity['text']} {relation['text']} {other_entity_identifier}"
                                )
                            else:
                                assert relation_type == "entity2"
                                relation_text = (
                                    f"{other_entity_identifier} {relation['text']} {entity['text']}"
                                )

                            if relation_type == "entity1":
                                entity_identifier = relation_text
                            else:
                                entity_identifier = prompts["entity_formatters"]["relation"].format(
                                    formatted_entity=formatter.format(entity_text=entity["text"]),
                                    relation=relation_text,
                                )
                            if entity_identifier not in entity_identifiers:
                                entity_identifiers.append(entity_identifier)
                                if relation_type == "entity1":
                                    entity_identifier_sort_keys.append(3)
                                else:
                                    entity_identifier_sort_keys.append(4)
                                entity_identifier_types.append("relation")

                if len(entity_identifiers) == 0:
                    if self.verbosity == "debug":
                        print(
                            f"No entity identifiers found for {entity['text']}, falling back to head."
                        )
                    entity_identifiers.append(formatters[0].format(entity_text=entity["text"]))
                    entity_identifier_sort_keys.append(0)
                    entity_identifier_types.append("head")

                for idx in sorted(
                    range(len(entity_identifiers)),
                    key=lambda _idx: (entity_identifier_sort_keys[_idx], _idx),
                ):
                    entity_identifier = entity_identifiers[idx]
                    question = prompts["verifiers"]["entity"].format(
                        entity_identifier=entity_identifier
                    )
                    question_manifests.append(
                        {
                            "type": "entity",
                            "entity_num": entity_num,
                            "entity_identifier": entity_identifier,
                            "entity_identifier_type": entity_identifier_types[idx],
                            "question": question,
                            "is_us": False,
                            "is_left": False,
                            "is_right": False,
                            "hash": entity["text"],
                        }
                    )
        else:
            assert question_type == "attributes_relations"
            for entity_num, entity in enumerate(entities):
                if not answers[entity_num]["present"]:
                    continue

                entity_identifier = answers[entity_num]["identifier"]
                for attribute_num, attribute in enumerate(attributes[entity_num]):
                    question = prompts["verifiers"]["attribute"].format(
                        entity_identifier=entity_identifier, attribute=attribute["text"]
                    )
                    question_manifests.append(
                        {
                            "type": "attribute",
                            "entity_num": entity_num,
                            "attribute_num": attribute_num,
                            "original_attribute_idx": attribute["original_attribute_idx"],
                            "question": question,
                            "hash": (entity["text"], attribute["text"]),
                        }
                    )

            covered_relations = set()
            for entity_num, entity_relations in relations.items():
                for relation_num, relation in enumerate(entity_relations):
                    if relation["original_relation_idx"] in covered_relations:
                        continue

                    entity1 = relation["entity1"]
                    entity2 = relation["entity2"]
                    if not answers[entity1]["present"]:
                        continue

                    if not answers[entity2]["present"]:
                        continue

                    entity1_identifier = answers[entity1]["identifier"]
                    entity2_identifier = answers[entity2]["identifier"]

                    if (
                        relation["is_part_of"]
                        and " of " in entity1_identifier
                        and "]" in entity1_identifier
                    ):
                        entity1_identifier = (
                            entity1_identifier[: entity1_identifier.index(" of ")]
                            + entity1_identifier[entity1_identifier.index("]") + 1 :]
                        ).strip()
                    elif (
                        relation["is_contains"]
                        and " of " in entity2_identifier
                        and "]" in entity2_identifier
                    ):
                        entity2_identifier = (
                            entity2_identifier[: entity2_identifier.index(" of ")]
                            + entity2_identifier[entity2_identifier.index("]") + 1 :]
                        ).strip()

                    question = prompts["verifiers"]["relation"].format(
                        entity1_identifier=entity1_identifier,
                        entity2_identifier=entity2_identifier,
                        relation=relation["text"],
                    )

                    covered_relations.add(relation["original_relation_idx"])
                    question_manifests.append(
                        {
                            "type": "relation",
                            "entity_num": entity_num,
                            "relation_num": relation_num,
                            "original_relation_idx": relation["original_relation_idx"],
                            "question": question,
                            "hash": (
                                entities[entity1]["text"],
                                relation["text"],
                                entities[entity2]["text"],
                            ),
                        }
                    )

        questions = []
        for question_manifest in question_manifests:
            if "source_text" in prompts["prefix"]:
                prefix = prompts["prefix"].format(
                    source_text=source_text,
                    target_text=target_text,
                )
            else:
                prefix = prompts["prefix"].format(text=target_text)

            instruction = prompts["instructions"]
            questions.append(f"{prefix}\n\n{question_manifest['question']}\n\n{instruction}")

        return questions, question_manifests

    def parse_result(self, result):
        relevant_tokens = ["1", "2", "3", "4", "5"]

        token_types, token_logprobs = [], []
        for logprob in result.outputs[0].logprobs[0].values():
            for relevant_token in relevant_tokens:
                if relevant_token in logprob.decoded_token.lower():
                    token_types.append(relevant_token)
                    token_logprobs.append(logprob.logprob)

        if len(token_logprobs) == 0:
            return self.get_lowest_granular_score()

        max_token_logprob = max(token_logprobs)
        shifted_token_logprobs = [logprob - max_token_logprob for logprob in token_logprobs]
        exp_token_logprobs = [np.exp(shifted_logprob) for shifted_logprob in shifted_token_logprobs]
        sum_exp_shifted_token_logprobs = sum(exp_token_logprobs)

        token_probs = [
            exp_token_logprob / sum_exp_shifted_token_logprobs
            for exp_token_logprob in exp_token_logprobs
        ]

        assert len(token_types) == len(token_probs)

        response_probs = {}
        for token, token_prob in zip(token_types, token_probs):
            if token not in response_probs:
                response_probs[token] = token_prob
            else:
                response_probs[token] += token_prob

        weighted_sum = 0
        for token in relevant_tokens:
            weighted_sum += response_probs.get(token, 0) * float(token)
        return weighted_sum

    def _batch_calculate_granular_scores(
        self,
        generation_scene_graphs,
        reference_scene_graphs,
        generation_texts,
        reference_texts,
        cache_keys=None,
        overwrite_cache=False,
        cache_only=False,
    ):
        assert (
            len(generation_scene_graphs)
            == len(reference_scene_graphs)
            == len(generation_texts)
            == len(reference_texts)
        )

        assert all(
            map(
                lambda input: isinstance(input, list),
                [
                    generation_scene_graphs,
                    reference_scene_graphs,
                    generation_texts,
                    reference_texts,
                    cache_keys if cache_keys is not None else [],
                ],
            )
        ), "all inputs must be (ordered) lists"

        answers, cached_idxs = defaultdict(lambda: defaultdict(dict)), set()
        if self.cache_dir is not None and cache_keys is not None:
            assert len(cache_keys) == len(generation_scene_graphs)

            cache_dir = self.get_cache_dir()
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            for idx, cache_key in enumerate(cache_keys):
                cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

                if not os.path.exists(cache_path) or overwrite_cache:
                    if self.verbosity == "debug":
                        print(f"Cache miss for {cache_key}!")
                    continue

                cached_idxs.add(idx)
                with open(cache_path, "rb") as f:
                    answers[idx] = pickle.load(f)

        if len(answers) < len(generation_scene_graphs):
            assert not cache_only, (
                f"Had {len(generation_scene_graphs) - len(answers)} cache misses in {cache_dir} but cache_only is True!"
            )
            if self.model is not None:
                model = self.model
            else:
                model = LLM(
                    model=self.model_type,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    tensor_parallel_size=self.tensor_parallel_size,
                    trust_remote_code=True,
                    enable_prefix_caching=self.enable_prefix_caching,
                    max_model_len=self.max_tokens,
                )
                if self.keep_alive:
                    self.model = model
            tokenizer = model.get_tokenizer()
            sampling_params = {
                "top_p": 1,
                "n": 1,
                "max_tokens": 5,
                "logprobs": 20,
                "temperature": 1,
            }
            template_config = {"add_generation_prompt": True, "enable_thinking": False}

            for question_type in ["parent_entities", "child_entities", "attributes_relations"]:
                questions, question_manifests = [], []
                for idx, (
                    generation_scene_graph,
                    reference_scene_graph,
                    generation_text,
                    reference_text,
                ) in enumerate(
                    zip(
                        generation_scene_graphs,
                        reference_scene_graphs,
                        generation_texts,
                        reference_texts,
                    )
                ):
                    if idx in cached_idxs:
                        continue

                    generation_entities, generation_attributes, generation_relations = (
                        self.extract_entities_attributes_relations(
                            generation_scene_graph, generation_text
                        )
                    )
                    reference_entities, reference_attributes, reference_relations = (
                        self.extract_entities_attributes_relations(
                            reference_scene_graph, reference_text
                        )
                    )

                    precision_questions, precision_question_manifests = self.prepare_questions(
                        entities=generation_entities,
                        attributes=generation_attributes,
                        relations=generation_relations,
                        source_text=generation_text,
                        target_text=reference_text,
                        answers=answers[idx]["precision"],
                        question_type=question_type,
                        direction="precision",
                    )
                    recall_questions, recall_question_manifests = self.prepare_questions(
                        entities=reference_entities,
                        attributes=reference_attributes,
                        relations=reference_relations,
                        source_text=reference_text,
                        target_text=generation_text,
                        answers=answers[idx]["recall"],
                        question_type=question_type,
                        direction="recall",
                    )

                    if len(precision_question_manifests) > 0:
                        assert "idx" not in precision_question_manifests[0]
                        assert "direction" not in precision_question_manifests[0]

                    if len(recall_question_manifests) > 0:
                        assert "idx" not in recall_question_manifests[0]
                        assert "direction" not in recall_question_manifests[0]

                    questions.extend(precision_questions + recall_questions)
                    question_manifests.extend(
                        [
                            {"idx": idx, "direction": "precision", **precision_question_manifest}
                            for precision_question_manifest in precision_question_manifests
                        ]
                        + [
                            {"idx": idx, "direction": "recall", **recall_question_manifest}
                            for recall_question_manifest in recall_question_manifests
                        ]
                    )

                if question_type in {"parent_entities", "child_entities"}:
                    rewrite_idxs, rewrite_inputs = [], []
                    for question_idx, question_manifest in enumerate(question_manifests):
                        if question_manifest["entity_identifier_type"] not in {
                            "attribute",
                            "relation",
                        }:
                            continue

                        rewrite_idxs.append(question_idx)
                        rewrite_inputs.append(
                            {
                                "prompt_token_ids": tokenizer.encode(
                                    tokenizer.apply_chat_template(
                                        [
                                            {
                                                "role": "user",
                                                "content": self.prompts[
                                                    question_manifest["direction"]
                                                ]["rewrite"][
                                                    question_manifest["entity_identifier_type"]
                                                ].format(
                                                    entity_identifier=question_manifest[
                                                        "entity_identifier"
                                                    ]
                                                ),
                                            }
                                        ],
                                        tokenize=False,
                                        **template_config,
                                    )
                                )[: self.max_tokens - 1]
                            }
                        )

                    if len(rewrite_inputs) > 0:
                        rewrite_results = model.generate(
                            rewrite_inputs, SamplingParams(temperature=0, max_tokens=20, n=1)
                        )
                        for rewrite_idx, rewrite_result in zip(rewrite_idxs, rewrite_results):
                            original_identifier = question_manifests[rewrite_idx][
                                "entity_identifier"
                            ]
                            rewritten_identifier = rewrite_result.outputs[0].text

                            assert original_identifier in questions[rewrite_idx]

                            questions[rewrite_idx] = rewritten_identifier.join(
                                questions[rewrite_idx].rsplit(original_identifier, 1)
                            )
                            question_manifests[rewrite_idx]["entity_identifier"] = (
                                rewritten_identifier
                            )

                batch_inputs = []
                for question, question_manifest in zip(questions, question_manifests):
                    inputs = {
                        "prompt_token_ids": tokenizer.encode(
                            tokenizer.apply_chat_template(
                                [
                                    {
                                        "role": "user",
                                        "content": question,
                                    }
                                ],
                                tokenize=False,
                                **template_config,
                            )
                        )[: self.max_tokens - 1]
                    }
                    batch_inputs.append(inputs)

                results = model.generate(
                    batch_inputs,
                    SamplingParams(**sampling_params),
                )

                if self.verbosity == "debug":
                    print(f"Question Type: {question_type}")
                    for question, result in zip(questions, results):
                        print(question)
                        print(result.outputs[0].text)
                        print(self.parse_result(result))
                        print()

                assert len(results) == len(questions) == len(question_manifests)

                for question, result, question_manifest in zip(
                    questions, results, question_manifests
                ):
                    idx = question_manifest["idx"]
                    direction = question_manifest["direction"]

                    if question_manifest["type"] == "entity" and (
                        question_manifest["is_us"]
                        or question_manifest["is_left"]
                        or question_manifest["is_right"]
                    ):
                        parsed = self.get_highest_granular_score()
                    else:
                        parsed = self.parse_result(result)

                    if question_manifest["type"] == "entity":
                        entity_num = question_manifest["entity_num"]

                        # our candidate identifiers are ordered according to how
                        # discriminating we think they are so we use the first highest scoring one
                        if entity_num in answers[idx][direction] and (
                            round(parsed) <= round(answers[idx][direction][entity_num]["score"])
                        ):
                            continue

                        present = parsed >= self.presence_threshold
                        answers[idx][direction][entity_num] = {
                            "present": present,
                            "identifier": question_manifest["entity_identifier"],
                            "score": parsed,
                            "question": question,
                            "response": result.outputs[0].text,
                            "hash": question_manifest["hash"],
                        }
                    else:
                        assert question_manifest["type"] in {"attribute", "relation"}

                        answer_key = (
                            question_manifest["type"],
                            question_manifest["entity_num"],
                            question_manifest["attribute_num"]
                            if question_manifest["type"] == "attribute"
                            else question_manifest["relation_num"],
                        )

                        # we only check for each attribute / relation once
                        assert answer_key not in answers[idx][direction], answer_key

                        answers[idx][direction][answer_key] = {
                            "score": parsed,
                            "question": question,
                            "response": result.outputs[0].text,
                            "hash": question_manifest["hash"],
                        }

                if self.verbosity == "debug":
                    for idx in answers:
                        print(f"Generation {idx}:")
                        for direction, direction_answers in answers[idx].items():
                            print(f"{direction}:")
                            for key, value in direction_answers.items():
                                print(f"  {key}: {value}")
                            print()

        if self.cache_dir is not None and cache_keys is not None:
            for idx, cache_key in enumerate(cache_keys):
                cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
                if not os.path.exists(cache_path) or overwrite_cache:
                    with open(cache_path, "wb") as f:
                        pickle.dump(answers[idx], f)

        return [
            {
                "generation": self.extract_entities_attributes_relations(
                    generation_scene_graph, generation_text
                ),
                "reference": self.extract_entities_attributes_relations(
                    reference_scene_graph, reference_text
                ),
                "answers": answers[idx],
            }
            for idx, (
                generation_scene_graph,
                reference_scene_graph,
                generation_text,
                reference_text,
            ) in enumerate(
                zip(
                    generation_scene_graphs,
                    reference_scene_graphs,
                    generation_texts,
                    reference_texts,
                )
            )
        ]

    def batch_calculate_granular_scores(
        self,
        generation_scene_graphs,
        reference_scene_graphs,
        generation_texts,
        reference_texts,
        cache_keys=None,
        overwrite_cache=False,
        cache_only=False,
    ):
        if self.batch_size is None:
            return self._batch_calculate_granular_scores(
                generation_scene_graphs=generation_scene_graphs,
                reference_scene_graphs=reference_scene_graphs,
                generation_texts=generation_texts,
                reference_texts=reference_texts,
                cache_keys=cache_keys,
                overwrite_cache=overwrite_cache,
                cache_only=cache_only,
            )
        else:
            granular_scores = []
            for (
                batch_generation_scene_graphs,
                batch_reference_scene_graphs,
                batch_generation_texts,
                batch_reference_texts,
                batch_cache_keys,
            ) in zip(
                chunked(generation_scene_graphs, self.batch_size),
                chunked(reference_scene_graphs, self.batch_size),
                chunked(generation_texts, self.batch_size),
                chunked(reference_texts, self.batch_size),
                chunked(
                    cache_keys if cache_keys is not None else [None] * len(generation_scene_graphs),
                    self.batch_size,
                ),
            ):
                granular_scores.extend(
                    self._batch_calculate_granular_scores(
                        generation_scene_graphs=batch_generation_scene_graphs,
                        reference_scene_graphs=batch_reference_scene_graphs,
                        generation_texts=batch_generation_texts,
                        reference_texts=batch_reference_texts,
                        cache_keys=batch_cache_keys,
                        overwrite_cache=overwrite_cache,
                        cache_only=cache_only,
                    )
                )
            return granular_scores

    def calculate_granular_scores(
        self,
        generation_scene_graph,
        reference_scene_graph,
        generation_text,
        reference_text,
        cache_key=None,
        overwrite_cache=False,
    ):
        return self.batch_calculate_granular_scores(
            generation_scene_graphs=[generation_scene_graph],
            reference_scene_graphs=[reference_scene_graph],
            generation_texts=[generation_text],
            reference_texts=[reference_text],
            cache_keys=[cache_key] if cache_key is not None else None,
            overwrite_cache=overwrite_cache,
        )[0]

    def enrich_answers(self, relations, answers):
        for entity_num, entity_relations in relations.items():
            for relation_num, relation in enumerate(entity_relations):
                if ("relation", entity_num, relation_num) in answers:
                    answers[("relation", relation["original_relation_idx"])] = answers[
                        ("relation", entity_num, relation_num)
                    ]

    def batch_calculate_coarse_scores(
        self,
        generation_scene_graphs,
        reference_scene_graphs,
        generation_texts,
        reference_texts,
        cache_keys=None,
        overwrite_cache=False,
        granular_scores=None,
    ):
        if granular_scores is None:
            granular_scores = self.batch_calculate_granular_scores(
                generation_scene_graphs=generation_scene_graphs,
                reference_scene_graphs=reference_scene_graphs,
                generation_texts=generation_texts,
                reference_texts=reference_texts,
                cache_keys=cache_keys,
                overwrite_cache=overwrite_cache,
            )

        assert (
            len(generation_scene_graphs)
            == len(reference_scene_graphs)
            == len(generation_texts)
            == len(reference_texts)
            == len(granular_scores)
        )

        coarse_scores = []
        for idx, granular_score in enumerate(granular_scores):
            coarse_score = {}
            for score_type, score_target in [
                ("precision", "generation"),
                ("recall", "reference"),
            ]:
                entities, attributes, relations = granular_score[score_target]

                granular_answers = granular_score["answers"][score_type]
                self.enrich_answers(relations, granular_answers)

                seen_relations = set()
                all_scores, entity_scores, attribute_scores, relation_scores = [], {}, [], []
                grouped_attribute_scores, grouped_relation_scores = (
                    defaultdict(list),
                    defaultdict(list),
                )
                for entity_num, entity in enumerate(entities):
                    if entity["is_us"] or entity["is_left"] or entity["is_right"]:
                        continue

                    if entity_num not in granular_answers:
                        raise Exception(
                            f"Entity {entity_num} not in answers: {entity['text']} {generation_texts[idx]} {reference_texts[idx]}."
                        )
                    else:
                        assert entity["text"] == granular_answers[entity_num]["hash"]

                        entity_present = granular_answers[entity_num]["present"]
                        entity_score = granular_answers[entity_num]["score"]
                        all_scores.append(entity_score)
                        entity_scores[entity_num] = entity_score

                    for attribute_num, attribute in enumerate(attributes[entity_num]):
                        if not entity_present:
                            attribute_score = self.get_lowest_granular_score()
                        else:
                            assert (
                                entity["text"],
                                attribute["text"],
                            ) == granular_answers[("attribute", entity_num, attribute_num)]["hash"]
                            attribute_score = granular_answers[
                                ("attribute", entity_num, attribute_num)
                            ]["score"]

                        all_scores.append(attribute_score)
                        attribute_scores.append(attribute_score)
                        grouped_attribute_scores[entity_num].append(attribute_score)

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
                            assert (
                                entities[relation["entity1"]]["text"],
                                relation["text"],
                                entities[relation["entity2"]]["text"],
                            ) == granular_answers[("relation", relation["original_relation_idx"])][
                                "hash"
                            ]
                            relation_score = granular_answers[
                                ("relation", relation["original_relation_idx"])
                            ]["score"]
                        else:
                            relation_score = self.get_lowest_granular_score()

                        all_scores.append(relation_score)
                        relation_scores.append(relation_score)
                        grouped_relation_scores[relation["entity1"]].append(relation_score)
                        grouped_relation_scores[relation["entity2"]].append(relation_score)

                coarse_score[score_type] = (
                    np.mean(all_scores)
                    if len(all_scores) > 0
                    else self.get_highest_granular_score()
                ) / self.get_highest_granular_score()

            coarse_score["f1"] = (
                2
                * (coarse_score["precision"] * coarse_score["recall"])
                / (coarse_score["precision"] + coarse_score["recall"] or 1.0)
            )

            coarse_scores.append(coarse_score)

        return coarse_scores

    def calculate_coarse_scores(
        self,
        generation_scene_graph,
        reference_scene_graph,
        generation_text,
        reference_text,
        cache_key=None,
        overwrite_cache=False,
        granular_scores=None,
    ):
        return self.batch_calculate_coarse_scores(
            generation_scene_graphs=[generation_scene_graph],
            reference_scene_graphs=[reference_scene_graph],
            generation_texts=[generation_text],
            reference_texts=[reference_text],
            cache_keys=[cache_key] if cache_key is not None else None,
            overwrite_cache=overwrite_cache,
            granular_scores=[granular_scores] if granular_scores is not None else None,
        )[0]
