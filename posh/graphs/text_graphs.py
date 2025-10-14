import os
import pickle
from tqdm import tqdm
from operator import itemgetter
from collections import defaultdict
from typing import List, TypedDict, Tuple, Dict

import spacy
from maverick import Maverick

Entity = TypedDict("Entity", {"head": str, "quantity": str, "attributes": List[str]})
Relation = TypedDict("Relation", {"subject": int, "relation": str, "object": int})
TextGraph = TypedDict("TextGraph", {"entities": List[Entity], "relations": List[Relation]})

CoreferenceResult = TypedDict(
    "CoreferenceResult",
    {
        "tokens": List[str],
        "clusters_token_offsets": List[List[Tuple[int, int]]],
        "clusters_token_text": List[List[str]],
    },
)


class SceneGraphExtractor:
    def __init__(
        self,
        spacy_model_name: str = "en_core_web_trf",
        coref_model_name: str = "sapienzanlp/maverick-mes-ontonotes",
        parse_batch_size: int = 32,
        cache_dir: str = None,
        verbosity: str = "quiet",
        keep_alive: bool = False,
    ):
        self.spacy_model_name = spacy_model_name
        self.coref_model_name = coref_model_name
        self.parse_batch_size = parse_batch_size
        self.cache_dir = cache_dir
        self.verbosity = verbosity
        self.keep_alive = keep_alive

        self.nlp = None
        self.maverick = None
        spacy.require_gpu()

    def get_cache_path(self) -> str:
        assert self.cache_dir is not None, "cache directory not set"
        cache_path_suffix = "__".join(
            [self.spacy_model_name.replace("_", "-"), self.coref_model_name.split("/")[-1]]
        )
        return os.path.join(self.cache_dir, f"{cache_path_suffix}.pkl")

    def get_parsed_docs(self, texts: List[str]) -> List[spacy.tokens.Doc]:
        if self.nlp is not None:
            nlp = self.nlp
        else:
            nlp = spacy.load(self.spacy_model_name)
            if self.keep_alive:
                self.nlp = nlp

        return list(
            tqdm(
                nlp.pipe(texts, batch_size=self.parse_batch_size),
                desc="parsing texts",
                leave=False,
                disable=self.verbosity == "quiet",
            )
        )

    def get_coreferences(self, docs: List[spacy.tokens.Doc]) -> List[CoreferenceResult]:
        if self.maverick is not None:
            maverick = self.maverick
        else:
            maverick = Maverick(self.coref_model_name)
            if self.keep_alive:
                self.maverick = maverick

        coreferences = []
        for doc in tqdm(
            docs, desc="getting coreferences", leave=False, disable=self.verbosity == "quiet"
        ):
            tokenized_sentences = []
            for sent in doc.sents:
                tokenized_sentences.append([token.text for token in sent])
            coreferences.append(maverick.predict(tokenized_sentences))

        return coreferences

    def get_token_cluster_mapping(
        self,
        tokenized_sentence: List[List[str]],
        tokenized_sentence_pos: List[List[str]],
        coreference: CoreferenceResult,
    ) -> Dict[Tuple[int, int], int]:
        flattened_token_to_token_mapping = {}
        for sentence_num, sentence_tokenized in enumerate(tokenized_sentence):
            for token_num in range(len(sentence_tokenized)):
                flattened_token_to_token_mapping[len(flattened_token_to_token_mapping)] = (
                    sentence_num,
                    token_num,
                )

        token_cluster_mapping = {}
        for cluster_num, cluster_members in enumerate(coreference["clusters_token_offsets"]):
            if len(cluster_members) <= 1:
                continue

            for cluster_member_s_inc, cluster_member_end_inc in cluster_members:
                cluster_member_sentence_num, first_noun, first_pronoun, cluster_member_range = (
                    None,
                    [],
                    None,
                    [],
                )
                for token_num in range(cluster_member_s_inc, cluster_member_end_inc + 1):
                    cluster_sentence_num, cluster_sentence_token_num = (
                        flattened_token_to_token_mapping[token_num]
                    )

                    if cluster_member_sentence_num is None:
                        cluster_member_sentence_num = cluster_sentence_num
                    elif cluster_sentence_num != cluster_member_sentence_num:
                        continue

                    # the coreference model, on occasion, includes dependent clauses in mentions
                    # which could result in incorrect merges if used naively; here we only consider
                    # up to the first contiguous noun / noun phrase in a mention to avoid this

                    if tokenized_sentence_pos[cluster_sentence_num][cluster_sentence_token_num] in {
                        "NOUN",
                        "PROPN",
                    }:
                        first_noun.append(cluster_sentence_token_num)
                    elif len(first_noun) > 0:
                        # here, we want to 1) ignore possessive markers and 2) noun-adjective modifiers
                        # that are sometimes included in cluster member spans in addition to the actual coreference
                        if tokenized_sentence[cluster_sentence_num][cluster_sentence_token_num] in {
                            "’s",
                            "'s",
                            "s",
                            "-",
                        }:
                            first_noun = []
                        else:
                            break

                    if (
                        tokenized_sentence_pos[cluster_sentence_num][cluster_sentence_token_num]
                        == "PRON"
                        and first_pronoun is None
                    ):
                        first_pronoun = cluster_sentence_token_num

                    cluster_member_range.append(cluster_sentence_token_num)

                if len(first_noun) > 0:
                    cluster_member_range = first_noun
                elif first_pronoun is not None:
                    cluster_member_range = [first_pronoun]

                for cluster_member_token_num in cluster_member_range:
                    if (
                        cluster_member_sentence_num,
                        cluster_member_token_num,
                    ) not in token_cluster_mapping or (
                        cluster_member_end_inc - cluster_member_s_inc + 1
                        < token_cluster_mapping[
                            (cluster_member_sentence_num, cluster_member_token_num)
                        ][1]
                    ):
                        token_cluster_mapping[
                            (cluster_member_sentence_num, cluster_member_token_num)
                        ] = (cluster_num, cluster_member_end_inc - cluster_member_s_inc + 1)

        return token_cluster_mapping

    def get_graph(self, doc: spacy.tokens.Doc, coreferences: CoreferenceResult) -> TextGraph:
        sentences = []
        tokenized_sentences, tokenized_sentences_pos, sentence_token_to_token_num = [], [], {}
        for sentence_num, sent in enumerate(doc.sents):
            sentences.append(sent.text.strip())
            tokenized_sentence_text, tokenized_sentence_pos = [], []
            for token_num, token in enumerate(sent):
                tokenized_sentence_text.append(token.text)
                tokenized_sentence_pos.append(token.pos_)
                sentence_token_to_token_num[(sentence_num, token_num)] = len(
                    sentence_token_to_token_num
                )
            tokenized_sentences.append(tokenized_sentence_text)
            tokenized_sentences_pos.append(tokenized_sentence_pos)

        token_cluster_mapping, cluster_token_mapping = {}, defaultdict(list)
        for (sentence_num, sentence_token_num), (cluster_num, _) in sorted(
            self.get_token_cluster_mapping(
                tokenized_sentences, tokenized_sentences_pos, coreferences
            ).items()
        ):
            token_cluster_mapping[
                sentence_token_to_token_num[(sentence_num, sentence_token_num)]
            ] = cluster_num
            cluster_token_mapping[cluster_num].append(
                sentence_token_to_token_num[(sentence_num, sentence_token_num)]
            )

        left_tokens, right_tokens = [], []
        for token_num in range(len(doc)):
            if token_num == 0:
                continue

            if doc[token_num - 1].text.lower() != "our":
                continue

            if doc[token_num].text.lower() == "left":
                left_tokens.append(token_num)
            elif doc[token_num].text.lower() == "right":
                right_tokens.append(token_num)

        left_cluster = max(cluster_token_mapping.keys(), default=-1) + 1
        for left_token in left_tokens:
            token_cluster_mapping[left_token] = left_cluster
            cluster_token_mapping[left_cluster].append(left_token)

        right_cluster = max(cluster_token_mapping.keys(), default=-1) + 1
        for right_token in right_tokens:
            token_cluster_mapping[right_token] = right_cluster
            cluster_token_mapping[right_cluster].append(right_token)

        components = []
        for sent_idx, sent in enumerate(doc.sents):
            # track processed tokens to avoid duplication
            processed_tokens = set()

            # keep track of token indices to component indices mapping
            token_to_component_idx = {}

            # track tokens that function as adjectives but might be nouns
            adjectival_noun_tokens = set()

            # pre-scan to identify noun compounds functioning as adjectives
            for token in sent:
                # check for noun compounds that modify other nouns
                if token.pos_ in ("NOUN", "PROPN") and token.dep_ in ("amod", "nmod", "compound"):
                    if token.head.pos_ in ("NOUN", "PROPN") and token.head != token:
                        # this noun is modifying another noun - mark it as adjectival
                        adjectival_noun_tokens.add(token.i)

            # first pass: create all components without relationships
            # 1. extract nouns first (but skip those that function as adjectives)
            for token in sent:
                if (
                    token.pos_ in ("NOUN", "PRON", "PROPN")
                    and not (token.dep_ == "conj" and token.head and token.head.pos_ == "ADJ")
                    and token.i not in adjectival_noun_tokens
                    and token.i not in processed_tokens
                ):
                    # we handle these as adjectives
                    if token.dep_ == "poss":
                        continue

                    # check for compound nouns
                    noun_tokens = [token]
                    for child in token.children:
                        if (
                            child.dep_.startswith("compound")
                            and child.pos_ in ("NOUN", "PROPN")
                            and child.i not in adjectival_noun_tokens
                            and child.i not in processed_tokens
                        ):
                            noun_tokens.append(child)

                    # sort by position and get text
                    noun_tokens.sort(key=lambda t: t.i)

                    # get token spans
                    token_indices = [t.i for t in noun_tokens]

                    # get character spans for each token
                    char_spans = [(t.idx, t.idx + len(t.text)) for t in noun_tokens]

                    # create the text by joining tokens
                    clause_text = " ".join([t.text for t in noun_tokens])

                    component_idx = len(components)
                    components.append(
                        {
                            "sentence_id": sent_idx,
                            "token_indices": token_indices,
                            "char_spans": char_spans,
                            "clause_text": clause_text,
                            "type": "noun",
                            "pos_tags": token.pos_,
                            "related_components": [],
                        }
                    )

                    # store mapping from token to component index
                    for t in noun_tokens:
                        token_to_component_idx[t.i] = component_idx
                        processed_tokens.add(t.i)

            # 2. extract adjectives (including noun compounds functioning as adjectives)
            for token in sent:
                if (
                    token.pos_ in ("ADJ", "NUM")
                    or token.i in adjectival_noun_tokens
                    or token.dep_ == "poss"
                    or (token.dep_ == "conj" and token.head and token.head.pos_ == "ADJ")
                    or (token.pos_ == "VERB" and token.dep_ == "amod")
                    or (
                        token.pos_ == "ADV" and token.dep_ == "advmod" and token.head.pos_ == "NOUN"
                    )
                ) and token.i not in processed_tokens:
                    # check for compound adjectives or noun compounds functioning as adjectives
                    adj_tokens = [token]
                    for child in token.children:
                        if (
                            (
                                child.dep_.startswith("compound")
                                and child.pos_ in ("ADJ", "ADV", "NOUN", "PROPN")
                            )
                            or (child.i in adjectival_noun_tokens)
                        ) and child.i not in processed_tokens:
                            adj_tokens.append(child)

                    # sort by position and get text
                    adj_tokens.sort(key=lambda t: t.i)

                    # get token spans
                    token_indices = [t.i for t in adj_tokens]

                    # get character spans for each token
                    char_spans = [(t.idx, t.idx + len(t.text)) for t in adj_tokens]

                    # create the text by joining tokens
                    clause_text = " ".join([t.text for t in adj_tokens])

                    component_idx = len(components)
                    components.append(
                        {
                            "sentence_id": sent_idx,
                            "token_indices": token_indices,
                            "char_spans": char_spans,
                            "clause_text": clause_text,
                            "clause_lemmas": [t.lemma_ for t in adj_tokens],
                            "type": "adj",
                            "pos_tags": "ADJ" if token.i in adjectival_noun_tokens else token.pos_,
                            "is_poss": token.dep_ == "poss",
                            "related_components": [],
                        }
                    )

                    # store mapping for all tokens in this component
                    for t in adj_tokens:
                        token_to_component_idx[t.i] = component_idx
                        processed_tokens.add(t.i)

            # 3. extract verbal relations
            for token in sent:
                if (
                    token.pos_ == "VERB" and token.dep_ != "amod"
                ) and token.i not in processed_tokens:
                    # identify the full verb phrase
                    verb_phrase_tokens = [token]

                    # add particles (for phrasal verbs like "sets out")
                    for child in token.children:
                        if (
                            child.i > token.i
                            and (
                                child.dep_ in ("prt", "compound:prt")
                                or (child.dep_ == "advmod" and child.head == token)
                            )
                        ) and child.i not in processed_tokens:
                            verb_phrase_tokens.append(child)

                    # sort tokens by position
                    verb_phrase_tokens.sort(key=lambda t: t.i)

                    # get token spans
                    token_indices = [t.i for t in verb_phrase_tokens]

                    # get character spans for each token
                    char_spans = [(t.idx, t.idx + len(t.text)) for t in verb_phrase_tokens]

                    # create the text by joining tokens
                    clause_text = " ".join([t.text for t in verb_phrase_tokens])

                    component_idx = len(components)
                    components.append(
                        {
                            "sentence_id": sent_idx,
                            "token_indices": token_indices,
                            "char_spans": char_spans,
                            "clause_text": clause_text,
                            "clause_lemmas": [t.lemma_ for t in verb_phrase_tokens],
                            "type": "verb",
                            "pos_tags": "VERB",
                            "related_components": [],
                        }
                    )

                    # store mapping for all tokens in this component
                    for t in verb_phrase_tokens:
                        token_to_component_idx[t.i] = component_idx
                        processed_tokens.add(t.i)

            # 4. extract prepositional phrases
            for token in sent:
                if (
                    token.pos_ == "ADP"
                    and token.dep_ in ("prep", "case", "conj", "agent")
                    and token.i not in processed_tokens
                ):
                    # get token spans
                    token_indices = [token.i]

                    # get character spans for each token
                    char_spans = [(token.idx, token.idx + len(token.text))]

                    # create the text from the token
                    clause_text = token.text

                    component_idx = len(components)
                    components.append(
                        {
                            "sentence_id": sent_idx,
                            "token_indices": token_indices,
                            "char_spans": char_spans,
                            "clause_text": clause_text,
                            "clause_lemmas": [token.lemma_],
                            "type": "prep",
                            "pos_tags": token.pos_,
                            "related_components": [],
                        }
                    )

                    # store mapping
                    token_to_component_idx[token.i] = component_idx
                    processed_tokens.add(token.i)

            # 5. extract adverbs
            for token in sent:
                if token.pos_ == "ADV" and token.i not in processed_tokens:
                    # we handle this with the adjectives
                    if token.head and token.dep_ == "advmod" and token.head.pos_ == "NOUN":
                        continue

                    # first check if this adverb is part of a multi-word adverbial where it's the head
                    head_of_advmod = False
                    adverb_children = []

                    # find adverb children of this token
                    for child in token.children:
                        if (
                            child.pos_ == "ADV"
                            and child.dep_ == "advmod"
                            and child.i not in processed_tokens
                        ):
                            adverb_children.append(child)
                            head_of_advmod = True

                    # if this is the head of an adverbial phrase, process the whole phrase
                    if head_of_advmod:
                        adv_phrase_tokens = [token] + adverb_children

                        # add any other compounds
                        for child in token.children:
                            if (
                                child.dep_.startswith("compound") or child.dep_ == "amod"
                            ) and child.i not in processed_tokens:
                                adv_phrase_tokens.append(child)

                        # sort tokens by position
                        adv_phrase_tokens.sort(key=lambda t: t.i)

                        # get token spans
                        token_indices = [t.i for t in adv_phrase_tokens]

                        # get character spans for each token
                        char_spans = [(t.idx, t.idx + len(t.text)) for t in adv_phrase_tokens]

                        # create the text by joining tokens
                        clause_text = " ".join([t.text for t in adv_phrase_tokens])

                        component_idx = len(components)
                        components.append(
                            {
                                "sentence_id": sent_idx,
                                "token_indices": token_indices,
                                "char_spans": char_spans,
                                "clause_text": clause_text,
                                "clause_lemmas": [t.lemma_ for t in adv_phrase_tokens],
                                "type": "adverb",
                                "pos_tags": token.pos_,
                                "related_components": [],
                            }
                        )

                        # store mapping for all tokens
                        for t in adv_phrase_tokens:
                            token_to_component_idx[t.i] = component_idx
                            processed_tokens.add(t.i)

                    # also check if this adverb modifies another adverb
                    # if so, skip it as it will be processed as part of that adverb's phrase
                    elif (
                        token.head.pos_ == "ADV"
                        and token.dep_ == "advmod"
                        and token.head.i not in processed_tokens
                    ):
                        # skip - this will be processed when we get to the head adverb
                        continue

                    # otherwise process as a single adverb or with non-adverb compounds
                    else:
                        # check if we're dealing with a multi-word adverbial
                        adv_phrase_tokens = [token]

                        # add any compound adverbs or other modifiers (like "far off", "very quickly")
                        for child in token.children:
                            if (
                                (child.dep_.startswith("compound") or child.dep_ == "amod")
                                and child.pos_ in ("ADV", "ADP", "PART", "ADJ")
                                and child.i not in processed_tokens
                            ):
                                adv_phrase_tokens.append(child)

                        # sort tokens by position
                        adv_phrase_tokens.sort(key=lambda t: t.i)

                        # get token spans
                        token_indices = [t.i for t in adv_phrase_tokens]

                        # get character spans for each token
                        char_spans = [(t.idx, t.idx + len(t.text)) for t in adv_phrase_tokens]

                        # create the text by joining tokens
                        clause_text = " ".join([t.text for t in adv_phrase_tokens])

                        component_idx = len(components)
                        components.append(
                            {
                                "sentence_id": sent_idx,
                                "token_indices": token_indices,
                                "char_spans": char_spans,
                                "clause_text": clause_text,
                                "clause_lemmas": [t.lemma_ for t in adv_phrase_tokens],
                                "type": "adverb",
                                "pos_tags": token.pos_,
                                "related_components": [],
                            }
                        )

                        # store mapping for all tokens
                        for t in adv_phrase_tokens:
                            token_to_component_idx[t.i] = component_idx
                            processed_tokens.add(t.i)

            # second pass: add relationship information
            for i, component in enumerate(components):
                if component["sentence_id"] != sent_idx:
                    continue  # skip components from other sentences

                if component["type"] == "adj":
                    # for adjectives, find the nouns they modify
                    seen = set()
                    for try_children in [False, True]:
                        for token_idx in component["token_indices"]:
                            token = doc[token_idx]

                            if not try_children:
                                candidates = [token.head]
                            else:
                                if len(component["related_components"]) > 0:
                                    continue
                                candidates = list(token.children)

                            while len(candidates) > 0:
                                candidate = candidates.pop()
                                seen.add(candidate.i)
                                # check both for regular adjectives and noun modifiers
                                if (
                                    candidate.pos_ in ("NOUN", "PROPN")
                                    and candidate.dep_ != "amod"
                                    and candidate.i in token_to_component_idx
                                ):
                                    noun_component_idx = token_to_component_idx[candidate.i]
                                    # add this relationship
                                    component["related_components"].append(
                                        {"index": noun_component_idx, "role": "modified_noun"}
                                    )
                                    # add back-reference
                                    components[noun_component_idx]["related_components"].append(
                                        {"index": i, "role": "modifier"}
                                    )
                                elif not try_children and candidate.dep_ == "amod":
                                    if candidate.head.i not in seen:
                                        candidates.append(candidate.head)
                                elif not try_children and candidate.pos_ == "AUX":
                                    for candidate_child in candidate.children:
                                        if (
                                            candidate_child.i not in seen
                                            and candidate_child.i not in component["token_indices"]
                                        ):
                                            candidates.append(candidate_child)
                elif component["type"] == "verb":
                    # for verbs, find subjects and objects
                    for token_idx in component["token_indices"]:
                        if doc[token_idx].pos_ != "VERB":
                            continue  # skip particles

                        token = doc[token_idx]

                        # find subjects
                        seen, subject_candidates = (
                            set(),
                            [
                                (True, token.head),
                                *[(False, child) for child in token.children],
                            ],
                        )
                        while len(subject_candidates) > 0:
                            is_head, relation = subject_candidates.pop()
                            seen.add(relation.i)
                            if (
                                (is_head and relation.pos_ in {"NOUN", "PROPN", "PRON"})
                                or relation.dep_ in ("nsubj", "nsubjpass")
                            ) and relation.i in token_to_component_idx:
                                subj_component_idx = token_to_component_idx[relation.i]
                                # add this relationship
                                component["related_components"].append(
                                    {"index": subj_component_idx, "role": "subject"}
                                )
                                # add back-reference
                                components[subj_component_idx]["related_components"].append(
                                    {"index": i, "role": "verb_subject"}
                                )
                            elif is_head and relation.pos_ == "AUX":
                                subject_candidates.extend(
                                    [
                                        (False, child)
                                        for child in relation.children
                                        if child.i not in seen
                                    ]
                                )

                        # find objects
                        for child in token.children:
                            if child.dep_ in ("dobj", "obj") and child.i in token_to_component_idx:
                                obj_component_idx = token_to_component_idx[child.i]
                                # add this relationship
                                component["related_components"].append(
                                    {"index": obj_component_idx, "role": "direct_object"}
                                )
                                # add back-reference
                                components[obj_component_idx]["related_components"].append(
                                    {"index": i, "role": "verb_direct_object"}
                                )

                elif component["type"] == "prep":
                    # for prepositions, find head and object
                    token = doc[component["token_indices"][0]]

                    # find the head
                    if token.head.i in token_to_component_idx:
                        head_component_idx = token_to_component_idx[token.head.i]
                        # add this relationship
                        component["related_components"].append(
                            {"index": head_component_idx, "role": "head"}
                        )
                        # add back-reference
                        components[head_component_idx]["related_components"].append(
                            {"index": i, "role": "prep_head"}
                        )

                    # find objects of the preposition
                    for child in token.children:
                        if child.dep_ in ("pobj", "obj") and child.i in token_to_component_idx:
                            obj_component_idx = token_to_component_idx[child.i]
                            # add this relationship
                            component["related_components"].append(
                                {"index": obj_component_idx, "role": "object"}
                            )
                            # add back-reference
                            components[obj_component_idx]["related_components"].append(
                                {"index": i, "role": "prep_object"}
                            )

                elif component["type"] == "adverb":
                    # for adverbs, find modified verbs, adjectives, or adverbs
                    for token_idx in component["token_indices"]:
                        token = doc[token_idx]
                        if (
                            token.head.pos_ in {"VERB", "ADJ", "ADV"}
                            and token.head.i not in component["token_indices"]
                            and token.head.i in token_to_component_idx
                        ):
                            verb_component_idx = token_to_component_idx[token.head.i]
                            # add this relationship
                            component["related_components"].append(
                                {
                                    "index": verb_component_idx,
                                    "role": "modified_%s"
                                    % {
                                        "VERB": "verb",
                                        "ADJ": "adjective",
                                        "ADV": "adverb",
                                    }[token.head.pos_],
                                }
                            )
                            # add back-reference
                            components[verb_component_idx]["related_components"].append(
                                {"index": i, "role": "adverbial_modifier"}
                            )

        # final pass: handle conjunctions
        # 1. create mapping from token indices to component indices for different types
        component_by_token = {}
        for i, component in enumerate(components):
            for token_idx in component["token_indices"]:
                component_by_token[token_idx] = (i, component["type"])

        # 2. find all conjunction relationships and handle them by type
        conj_pairs = []
        for token in doc:
            if (
                token.dep_ == "conj"
                and token.i in component_by_token
                and token.head.i in component_by_token
                and abs(token.i - token.head.i) <= 8
            ):
                head_info = component_by_token[token.head.i]
                conj_info = component_by_token[token.i]

                # only process if both are of the same type
                if head_info[1] == conj_info[1]:
                    conj_pairs.append((head_info[0], conj_info[0], head_info[1]))

        # 3. propagate relationships between conjoined elements
        for head_idx, conj_idx, comp_type in conj_pairs:
            # copy relationships from the head to the conjoined component
            head_related = components[head_idx]["related_components"]
            conj_related = components[conj_idx]["related_components"]

            # track existing relationships to avoid duplicates
            existing_conj_relations = {(rel["index"], rel["role"]) for rel in conj_related}
            existing_conj_relation_types = {rel["role"] for rel in conj_related}

            # different handling based on component type
            if comp_type in "prep":
                # for prepositions, we want to replace the head completely with the
                # head of the main preposition, and not copy object relationships

                # 1. first, find the head relationship from the head preposition
                head_relation = None
                for rel in head_related:
                    if rel["role"] == "head":
                        head_relation = rel.copy()
                        break

                object_relation = None
                for rel in head_related:
                    if rel["role"] == "object":
                        object_relation = rel.copy()
                        break

                # 2. if we found a head relation from the head preposition
                if head_relation:
                    # remove any existing head relations from the conjunct
                    components[conj_idx]["related_components"] = [
                        rel for rel in conj_related if rel["role"] != "head"
                    ]

                    # add the head relation from the head preposition
                    components[conj_idx]["related_components"].append(head_relation)

                    # update conj_related to match the updated component
                    conj_related = components[conj_idx]["related_components"]

                    # add back-reference from the related component
                    target_comp = components[head_relation["index"]]

                    # make sure the back-reference doesn't already exist
                    existing_back_refs = [
                        (rel["index"], rel["role"]) for rel in target_comp["related_components"]
                    ]
                    if (conj_idx, "prep_head") not in existing_back_refs:
                        target_comp["related_components"].append(
                            {"index": conj_idx, "role": "prep_head"}
                        )

                if not object_relation:
                    conj_object_relation = None
                    for rel in conj_related:
                        if rel["role"] == "object":
                            conj_object_relation = rel.copy()
                            break

                    if conj_object_relation:
                        components[head_idx]["related_components"].append(conj_object_relation)
            else:
                # for other types (nouns, adjectives, verbs, adverbs)
                # add head's relationships to conjoined component
                for rel in head_related:
                    # skip if would create self-reference
                    if rel["index"] == conj_idx:
                        continue

                    rel_key = (rel["index"], rel["role"])
                    if rel_key not in existing_conj_relations and (
                        rel["role"]
                        not in {
                            "modifier",
                            "modified_noun",
                            "verb_subject",
                            "subject",
                            "direct_object",
                            "prep_head",
                        }
                        or rel["role"] not in existing_conj_relation_types
                    ):
                        # add this relationship to the conjoined component
                        conj_related.append(rel.copy())

                        # add back-reference from the related component
                        target_comp = components[rel["index"]]
                        # determine the appropriate role for back-reference
                        back_role = None
                        for back_rel in target_comp["related_components"]:
                            if back_rel["index"] == head_idx:
                                back_role = back_rel["role"]
                                break

                        if back_role:
                            target_comp["related_components"].append(
                                {"index": conj_idx, "role": back_role}
                            )

        # sort components by the position of their first token
        sorted_components = sorted(
            components,
            key=lambda x: min(x["token_indices"]) if x["token_indices"] else float("inf"),
        )

        # create mapping from old indices to new indices
        old_to_new_idx = {
            i: new_i for new_i, i in enumerate([components.index(c) for c in sorted_components])
        }
        del components

        # update all relationship indices
        for component in sorted_components:
            for relation in component["related_components"]:
                relation["index"] = old_to_new_idx[relation["index"]]

        # and finally, augment with coreferent component indices

        token_indices_to_components = defaultdict(list)
        for component_num, component in enumerate(sorted_components):
            for token_idx in component["token_indices"]:
                token_indices_to_components[token_idx].append(component_num)

        for component_num, component in enumerate(sorted_components):
            coreferent_components, coreferent_clusters = set(), set()
            for token_idx in component["token_indices"]:
                if token_idx not in token_cluster_mapping:
                    continue

                cluster_num = token_cluster_mapping[token_idx]
                for token_idx in cluster_token_mapping[cluster_num]:
                    for coreferent_component_num in token_indices_to_components[token_idx]:
                        if component_num != coreferent_component_num:
                            coreferent_clusters.add(cluster_num)
                            coreferent_components.add(coreferent_component_num)
            component["coreferent_components"] = list(sorted(coreferent_components))
            component["coreferent_clusters"] = list(sorted(coreferent_clusters))

        ## PREPARE SCENE GRAPH

        entities, component_to_entity_mapping = [], {}
        for component_id, component in enumerate(sorted_components):
            if component["type"] != "noun":
                continue

            if component["pos_tags"] == "PRON" and len(component["related_components"]) == 0:
                continue

            is_us = component["clause_text"].lower() in {"us", "our", "we"}
            is_left = left_cluster in component["coreferent_clusters"]
            is_right = right_cluster in component["coreferent_clusters"]

            earlier_mentions = [
                component_to_entity_mapping[coreferent_component_id]
                for coreferent_component_id in sorted(component.get("coreferent_components", []))
                if coreferent_component_id in component_to_entity_mapping
            ]

            if len(earlier_mentions) == 0:
                component_to_entity_mapping[component_id] = len(entities)
                entities.append(
                    {
                        "head": component["clause_text"],
                        "heads": [component["clause_text"]],
                        "quantity": None,
                        "attributes": [],
                        "sentence_idxs": [component["sentence_id"]],
                        "text_spans": [*component["char_spans"]],
                        "attribute_sentence_idxs": {},
                        "attribute_text_spans": {},
                        "is_us": is_us,
                        "is_left": is_left,
                        "is_right": is_right,
                    }
                )
            else:
                first_mention = min(earlier_mentions)
                component_to_entity_mapping[component_id] = first_mention
                entities[first_mention]["heads"].append(component["clause_text"])
                entities[first_mention]["text_spans"].extend(component["char_spans"])
                entities[first_mention]["sentence_idxs"].append(component["sentence_id"])

        relations, handled_components, existing_relations = [], set(), {}
        for component_id, component in enumerate(sorted_components):
            if component_id in handled_components:
                continue

            if component_id in component_to_entity_mapping:
                assert component["type"] == "noun"
                continue

            if component["type"] == "adj":
                for related_component in component["related_components"]:
                    if (
                        related_component["role"] != "modified_noun"
                        or related_component["index"] not in component_to_entity_mapping
                    ):
                        continue

                    modified_entity_id = component_to_entity_mapping[related_component["index"]]
                    if not component["is_poss"]:
                        handled_components.add(component_id)
                        attribute = component["clause_text"]
                        if attribute not in entities[modified_entity_id]["attributes"]:
                            entities[modified_entity_id]["attributes"].append(attribute)
                            entities[modified_entity_id]["attribute_sentence_idxs"][attribute] = []
                            entities[modified_entity_id]["attribute_text_spans"][attribute] = []
                        entities[modified_entity_id]["attribute_sentence_idxs"][attribute].append(
                            component["sentence_id"]
                        )
                        entities[modified_entity_id]["attribute_text_spans"][attribute].extend(
                            component["char_spans"]
                        )
                    elif len(component["coreferent_components"]) > 0:
                        earlier_mentions = [
                            component_to_entity_mapping[coreferent_component_id]
                            for coreferent_component_id in sorted(
                                component["coreferent_components"]
                            )
                            if coreferent_component_id in component_to_entity_mapping
                        ]

                        if len(earlier_mentions) >= 1:
                            handled_components.add(component_id)
                            possessing_entity_id = earlier_mentions[0]

                            rel_key = (modified_entity_id, possessing_entity_id, "part of")
                            if rel_key not in existing_relations:
                                existing_relations[rel_key] = len(relations)
                                relations.append(
                                    {
                                        "subject": modified_entity_id,
                                        "object": possessing_entity_id,
                                        "relation": "part of",
                                        "source": "poss",
                                        "sentence_idxs": [],
                                        "text_spans": [],
                                    }
                                )
                            relations[existing_relations[rel_key]]["sentence_idxs"].append(
                                component["sentence_id"]
                            )
                            relations[existing_relations[rel_key]]["text_spans"].extend(
                                component["char_spans"]
                            )
            elif component["type"] == "verb":
                verb_pieces = [(component_id, component["clause_text"])]

                roles = defaultdict(list)
                seen_do, related_component_stack = False, [(True, component_id)]
                while len(related_component_stack) > 0:
                    is_parent, current_component_id = related_component_stack.pop()
                    for related_component in sorted(
                        sorted_components[current_component_id]["related_components"],
                        key=itemgetter("index"),
                    ):
                        if is_parent and related_component["role"] == "subject":
                            roles["subject"].append(related_component["index"])
                        elif related_component["role"] == "prep_head":
                            if related_component["index"] < current_component_id:
                                continue

                            verb_pieces.append(
                                (
                                    related_component["index"],
                                    sorted_components[related_component["index"]]["clause_text"],
                                )
                            )
                            related_component_stack.append((False, related_component["index"]))
                        elif related_component["role"] in {"object", "direct_object"}:
                            if related_component["role"] == "direct_object" or not seen_do:
                                seen_do = seen_do or related_component["role"] == "direct_object"
                                roles["object"].append(related_component["index"])
                            else:
                                roles["prep_object"].append(related_component["index"])

                if len(roles["subject"]) == 0:
                    continue
                else:
                    if len(roles["object"]) == 0:
                        for subject_id in roles["subject"]:
                            if subject_id not in component_to_entity_mapping:
                                continue

                            handled_components.update(
                                {verb_component_id for verb_component_id, _ in verb_pieces}
                            )
                            verb_text = " ".join(
                                [clause_text for _, clause_text in sorted(verb_pieces)]
                            )
                            verb_char_spans = [
                                char_span
                                for verb_component_id, _ in verb_pieces
                                for char_span in sorted_components[verb_component_id]["char_spans"]
                            ]

                            handled_components.add(component_id)
                            entity_id = component_to_entity_mapping[subject_id]
                            if verb_text not in entities[entity_id]["attributes"]:
                                entities[entity_id]["attributes"].append(verb_text)
                                entities[entity_id]["attribute_sentence_idxs"][verb_text] = []
                                entities[entity_id]["attribute_text_spans"][verb_text] = []
                            entities[entity_id]["attribute_sentence_idxs"][verb_text].append(
                                component["sentence_id"]
                            )
                            entities[entity_id]["attribute_text_spans"][verb_text].extend(
                                verb_char_spans
                            )
                    else:
                        for verb_subj, subject_id in sorted(
                            [(True, subject_id) for subject_id in roles["subject"]]
                            + [(False, subject_id) for subject_id in roles["object"]]
                        ):
                            if subject_id not in component_to_entity_mapping:
                                continue

                            head_entity_id = component_to_entity_mapping[subject_id]
                            for object_id in sorted(
                                roles["object"] if verb_subj else roles["prep_object"]
                            ):
                                if object_id not in component_to_entity_mapping:
                                    continue

                                verb_text = " ".join(
                                    [
                                        clause_text
                                        for verb_component_id, clause_text in sorted(
                                            set(verb_pieces)
                                        )
                                        if (subject_id <= verb_component_id <= object_id)
                                        and (
                                            sorted_components[verb_component_id]["type"] == "verb"
                                            or (
                                                verb_subj
                                                or verb_component_id not in handled_components
                                            )
                                        )
                                    ]
                                )
                                verb_char_spans = [
                                    char_span
                                    for verb_component_id, _ in sorted(set(verb_pieces))
                                    for char_span in sorted_components[verb_component_id][
                                        "char_spans"
                                    ]
                                    if (subject_id <= verb_component_id <= object_id)
                                    and (
                                        sorted_components[verb_component_id]["type"] == "verb"
                                        or (
                                            verb_subj or verb_component_id not in handled_components
                                        )
                                    )
                                ]
                                handled_components.update(
                                    {
                                        verb_component_id
                                        for verb_component_id, _ in verb_pieces
                                        if (subject_id <= verb_component_id <= object_id)
                                    }
                                )

                                if len(verb_char_spans) == 0:
                                    continue

                                object_entity_id = component_to_entity_mapping[object_id]
                                rel_key = (head_entity_id, object_entity_id, verb_text)
                                if rel_key not in existing_relations:
                                    existing_relations[rel_key] = len(relations)
                                    relations.append(
                                        {
                                            "subject": head_entity_id,
                                            "object": object_entity_id,
                                            "relation": verb_text,
                                            "source": "verb",
                                            "sentence_idxs": [],
                                            "text_spans": [],
                                        }
                                    )

                                handled_components.add(component_id)
                                relations[existing_relations[rel_key]]["sentence_idxs"].append(
                                    component["sentence_id"]
                                )
                                relations[existing_relations[rel_key]]["text_spans"].extend(
                                    verb_char_spans
                                )

        for component_id, component in enumerate(sorted_components):
            if component["type"] != "prep" or component_id in handled_components:
                continue

            head_component_ids = [
                related_component["index"]
                for related_component in component["related_components"]
                if related_component["role"] == "head"
            ]
            object_component_ids = [
                related_component["index"]
                for related_component in component["related_components"]
                if related_component["role"] == "object"
            ]

            for head_component_id in head_component_ids:
                if head_component_id not in component_to_entity_mapping:
                    continue

                head_entity_id = component_to_entity_mapping[head_component_id]

                for object_component_id in object_component_ids:
                    if object_component_id not in component_to_entity_mapping:
                        continue

                    object_entity_id = component_to_entity_mapping[object_component_id]

                    rel_key = (head_entity_id, object_entity_id, component["clause_text"])
                    if rel_key not in existing_relations:
                        existing_relations[rel_key] = len(relations)
                        relations.append(
                            {
                                "subject": head_entity_id,
                                "object": object_entity_id,
                                "relation": component["clause_text"],
                                "source": "prep",
                                "sentence_idxs": [],
                                "text_spans": [],
                            }
                        )

                    handled_components.add(component_id)
                    relations[existing_relations[rel_key]]["sentence_idxs"].append(
                        component["sentence_id"]
                    )
                    relations[existing_relations[rel_key]]["text_spans"].extend(
                        component["char_spans"]
                    )

        return {"entities": entities, "relations": relations, "sentences": sentences}

    def _get_graphs(self, texts: List[str]) -> List[TextGraph]:
        texts = [text.strip() for text in texts]

        docs = self.get_parsed_docs(texts)

        coreferences = self.get_coreferences(docs)

        graphs = [self.get_graph(doc, coreference) for doc, coreference in zip(docs, coreferences)]

        return graphs

    def get_graphs(self, texts: List[str], refresh_cache: bool = False) -> List[TextGraph]:
        to_process = {}
        if self.cache_dir is not None:
            cache_path = self.get_cache_path()
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            else:
                self.cache = {}

            if not refresh_cache:
                for text_idx, text in enumerate(texts):
                    if text not in self.cache:
                        to_process[text_idx] = len(to_process)
            else:
                to_process = {text_idx: text_idx for text_idx in range(len(texts))}
        else:
            cache_path = None
            to_process = {text_idx: text_idx for text_idx in range(len(texts))}

        if len(to_process) > 0:
            processed_sgs = self._get_graphs([texts[text_idx] for text_idx in sorted(to_process)])
        else:
            processed_sgs = []

        sgs, cache_updated = [], False
        for text_idx, text in enumerate(texts):
            if text_idx in to_process:
                if isinstance(processed_sgs[to_process[text_idx]], Exception):
                    sgs.append(None)
                    print(
                        f"error processing text {text_idx}: {processed_sgs[to_process[text_idx]]}"
                    )
                    print(to_process[text_idx])
                    continue

                sg = processed_sgs[to_process[text_idx]]
                if self.cache_dir is not None:
                    self.cache[text] = sg
                    self.cache[text.strip()] = sg
                    cache_updated = True
                sgs.append(sg)
            else:
                sgs.append(self.cache[text])

        if cache_path is not None and cache_updated:
            with open(cache_path, "wb") as f:
                pickle.dump(self.cache, f)

        return sgs
