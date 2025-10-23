import os

from .graphs.text_graphs import SceneGraphExtractor
from .scorers.qa_scene_graph_scorer import QASceneGraphScorer


class PoSh:
    def __init__(
        self,
        sg_spacy_model="en_core_web_trf",
        sg_coref_model="sapienzanlp/maverick-mes-ontonotes",
        sg_parse_batch_size=128,
        qa_model="Qwen/Qwen3-14B",
        qa_presence_threshold=2,
        qa_max_tokens=2000,
        qa_gpu_memory_utilization=0.9,
        qa_tensor_parallel_size=1,
        qa_enable_prefix_caching=True,
        unofficial=False,
        cache_dir=None,
        verbosity="quiet",
        keep_alive=False,
    ):
        self.verbosity = verbosity
        self.ref_sg_cache = {}
        self.scene_graph_extractor = SceneGraphExtractor(
            spacy_model_name=sg_spacy_model,
            coref_model_name=sg_coref_model,
            parse_batch_size=sg_parse_batch_size,
            cache_dir=os.path.join(cache_dir, "scene_graphs") if cache_dir else None,
            verbosity=verbosity,
            keep_alive=keep_alive,
        )
        self.qa_scene_graph_scorer = QASceneGraphScorer(
            model_type=qa_model,
            presence_threshold=qa_presence_threshold,
            max_tokens=qa_max_tokens,
            gpu_memory_utilization=qa_gpu_memory_utilization,
            tensor_parallel_size=qa_tensor_parallel_size,
            enable_prefix_caching=qa_enable_prefix_caching,
            unofficial=unofficial,
            cache_dir=os.path.join(cache_dir, "granular_scores") if cache_dir else None,
            verbosity=verbosity,
            keep_alive=keep_alive,
        )

    def evaluate(self, generations, references, cache_keys=None, overwrite_cache=False):
        assert isinstance(generations, list) and isinstance(references, list), (
            f"Generations and references must be lists: {type(generations)} != {type(references)}"
        )
        assert len(generations) == len(references), (
            f"Lengths of generations and references must match: {len(generations)} != {len(references)}"
        )

        if cache_keys is not None:
            assert self.qa_scene_graph_scorer.cache_dir is not None, (
                "Cache keys provided but cache directory is not set"
            )
            assert len(cache_keys) == len(generations), (
                f"Lengths of cache keys and generations must match: {len(cache_keys)} != {len(generations)}"
            )

        if self.verbosity == "debug":
            print("Extracting scene graphs...")

        texts_to_extract = {generation for generation in generations}
        for reference in references:
            if overwrite_cache or reference not in self.ref_sg_cache:
                texts_to_extract.add(reference)

        texts_to_extract = list(sorted(texts_to_extract))
        if len(texts_to_extract) > 0:
            extracted_sgs = {
                text: sg
                for text, sg in zip(
                    texts_to_extract, self.scene_graph_extractor.get_graphs(texts_to_extract)
                )
            }
        else:
            extracted_sgs = {}

        generation_sgs, reference_sgs = [], []
        for source, source_texts in [("generation", generations), ("reference", references)]:
            if source == "generation":
                sgs = generation_sgs
            else:
                sgs = reference_sgs

            for source_text in source_texts:
                if source == "generation" or source_text not in self.ref_sg_cache:
                    sg = extracted_sgs[source_text]
                    if source == "generation":
                        self.ref_sg_cache[source_text] = sg
                else:
                    sg = self.ref_sg_cache[source_text]

                sgs.append(sg)

        if self.verbosity == "debug":
            print("Scoring...")

        coarse_scores = self.qa_scene_graph_scorer.batch_calculate_coarse_scores(
            generation_sgs,
            reference_sgs,
            generations,
            references,
            cache_keys=cache_keys,
            overwrite_cache=overwrite_cache,
        )

        assert len(coarse_scores) == len(generations) == len(references), (
            f"Lengths of coarse scores, generations, and references must match: {len(coarse_scores)} != {len(generations)} != {len(references)}"
        )

        if self.verbosity == "debug":
            for coarse_score in coarse_scores:
                print(f"Coarse Score: {coarse_score}")

        return coarse_scores
