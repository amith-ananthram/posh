# Are your detailed image descriptions what you (really really) want?  Let PoSh be the judge.

![A visualization of PoSh.  On the lefthand side is a painting of a knight in armor striding toward two women, one wearing a white dress and holding a gold chalice, the other wearing a blue dress.  To the right of the image are two horizontal frames depicting how PoSh calculates scores.  The top frame focuses on scoring mistakes in a generation.  The bottom frame focuses on scoring omissions in a reference.  The first column shows parts of a generation on top and relevant parts of its reference on the bottom -- in each text, the shared entities are highlighted: knight, a woman in white, a woman in blue and a gold chalice.  In the second column, we see the first step of PoSh, scene graph extraction.  We extract two scene graphs, one for the generation and the other for the reference.  The graphs are visualized with matching entities colored similarly.  In the third column, we see the second step of PoSh, granularly scoring scene graph elements by comparing them against the other text (i.e. comparing generation scene graph elements to the reference text and reference scene graph elements to the generation text).  Depicted are questions that are passed to the QA model.  For example, for the generation, the QA model is asked if reference describes the relation between the woman in a blue dress and the chalice as "hold".  As the reference correctly specifies that the woman in white is holding the chalice, this mistake receives a low score.  In the fourth column, we see the final step of PoSh, aggregating granular scores into a single coarse score for mistakes and omissions by taking their mean.](figures/posh.png "PoSh")

PoSh is an interpretable, replicable metric for detailed description evaluation that produces both granular and coarse scores for the mistakes, omissions and overall quality of a generation in comparison to a reference.  It does so in three steps:

1) Given a generated description and its reference, PoSh extracts scene graphs that reduce each text's surface diversity to its objects, attributes and relations.
2) Using each scene graph as a *structured rubric*, PoSh produces granular scores for the presence of its components in the other text through QA.
3) PoSh aggregates these granular scores for each scene graph to produce interpretable, coarse scores for mistakes and omissions.

To validate PoSh, we collect a new benchmark named DOCENT of artwork from the U.S. National Gallery of Art with expert written references paired with both granular and coarse judgments of model generations from art history students.  DOCENT allows evaluating detailed image description metrics and detailed image descriptions themselves.  To learn more, please see [DOCENT](docent/README.md).

In our evaluations, PoSh is a better proxy for the human judgments in DOCENT than existing open-weight metrics (and GPT4o-as-a-Judge).  Moreover, PoSh is robust to image type and source model, performing well on CapArena.  Finally, we find that PoSh is an effective reward function, outperforming SFT on the 1,000 training images on DOCENT.

To replicate our evaluation of PoSh on DOCENT and CapArena, please run the following on a single H100 GPU:

```
conda env create -f environment.yml
conda activate posh

python evaluate_posh.py --benchmark docent
python evaluate_posh.py --benchmark caparena
```

# Usage

To use PoSh, simply instantiate a PoSh instance and call #evaluate.

```
from posh.posh import PoSh

posh = PoSh(
    qa_gpu_memory_utilization=args.gpu_memory_utilization,
    qa_tensor_parallel_size=args.tensor_parallel_size,
    qa_enable_prefix_caching=args.enable_prefix_caching,
    verbosity=args.verbosity,
)

generations, references = [# your generations here], [# your references here]

coarse_scores = posh.evaluate(generations=generations, references=references)
```

