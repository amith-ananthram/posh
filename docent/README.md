# DOCENT

![A visualization of DOCENT.  On the left hand side is an example of the artwork in DOCENT, a lithograph of a bird holding a fish in its talons.  On the top are examples of DOCENT’s granular annotations.  We see the work’s correct reference description and a model generated description.  In the reference description, textual spans corresponding to visual details omitted in the generated description are highlighted.  In the generated description, textual spans corresponding to incorrect details are highlighted.  On the bottom are examples of DOCENT’s coarse annotations.  We see two model generated descriptions.  Beneath these descriptions are overall rankings of these descriptions in terms of their mistakes, omissions and overall quality.](../figures/docent.png "DOCENT")

DOCENT is a benchmark for evaluating detailed image descriptions and detailed image description metrics.

It contains 1,750 paintings, sketches, and sculptures with detailed reference descriptions authored by experts from the U.S. National Gallery of Art. For 100 images, we generate descriptions from current VLMs and collect both granular judgments (textual spans containing mistakes and omissions) and coarse judgments (rankings of generation pairs) from art history students.

## Judgments

The judgments in DOCENT are available on HuggingFace:

granular: https://huggingface.co/datasets/amitha/docent-eval-granular
coarse: https://huggingface.co/datasets/amitha/docent-eval-coarse

## Leaderboard

The train, validation and test images that comprise the DOCENT leaderboard, which tracks model progress on detailed image description, are also on HuggingFace:

https://huggingface.co/datasets/amitha/docent