# Topic 11: the VGGT-Ω rolling-shutter claim (VERIFIED)

**Source.** The local copy `VGGT-Ω.pdf` in the repo root.
- PDF metadata gives `/arXivID: https://arxiv.org/abs/2605.15195v1` and DOI 10.48550/arXiv.2605.15195.
- Authors (from the PDF metadata and title page): Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schönberger, Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, Christian Rupprecht. Affiliations: VGG Oxford and Meta AI.
- The venue is not stated in the PDF. Treat it as arXiv 2026 unless another page says otherwise.

## Quote 1: rolling-shutter rejection (Appendix A.3 "VLM Prompt", pp. 24–25 of the PDF)

> **A.3. VLM Prompt.** We use a VLM to discard videos not suitable for multi-view geometry. We prompt the VLM as follows: …
> STEP 1: CHECK FOR HARD REJECTION FACTORS ‘‘Assess the following. If ANY are present, the Classification is ‘REJECT HARD’’’: …
> 3. Extreme Visual Failure: Severe motion blur (edges lost), severe rolling shutter (wobble/jello effect), or corruption/glitching.

- The STEP 1 header is on PDF page 24, whose printed page number is 24.
- Item 3 is on PDF page 25.
- The same prompt also hard-rejects uncalibrated fisheye and 360° footage (item 5). It soft-rejects rotation-only clips: "Does it only rotate (pan/tilt) … without physical translation?"

**What this supports.** VGGT-Ω's video training data was deliberately filtered to remove *severe* rolling shutter. So the model was not trained to handle strong RS. Mild RS is not filtered, so some RS is likely present in its training data.

**What it does not support.** It does not show that VGGT-Ω fails on RS. That has to be measured. It is a VLM filter, applied by eye-level judgement, with no readout-time threshold.

## Quote 2: confidence loss when fine-tuning (PDF page 13), used for the EXP2 §8 training rules

> Second, we found that the aleatoric uncertainty (confidence) loss can be unstable during fine-tuning. We therefore recommend initially removing this loss, particularly when the fine-tuning dataset is very small. If the downstream task requires confidence estimates, the uncertainty head can be fine-tuned separately.

The same paragraph also says:

> When fine-tuning VGGT/VGGT-Ω for tasks besides reconstruction, we suggest increasing the warm-up ratio, e.g. from 5% to 10–15%, and use a full scheduling cycle.

**Status.** Both quotes are VERIFIED. They were extracted with pypdf on 2026-10-29 from the arXiv-generated PDF (Creator: arXiv GenPDF).
