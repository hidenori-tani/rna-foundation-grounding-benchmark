# Grounding RNA foundation models in transcript dynamics

**Hidenori Tani**^1,\*

^1 Department of Molecular Biology, Yokohama University of Pharmaceutical Sciences, 601 Matano-cho, Totsuka-ku, Yokohama, Kanagawa 245-0066, Japan

\* Corresponding author. hidenori.tani@yok.hamayaku.ac.jp
ORCID: 0000-0001-6390-4136

---

## Abstract

Long non-coding RNAs (lncRNAs) outnumber annotated protein-coding genes yet remain
functionally opaque, and a wave of RNA foundation models promises to read function
directly from sequence. Using transcript half-life as a concrete functional axis,
we benchmark five lncRNA-relevant representation classes — two directly evaluated,
three approximated by tractable proxies — against measured RNA turnover across
four mammalian cell systems, covering 256 lncRNAs with transcriptome-wide
half-life measurements. No tested or proxy-evaluated representation exceeded
AUROC 0.70 or supported robust continuous regression, and all performed similarly
to simple k-mer composition baselines; performance on transcripts with documented
dynamic regulation was systematically weak. We interpret this shared ceiling as
an *observability gap* between the static pretraining signal available to
sequence-only models and the dynamic cellular state on which lncRNA function
most clearly depends. We introduce *dynamic grounding* — a model-agnostic,
post-hoc conditioning of static predictions on measured state variables such as
turnover and localisation — and advocate a tiered static × dynamic evaluation
standard for foundation models whose labels depend on unobserved state.

---

## Introduction

Long non-coding RNAs (lncRNAs) outnumber annotated protein-coding genes in the human
genome, yet they remain the most functionally opaque class of transcripts. For proteins, sequence
now increasingly predicts structure, and structure increasingly predicts
function^1,2^. For lncRNAs, neither implication holds reliably: primary sequence rarely
predicts secondary structure with useful accuracy, and structure predicts function only
in a handful of well-characterised cases such as NEAT1, an architectural lncRNA
central to paraspeckle biology^3^, and XIST, which guides chromatin modification on the inactive X
chromosome^4^. The gulf between sequence and function motivates an alternative
framing: can deep learning read lncRNA function directly from nucleotide sequence?

The past three years have seen an explosion of RNA-centric foundation models built on
this premise. RNA-FM^5^ adapted BERT-style masked-language pretraining to 23 million
non-coding RNA sequences. RiNALMo^6^ scaled this recipe to 650 million parameters and
36 million sequences. Genomic language models such as Evo^7^ extended the paradigm to
7 billion parameters across DNA, RNA and phage genomes. Structure predictors such as
RhoFold+^8^ brought AlphaFold-style geometry to RNA, and task-specific networks such
as DeepLncLoc^9^ target individual lncRNA properties. A practitioner entering this
space today can therefore choose from a broad, publicly released toolkit of
sequence-to-function models.

Yet when representations drawn from these models are asked to predict a concrete
functional readout for lncRNAs — how long a transcript survives in the cell — the
task turns out to be harder than the sequence-only framing would suggest. In the
benchmark presented here, none of the tested or proxy-evaluated representations
showed a reproducible advantage over simple k-mer baselines, and the transcripts on
which every representation failed shared a single qualitative feature: published
evidence of dynamic regulation.

Rather than treat this as a negative result specific to lncRNA turnover, we argue
that it is a concrete instance of a broader issue in foundation-model evaluation.
The dynamic axes on which lncRNA function most clearly depends — transcript
turnover, subcellular localisation, condition-specific induction — are measurable
properties of the cellular state, not properties of the sequence, and they are
systematically absent from the pretraining corpora that modern RNA foundation
models learn from. We refer to this as an *observability gap*: a systematic
mismatch between the variables observable at pretraining time and the variables
that determine downstream evaluation labels. Under such a mismatch, increasing
model capacity without enlarging the observed-variable set cannot recover the
missing signal. On that framing, this Perspective makes three connected
proposals: (i) observability gaps should be stated explicitly in foundation-model
documentation and benchmarking practice; (ii) *dynamic grounding* — post-hoc
conditioning on orthogonal measured state variables — is a natural mitigation
that does not require retraining; and (iii) a tiered static × dynamic evaluation
standard should become a minimum bar when evaluation labels depend on unobserved
state.

The empirical scope of this Perspective is deliberately narrow: lncRNA *stability*
measured by half-life, as a single functional axis on which the argument can be
made concrete. Other functional tasks — interaction prediction, subcellular localisation,
expression-dynamics prediction — are beyond our primary benchmark but motivate the
framework proposed later.

---

## RNA AI for lncRNAs today

We organise the current landscape of lncRNA-relevant models by pretraining paradigm
rather than by chronology (Fig. 1, Table 1). Four pretraining families are represented
in the public toolchain: sequence foundation models, genomic language models,
structure predictors, and function-specific networks. This organisation mirrors how a
practitioner chooses a model — the first decision is what the model was trained to do,
not when it was released.

*Sequence foundation models* learn from RNA sequence alone. RNA-FM^5^ is the canonical
example (BERT-style, 100 million parameters, 1,024-nucleotide context, 23 million
ncRNA sequences); its 640-dimensional per-residue embedding has become a default
featurization for downstream RNA tasks. RiNALMo^6^ scales this recipe to 650 million
parameters with ALiBi positional encoding. Both learn from sequence alone: turnover,
localisation and abundance are absent from their supervision signal.

*Genomic language models* extend this to mixed DNA/RNA corpora. Evo^7^ trains 7
billion parameters over prokaryotic and phage genomes with a 131-kilobase
single-nucleotide-resolution context (StripedHyena architecture). For the lncRNA
benchmark we substitute ERNIE-RNA^16^ (86 M parameters, transformer with
base-pairing-aware attention), the closest publicly released post-2023
genomic-scale RNA language model that is tractable on CPU; we treat it as an
Evo-*class* representation (genomic-scale LM pretrained on nucleotide sequence)
rather than as Evo itself, and the proxy-to-full gap is revisited in Methods and
the Outlook. For lncRNAs — eukaryotic, poorly conserved and often multi-exonic —
the applicability of prokaryote-dominant pretraining is a testable hypothesis
that our benchmark addresses.

*Structure predictors* such as RhoFold+^8^ bring end-to-end tertiary-structure
prediction to RNA. Their MSA-dependency is a well-known failure mode for lncRNAs,
where evolutionary conservation is weak and Rfam coverage is sparse.

*Function-specific networks* such as DeepLncLoc^9^ train directly against a functional
axis — subcellular localisation in that case. Whether such narrowly specialised
representations transfer to a *different* function (here, half-life) is an open
question we test.

Across these four paradigms the downstream task space for lncRNAs is dominated by
localisation, RNA–RNA / RNA–protein interaction, and structural prediction. The four
paradigms differ sharply in architecture and training corpus, but they share a single
structural feature: none of them receives any direct supervision from dynamic cellular
state — turnover, localisation, stress response or cell-cycle phase are absent from
the pretraining signal of every model listed above. Functional axes that remain
comparatively unbenchmarked accordingly cluster around this same axis: *half-life*
(addressed here), *stress-responsive induction kinetics* and *condensate-forming
propensity* are all dynamic, context-dependent properties that cannot be read
faithfully from a static sequence snapshot. Whether any amount of sequence-only
pretraining can internalise such dynamic signal is the open empirical question our
benchmark is designed to probe.

---

## Benchmarking RNA turnover as a functional axis

We benchmark five representation classes against transcriptome-wide RNA
half-life ground truth. Two are evaluated directly (RNA-FM, DeepLncLoc) and three
through CPU-feasible proxies chosen to preserve their representational class
(4-mer composition for RiNALMo 650 M; ERNIE-RNA 86 M for Evo-class genomic
language models; ViennaRNA thermodynamic descriptors for RhoFold+). Box 1 makes
the direct-versus-proxy distinction precise; because three classes are accessed
through proxies rather than full-scale inference on the original weights, the
benchmark is designed to probe *class-level* ceilings rather than to rank
individual models, and within-class numerical differences should be read with
that scope in mind. Ground truth was produced by three complementary
pulse-labelling chemistries: BRIC-seq^10^ on HeLa and HeLa-TetOff, SLAM-seq^11^
on mouse embryonic stem cells, and TimeLapse-seq^12^ on K562. Fig. 1 summarises
the chemistry and the resulting distribution of measured half-lives. The unified
test set comprises 256 GENCODE v44 lncRNAs expressed above TPM ≥ 3 in at least
one cell system; 116 are binary-classifiable as stable (half-life > 4 h) or
unstable (< 2 h), and 140 are intermediate and used only for continuous
regression. For each sequence we extracted per-transcript embeddings from each
of the five representation classes and evaluated them with a common downstream
stack: logistic regression, ridge regression, and a 2-layer MLP with 64 hidden
units.

**No tested representation achieved strong, stable classification.** Under
gene-disjoint 5-fold stratified cross-validation (the primary evaluation protocol,
Methods), the best representation × classifier combinations are the RiNALMo-proxy
4-mer MLP (AUROC 0.695 ± 0.155), the DeepLncLoc 3-mer MLP (0.690 ± 0.167), the RNA-FM
MLP (0.672 ± 0.141) and the Evo-class proxy (ERNIE-RNA) MLP (0.655 ± 0.149)
(Fig. 2). The RhoFold+ proxy (ViennaRNA thermodynamic descriptors) is the weakest at
0.396 ± 0.118 with MLP, improving to 0.589 ± 0.133 with logistic regression. The four
leading combinations fall within approximately one standard deviation of each other
and their bootstrap 95% confidence intervals overlap substantially; we therefore read
them as performing similarly within the variance of this benchmark rather than as a
strict ranking. No tested representation exceeded AUROC 0.70 under our evaluation
setting — a level we adopt here as an informal usefulness threshold, consistent
with recent independent RNA foundation-model comparisons^13^. Taken together, these results suggest that under the
conditions tested, scaling or swapping among sequence-only representations is
insufficient to overcome a shared ceiling in lncRNA half-life classification.

**Continuous half-life regression is near chance.** Predicting log₂(half-life) in
hours as a continuous variable against the full 256-sequence set, the best single
model is the DeepLncLoc 3-mer MLP with Spearman ρ = 0.186 ± 0.087, followed by RNA-FM
(ρ = 0.153 ± 0.110) (Fig. 3). RMSE in log₂(h) units clusters between 1.02 and 1.05
across all representations — consistent with a near-intercept fit. The near-intercept
behaviour of every representation on continuous half-life is consistent with the view
that the remaining predictive information resides in dynamic cellular state rather
than in sequence alone.

**A classical k-mer baseline performs similarly to every tested representation.**
The 3-mer (64 dimensions) and 4-mer (256 dimensions) composition vectors fed to an
MLP recover AUROC values within the cross-validation variance of RNA-FM's
640-dimensional pretrained embedding (Fig. 2, colour-coded by representation class).
Read together with the recent RNAGenesis comparisons^13^, this observation
motivates testing whether dynamic measurements add information that is not already
recoverable from sequence-only pretraining; it does not, by itself, settle the
question for larger directly-evaluated weights. Whether this ceiling lifts with
GPU-scale replication of full RiNALMo 650 M and Evo 7 B is an empirical question
that the proxy benchmark cannot close on its own.

**Cross-cell generalisation is noisy.** Under leave-one-cell-out cross-validation,
absolute AUROC values fluctuate widely (0.26 ≤ SD ≤ 0.40 across cells) because per-cell
classifiable N is small. We report LOCO as a diagnostic, not as a primary metric.

---

## A shared blind spot: consensus-failure lncRNAs

If the performance ceiling were an architecture-side artefact, one would expect
different architectures to fail on different transcripts. Instead, we find a
consensus-failure set of 12 lncRNAs misclassified by every one of the five
representation classes under gene-disjoint cross-validation (Fig. 4). This
analysis
is deliberately *hypothesis-generating*: the sample is too small to claim formal
statistical enrichment of any biological category, and we frame what follows as a
qualitative pattern that a larger calibrated benchmark should test directly. Within
that caveat, the set includes lncRNAs whose biology is well-established as
context-dependent — notably *XIST*^4^, which guides developmental
X-chromosome inactivation through chromatin-state–dependent regulation, and
*NORAD*^14^, a cytoplasmic lncRNA induced by DNA damage that sequesters PUMILIO
proteins — alongside additional transcripts (*TTTY14*, *MCM3AP-AS1*, *NBR2* and
others) with reported cell-type-specific or stress-responsive expression. These
serve as illustrative examples of transcripts for which a static sequence snapshot
is unlikely to capture the relevant regulatory context, rather than as a
biological claim about the 12-transcript set as a whole. The pattern is consistent with the
observability-gap reading: transcripts whose function is carried by cellular
state are the ones on which static representations agree least.

A length-stratified breakdown (Supplementary Fig. 1) shows RNA-FM performing
relatively better on shorter lncRNAs and degrading on longer ones — consistent
with its 1,024-nucleotide context window — whereas the Evo-class proxy shows the
opposite trend. Context length constrains which static features a model
can access, but does not supply the dynamic signal missing from every
representation uniformly.

---

## Dynamic grounding: closing the observability gap

Together, the consensus failure, the k-mer-competitive ceiling and the near-chance
regression point to a shared missing ingredient: the dynamic state of the cell.
Every representation tested is pretrained on a frozen transcriptome snapshot without
kinetic, subcellular or condition-dependent annotation. The relevant state variables
are themselves measurable — RNA turnover at transcriptome scale by BRIC-seq^10^,
SLAM-seq^11^ and TimeLapse-seq^12^; subcellular localisation by CeFra-seq^17^ and
APEX-seq^18^ — and the field has already collected them. What is missing is a
principled way to connect them back to models that were never exposed to them.

Viewed as a machine-learning problem, this is a specific instance of a more general
pattern: a pretrained representation *r(x)* is evaluated on a label *y* that depends
not on *x* alone but on *(x, z)*, where *z* is an orthogonal state variable never
seen at pretraining time. Scaling *r* cannot recover what *z* carries; conditioning
*r* on a measured *z* can. We call the resulting procedure *dynamic grounding*: a
model-agnostic post-hoc conditioning of static foundation-model predictions on
measured state variables that capture the axes the pretraining corpus leaves out.

For the lncRNA half-life setting, this specialises to the tiered construction
sketched in Fig. 5. The top tier is the static model output, preserved as-is. The
middle tier is the joint turnover-and-localisation prior *p(t), ℓ(t)*, computed from
published measurements on the closest matched cell system. The bottom tier is a
calibrated output projected onto interpretable biology bins (stable-nuclear,
stable-cytoplasmic, unstable-nuclear, unstable-cytoplasmic, intermediate). The
grounding layer is model-agnostic: it can be applied to any of the five
representation classes evaluated here, or to any future sequence model, without
retraining the underlying weights. A minimal implementable prototype (Box 2)
requires only published resources and a single calibration step, and is therefore
immediately actionable — no new data, no new training runs, no privileged access to
foundation-model weights.

Dynamic grounding also implies a corresponding change in how models should be
evaluated. We propose a **tiered, observability-aware evaluation standard** with two
axes. The *static axis* tests raw model output against sequence-only ground truth —
the current default. The *dynamic axis* tests calibrated output against turnover and
localisation ground truth — the standard this Perspective argues must now be adopted.
A representation that performs well on the static axis but degrades on the dynamic
axis is exposed as sequence-saturating; one that performs on both is biology-aware.
Under this standard, every representation class benchmarked here would be classified
as static-axis only. The same framing extends naturally beyond lncRNAs: any
foundation model evaluated on a label that depends on an unobserved state variable
should, in principle, be benchmarked on both axes.

**Why not retrain, fine-tune, or scale?** Three natural objections deserve
explicit replies. *Retraining.* The most direct response to an observability
gap is to fold turnover and localisation measurements into the pretraining
corpus itself. In the long run this is almost certainly where the field will
land. In the short run it is not tractable: the union of published BRIC-seq,
SLAM-seq and TimeLapse-seq experiments covers of the order of 10^4^ annotated
transcripts across a handful of cell systems^10–12^, two to three orders of
magnitude below the 10^7^ sequences that current RNA foundation models absorb
at pretraining time^5–8^. Dynamic grounding does not replace that future
effort; it uses the dynamic measurements where they are currently abundant —
at the level of individual transcripts and cell systems — and leaves
pretraining scale untouched. *Fine-tuning.* A weight-update-based response
would fine-tune each foundation model on the available turnover labels and
publish a new checkpoint per model. This tightly couples the correction to
one model generation, rederives it whenever weights change, and fails for
models whose weights are not released. Grounding is model-agnostic by
construction: the *p(t), ℓ(t)* prior is computed once from measurements and
composed with any static output, released or closed. That also preserves the
fairness of cross-model comparison — the static representation being
benchmarked is not itself altered by the grounding step. *Scaling.* One might
suspect that a longer context window, a richer tokeniser or a larger
parameter count would eventually close the gap without any auxiliary signal.
The length-tertile and k-mer-baseline results argue against this reading:
a 4-mer composition feature that preserves no positional information matched
100 M- to 7 B-parameter sequence models (Fig. 2, Supplementary Fig. 1). The
bottleneck is not sequence representation; it is the absence of the state
variable *z* from the pretraining distribution. A corollary is that *z* need
not be measured perfectly to be useful. Published turnover and localisation
measurements carry their own batch effects and cell-line dependence, and
dynamic grounding deliberately treats *z* as a bucketed empirical prior
rather than a gold label, so that modest measurement noise translates into
modest rather than pathological calibration error. *Coverage.* A final
concern is whether the published turnover atlases themselves cover lncRNAs
densely enough to serve as a grounding prior. Transcriptome-wide labelling
methods bias toward more highly expressed transcripts, and lncRNAs are on
average expressed at lower levels than protein-coding genes. The practical
response, implemented in the prototype in Box 2, is to compute *p(t)* with
explicit confidence intervals per expression stratum and to fall back to a
neutral prior for transcripts outside the measured coverage envelope — an
explicit statement that the static prediction cannot be grounded on this
transcript. That behaviour is preferable to silent overconfidence in the
ungrounded case, and it turns the coverage boundary of the dynamic atlas
into a first-class benchmark property rather than a hidden caveat.

---

## Outlook

Three limitations frame what this benchmark can and cannot support. First, three of
the five representation classes are assessed through CPU-feasible proxies rather
than through direct inference on the original weights; GPU-scale replication with
full RiNALMo 650 M and Evo 7 B is reserved for follow-up work, and the proxy-to-full
AUROC gap is expected on the basis of independent zero-shot comparisons^13^ to be of
the order of a few percentage points — small relative to the within-fold variance
observed here, but not settled by this benchmark alone. Second, the test set of 256
lncRNAs (116 classifiable) is modest, per-fold AUROC standard deviations are
correspondingly large, and gene-disjoint cross-validation does not yet control for
functional-family leakage. Third, the ground-truth half-life datasets differ in
chemistry and cell system; per-cell calibration reduces but does not eliminate batch
effects.

Against those caveats, two readings are supported both by this benchmark and by
independent RNA foundation-model comparisons^13^. First, static sequence-only
representations — whether evaluated directly or via proxy — did not support
robust lncRNA half-life prediction under the conditions tested, motivating
tests of whether dynamic measurements add information that pretraining does not
recover. Second, the field already has those measurements: transcriptome-wide
turnover^10–12^ and localisation^17,18^ are publicly available, and dynamic
grounding (Fig. 5, Box 2) provides a concrete, retraining-free route to bring
them into contact with existing foundation-model outputs. Building
biology-grounded RNA foundation models will eventually require pretraining
corpora with kinetic and spatial annotation as first-class supervision, but the
tiered, observability-aware evaluation standard argued for here can be adopted
today on existing models and data.

More broadly, the pattern we document — a static-axis ceiling accompanied by
systematic weakness on dynamically regulated substrates — is unlikely to be
specific to lncRNA stability. Analogous dynamic axes (ribosome occupancy
kinetics, chromatin-state half-life, splicing response times) are measurable but
similarly absent from the current generation of foundation-model training
corpora. We offer the tiered evaluation standard as a common baseline for the
RNA and machine-learning communities, and as a testable template for any
foundation model whose evaluation labels depend on unobserved state.

---

## Box 1. Terminology

To avoid confusion between classes of inputs used in the benchmark, three terms are
used throughout.

- **Directly evaluated model** — the published model run with its original weights
  and inference pipeline (here: RNA-FM, DeepLncLoc).
- **Proxy representation** — a CPU-feasible surrogate that captures the
  *representational class* of a published model without reproducing its weights
  (here: 4-mer composition as a proxy for RiNALMo 650M; ERNIE-RNA 86M as a proxy for
  Evo-class 7B genomic language models; ViennaRNA 2-D descriptors as a proxy for
  RhoFold+).
- **Classical baseline** — non-neural sequence features (k-mer composition) used for
  reference, conceptually separate from the proxy representations above.

Full-scale GPU replication of the proxied models is planned and will be reported
separately. Throughout this Perspective, "five representations" refers to the
representational classes evaluated, not to direct inference of all five published
weights.

---

## Box 2. Minimal implementable prototype of dynamic grounding

A minimal version of the dynamic-grounding layer requires only published resources
and a single calibration step.

Given a transcript *t*:

1. Run the static model *M* to obtain a raw stability score *s(t) ∈ [0, 1]*.
2. Retrieve the half-life posterior *p(t) = N(μ, σ²)* of *t* from the closest matched
   BRIC-seq / SLAM-seq / TimeLapse-seq dataset; default to the cell-system average
   when *t* is unmeasured.
3. Retrieve the localisation prior *ℓ(t)* from CeFra-seq^17^ / APEX-seq^18^, or —
   when such measurements are unavailable — fall back on DeepLncLoc predictions as
   a pragmatic surrogate, noting that this reuses one of the evaluated models as a
   provider of prior information rather than as a target of grounding.
4. Combine into a calibrated prediction *s′(t) = σ(w₁ · logit s(t) + w₂ · z(μ_p)
   + w₃ · logit ℓ(t))*, where *z(μ_p)* is the z-standardised posterior-mean
   half-life and the three weights are fitted by logistic calibration on a
   held-out fold.

In practice, step 4 reduces to a low-dimensional logistic regression that combines
the static score with two orthogonal priors, and can be implemented with standard
calibration toolkits. The only additional inputs beyond the original model are
(*p*, *ℓ*), both of which are recoverable from publicly available resources for
any cell system with a BRIC-seq / SLAM-seq / TimeLapse-seq dataset. Richer
conditioning (structured priors, kinetic-aware pretraining objectives, multi-task
training on turnover labels) are natural next steps but are not required for the
framework to be useful.

---

## Methods summary

The benchmark is held to the minimum detail needed to interpret the Perspective;
full protocols and code will accompany the deposition. Test set: 256 lncRNAs from
GENCODE v44 (human) and GENCODE vM33 (mouse) expressed above TPM ≥ 3 in at least
one of four cell systems (HeLa-TetOff, K562, mouse embryonic stem cells, mouse
embryonic fibroblasts); binary labels stable (t½ > 4 h, n = 40) and unstable
(t½ < 2 h, n = 76) define the 116-sequence classification subset, with the
remaining 140 transcripts used only for continuous regression.
Half-life ground truth combines BRIC-seq^10^, SLAM-seq^11^ and TimeLapse-seq^12^
across the four cell systems. Representations: two direct (RNA-FM^5^, DeepLncLoc^9^)
and three proxies capturing representational class (4-mer composition for
RiNALMo^6^; ERNIE-RNA^16^ for Evo-class^7^ genomic language models; ViennaRNA^15^
thermodynamic descriptors for RhoFold+^8^; see Box 1). Downstream stack: logistic
regression, ridge regression, and a 2-layer MLP (64 hidden units). Primary
cross-validation is gene-disjoint 5-fold stratified, so that no `gene_id`
contributes to both the training and test folds; leave-one-cell-out is reported as
a diagnostic only. Uncertainty is reported as per-fold mean ± standard deviation
with bootstrap 95% confidence intervals (1,000 resamples) for the headline
comparisons; formal paired tests (DeLong, paired bootstrap) are under-powered at
this sample size and are treated as a limitation.

## Data and code availability

Annotation (GENCODE v44 human, vM33 mouse) and the half-life ground-truth
datasets are publicly available (BRIC-seq: DDBJ DRA000345–350, DRA000357–361;
SLAM-seq: GEO GSE99978; TimeLapse-seq: GEO GSE95854). Model weights follow each
model's original repository. The Phase 1 data pipeline, Phase 2 benchmark
scripts (CPU-proxy and GPU notebooks), processed test set, embeddings,
evaluation outputs, and figure generators are openly available at
https://github.com/hidenori-tani/rna-foundation-grounding-benchmark and
archived on Zenodo at https://doi.org/10.5281/zenodo.19679759 under an MIT
(code) and CC BY 4.0 (data and figures) dual licence. The repository includes
a top-level `reproduce.sh` that regenerates the full CPU-proxy pipeline
(~45 min on a laptop) end-to-end; Colab notebooks under `benchmark/colab/`
replicate the GPU results for RiNALMo-650M, Evo-7B, and RhoFold+.

## Acknowledgements

The author thanks the laboratories that generated and publicly released the
RNA half-life datasets and RNA AI model weights used here. No dedicated funding
supported this Perspective.

## Author contributions

H.T. designed the study, conducted all analyses and wrote the manuscript.

## Competing interests

The author declares no competing interests.

---

## Figure legends

**Figure 1.** Conceptual overview. Static RNA AI models predict lncRNA function from
sequence alone, missing the dynamic axis (turnover, localisation) captured by
BRIC-seq, SLAM-seq and TimeLapse-seq. Dynamic grounding conditions static
predictions on measurable biological state.

**Figure 2.** AUROC heatmap across five RNA AI representation classes and three
downstream classifiers (logistic regression, ridge regression, MLP) under
gene-disjoint 5-fold stratified cross-validation on the 116-sequence classifiable
subset.

**Figure 3.** Predicted versus measured log₂(half-life) for each representation
class, with Spearman ρ in panel headers.

**Figure 4.** Consensus-failure analysis: the 12 lncRNAs misclassified by every
one of the five representation classes under gene-disjoint 5-fold stratified
cross-validation.

**Figure 5.** Dynamic grounding framework: static prediction → dynamic constraint
(turnover + localisation priors) → biology-aware calibrated output. Worked example:
*NORAD* (cytoplasmic, stress-responsive, a consensus-failure transcript under
static-only evaluation). The grounding layer is model-agnostic and compatible with
any existing sequence-only RNA foundation model.

## Table legends

**Table 1.** Five RNA AI representation classes benchmarked: original model,
direct-versus-proxy evaluation, embedding dimensionality, licence and primary
reference.

## Supplementary figure legends

**Supplementary Figure 1.** Length-stratified classification performance across
the five representation classes, binned by transcript-length tertile (short
< 2,387 nt; mid 2,387–4,159 nt; long > 4,159 nt).

---

## References

1. Jumper, J. *et al.* Highly accurate protein structure prediction with AlphaFold.
   *Nature* **596**, 583–589 (2021).
2. Abramson, J. *et al.* Accurate structure prediction of biomolecular interactions
   with AlphaFold 3. *Nature* **630**, 493–500 (2024).
3. Hirose, T. *et al.* NEAT1 long noncoding RNA regulates transcription via protein
   sequestration within subnuclear bodies. *Mol. Biol. Cell* **25**, 169–183 (2014).
4. Statello, L. *et al.* Gene regulation by long non-coding RNAs and its biological
   functions. *Nat. Rev. Mol. Cell Biol.* **22**, 96–118 (2021).
5. Chen, J. *et al.* Interpretable RNA foundation model from unannotated data for
   highly accurate RNA structure and function predictions. *bioRxiv*
   10.1101/2022.08.06.503062 (2022).
6. Penić, R. J. *et al.* RiNALMo: general-purpose RNA language models can generalize
   well on structure prediction tasks. *Nat. Commun.* **16**, 5671 (2025).
7. Nguyen, E. *et al.* Sequence modeling and design from molecular to genome scale
   with Evo. *Science* **386**, eado9336 (2024).
8. Shen, T. *et al.* Accurate RNA 3D structure prediction using a language
   model-based deep learning approach. *Nat. Methods* **21**, 2287–2298 (2024).
9. Zeng, M. *et al.* DeepLncLoc: a deep learning framework for long non-coding RNA
   subcellular localization prediction based on subsequence embedding.
   *Brief. Bioinform.* **23**, bbab360 (2022).
10. Tani, H. *et al.* Genome-wide determination of RNA stability reveals hundreds of
    short-lived noncoding transcripts in mammals. *Genome Res.* **22**, 947–956
    (2012).
11. Herzog, V. A. *et al.* Thiol-linked alkylation of RNA to assess expression
    dynamics. *Nat. Methods* **14**, 1198–1204 (2017).
12. Schofield, J. A. *et al.* TimeLapse-seq: adding a temporal dimension to RNA
    sequencing through nucleoside recoding. *Nat. Methods* **15**, 221–225 (2018).
13. Zhang, Z. *et al.* RNAGenesis: a generalist foundation model for functional RNA
    therapeutics. *bioRxiv* 10.1101/2024.12.30.630826 (2024).
14. Lee, S. *et al.* Noncoding RNA NORAD regulates genomic stability by
    sequestering PUMILIO proteins. *Cell* **164**, 69–80 (2016).
15. Lorenz, R. *et al.* ViennaRNA Package 2.0. *Algorithms Mol. Biol.* **6**, 26
    (2011).
16. Yin, W. *et al.* ERNIE-RNA: an RNA language model with structure-enhanced
    representations. *Nat. Commun.* **16**, 10076 (2025).
17. Benoit Bouvrette, L. P. *et al.* CeFra-seq reveals broad asymmetric mRNA and
    noncoding RNA distribution profiles in Drosophila and human cells. *RNA*
    **24**, 98–113 (2018).
18. Fazal, F. M. *et al.* Atlas of subcellular RNA localization revealed by
    APEX-seq. *Cell* **178**, 473–490 (2019).
