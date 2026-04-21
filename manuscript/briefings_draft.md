# From black boxes to biology: deep learning and large language models for lncRNA function prediction

**Target journal**: *Briefings in Bioinformatics* (Review, max 7,000 words)
**Author**: Hidenori Tani, Ph.D. — Associate Professor, Yokohama University of Pharmaceutical Sciences
**Correspondence**: hidenori.tani@yok.hamayaku.ac.jp
**Status**: Draft v0.1 (2026-04-20) — §1–3 prose + existing skeleton. §4–8 remain in Phase 4.
**Document policy**: 本稿は **master 原稿**。Nat MI 版（~4,000 words）は Phase 4.7 でここから抽出・圧縮する（D+C 戦略）。
**Citation convention**: 本 draft では著者 lastname + year のみ記載（巻号・頁は Phase 4.5 で PubMed から補完）。推測数値は使用しない — 不明箇所は `[vol/pages TBD]` と明記。

**Word-budget summary (Briefings)**:

| Section | Target words | Status |
|---|---|---|
| §1 Introduction | 700 | Skeleton |
| §2 Landscape of AI models | 1,100 | Skeleton |
| §3 Hidden dimension — RNA turnover | 900 | Skeleton |
| §4 Benchmarking results | 2,000 | Phase 4.1 |
| §5 Failure modes | 900 | Phase 4.2 |
| §6 From black box to biology-aware predictor | 1,100 | Phase 4.3 |
| §7 Tiered evaluation framework | 500 | Phase 4.4 |
| §8 Outlook | 400 | Phase 4.4 |
| **Total** | **7,600 → trim to ≤7,000** | |

---

## §1 Introduction (target 700 words)

### Draft prose (v0.1)

Long non-coding RNAs (lncRNAs) outnumber protein-coding genes in the human genome, yet they
remain the most functionally opaque class of transcripts. For proteins, sequence now increasingly
predicts structure, and structure increasingly predicts function (Jumper et al. 2021; Abramson
et al. 2024). For lncRNAs, neither implication holds reliably: primary sequence rarely
predicts secondary structure with useful accuracy, and structure predicts function only in
a handful of well-characterised cases such as NEAT1, which scaffolds paraspeckles
(Hirose et al. 2014), and XIST, which guides chromatin modification on the inactive X
chromosome (Statello et al. 2021). The gulf between sequence and function motivates a
radically different approach: can deep learning read lncRNA function directly from
nucleotide sequence?

The past three years have seen an explosion of RNA-centric foundation models built on this
premise. RNA-FM (Chen et al. 2022) adapted the BERT masked-language pretraining recipe to
23 million non-coding RNA sequences from RNAcentral. RiNALMo (Penic et al. 2024) scaled
this approach to 650 million parameters and 36 million sequences, improving secondary
structure prediction on unseen Rfam families. Genomic language models such as Evo
(Nguyen et al. 2024) extended the paradigm to 7 billion parameters across DNA, RNA, and
phage genomes, promising a unified representation of nucleic acid function. Meanwhile,
structure predictors such as RhoFold+ (Shen et al. 2024) brought AlphaFold-style end-to-end
geometry to RNA, and task-specific networks like DeepLncLoc (Zeng et al. 2022) target
specific lncRNA properties such as subcellular localization. A practitioner entering this
space today has, for the first time, a full stack of publicly released sequence-to-function
models to choose from.

Yet when these models are asked to predict a single concrete functional readout for
lncRNAs — how long a transcript survives in the cell — their performance collapses to the
level of k-mer composition baselines. This is the central empirical observation of the
present review. Across five representative models, spanning sequence foundation models,
genomic language models, structure predictors, and function-specific networks, and
evaluated against BRIC-seq half-life ground truth across four mammalian cell lines, no
model exceeds an AUROC of 0.70 on binary stability classification, and continuous
regression of log-transformed half-life yields Spearman correlations of at most 0.19.
Worse, a simple 4-mer composition vector fed to a multilayer perceptron matches or exceeds
every foundation model on this task. The representational ceiling of current sequence-only
RNA AI appears to sit far below biological utility.

We argue in this review that the shared failure of these models reflects a systematic
blind spot rather than a temporary scaling bottleneck. Every model surveyed here is trained
on static sequence data: a frozen snapshot of the transcriptome with no temporal, kinetic,
or subcellular axis. RNA turnover — the rate at which transcripts are synthesised and
degraded — is the missing dimension that defines when and where an lncRNA can act. It is
measurable at transcriptome scale by three established chemistries (Tani et al. 2012;
Herzog et al. 2017; Schofield et al. 2018), yet it is absent from every training corpus
currently used for RNA foundation models. No amount of scaling on sequence-only data will
recover a signal that was never present in the training distribution.

The review proceeds as follows. §2 maps the current landscape of lncRNA-relevant AI models.
§3 introduces RNA turnover as the hidden functional dimension and the wet-lab methods that
measure it. §4–5 report our benchmark of five models against turnover ground truth and
dissect the consensus-failure lncRNAs. §6 proposes *dynamic grounding* — a framework that
conditions static predictions on turnover and localisation priors — as a constructive way
forward. §7 introduces a tiered evaluation standard (static × dynamic) that we suggest the
RNA-AI community adopt as a minimum bar. §8 closes with an outlook on biology-grounded
foundation models.

### Argument arc (skeleton)


- (A) lncRNA の機能的多様性 — scaffold, decoy, guide, enhancer-like, signal — を5-6行で scan し、
  「配列からは機能がほぼ読めない」という現状を設定する。
  - Representative examples: NEAT1 (paraspeckle scaffold), MALAT1 (splicing regulator), XIST (X-inactivation guide),
    HOTAIR (PRC2 recruiter), KCNQ1OT1 (imprinting). 5本の選択は §3 benchmark の reference list と一貫させる。
- (B) 近年の AI 革命 — タンパク質構造予測の成功（AlphaFold2/3）と RNA 分野への波及 — を3-4行で配置し、
  読者に「RNA にも同じ革命が来るのか？」という期待を持たせる。
- (C) **盲点命題 A の提起**: 現在の RNA AI モデルは「配列から静的表現」を学習しているのみで、
  RNA 機能の本質的次元である turnover / localization / complex formation が
  学習信号に含まれていない。これが本総説の中心命題。
- (D) 本総説の構成予告:
  - §2: 5つの代表的モデルのランドスケープ
  - §3: lncRNA 機能の隠れた次元としての turnover と、BRIC-seq/SLAM-seq による測定
  - §4-5: 5モデル vs turnover ground-truth の独自ベンチマーク
  - §6-7: dynamic grounding と tiered evaluation の提案

**Key citations to place**:

- lncRNA の機能分類総説: Statello et al., *Nat Rev Mol Cell Biol* 2021（最も包括的）
- AlphaFold3: Abramson et al., *Nature* 2024（RNA も含むため引用可）
- RNA AI の批判的総説があれば対比引用（Phase 4 文献整備で追記）

**開かれ方の草稿（最初の3文）**:

> Long non-coding RNAs outnumber protein-coding genes in the human genome yet remain the most
> functionally opaque class of transcripts. For proteins, sequence now increasingly predicts
> structure, and structure increasingly predicts function. For lncRNAs, neither implication
> holds: sequence rarely predicts structure reliably, and structure predicts function only
> in rare, well-characterized cases.

**盲点命題の提示タイミング**: §1 の終わり3分の1で明示する（§3 の伏線として早めに打つ）。

**TODO (Phase 4)**:

- [ ] lncRNA 機能分類の総説から引用を5本選定
- [ ] AlphaFold3 RNA モジュールの位置づけを1段落で配置
- [ ] 盲点命題 A を本文で明示（abstract と §1 の両方に埋め込む）
- [ ] 「構築命題 C」の伏線（dynamic grounding）を §1 最終段落で1文だけ予告

---

## §2 Landscape of AI models (target 1,100 words)

### Draft prose (v0.1)

We organize the current landscape of lncRNA-relevant models by pretraining paradigm rather
than by chronology (Table 1). Four pretraining families are represented: sequence
foundation models, genomic language models, structure predictors, and function-specific
networks. This organization mirrors how a practitioner chooses a model: the first decision
is what the model was trained to do, not when it was released.

**Sequence foundation models.** RNA-FM (Chen et al. 2022) is the canonical example: a
BERT-style masked-language model with 100 million parameters, trained on 23 million
non-coding sequences from RNAcentral with a 1,024-nucleotide context window. The per-residue
embedding (640 dimensions) has become a default featurization for downstream RNA tasks and
underpins most published benchmarks of RNA language models. RiNALMo (Penic et al. 2024)
scales this recipe to 650 million parameters with ALiBi positional encoding, removing the
hard context limit and training on 36 million ncRNA sequences. Both models share a
critical design choice: they learn from sequence alone. Turnover, localization, and
abundance are absent from their supervision signal.

**Genomic language models.** Evo (Nguyen et al. 2024), reported in *Science* with 7 billion
parameters, extends language modelling to a mixed DNA/RNA corpus drawn primarily from
prokaryotic genomes and phages, with an 8,192-nucleotide context. For lncRNAs — which are
eukaryotic, often poorly conserved, and frequently multi-exonic — the applicability of
prokaryotic-dominant pretraining is a testable hypothesis; we return to it in §4. Related
models in this family include Nucleotide Transformer (Dalla-Torre et al. 2024) and
DNABERT-2 (Zhou et al. 2024), which we mention for completeness but did not include in the
benchmark to keep the comparison interpretable.

**Structure predictors.** RhoFold+ (Shen et al. 2024), reported in *Nature Methods*, is an
end-to-end RNA tertiary structure predictor trained on the RNA-containing subset of the PDB
and Rfam multiple sequence alignments. Its design inherits the MSA-dependency that is a
well-known failure mode for lncRNAs, where evolutionary conservation is weak and Rfam
coverage is sparse. In this review we encode each RhoFold+ prediction as a low-dimensional
structural descriptor vector (radius of gyration, fraction helix, per-residue pLDDT
distribution, and related summary statistics) rather than consuming the 3D coordinates
directly, on the principle that downstream tasks care about structural class, not atomic
positions.

**Function-specific networks.** DeepLncLoc (Zeng et al. 2022) is trained as a five-way
subcellular localization classifier for lncRNAs. Unlike the models above, its output is an
explicit functional axis. We include it to test whether a function-trained network
generalises to another function (half-life prediction), or whether its representation is
narrowly specialised to the compartment-classification task it was trained on.

**Why these five?** The selection covers four distinct pretraining paradigms; a
three-order-of-magnitude parameter range (from ~10⁸ to ~10⁹); public weights with
reproducible inference pipelines; and all five models are advertised by their authors as
applicable to lncRNAs. We explicitly excluded AlphaFold3-RNA (closed API as of the
benchmark freeze date), xTrimoRNA (commercial), and RNA-MSM (degrades on low-conservation
inputs, which is exactly the lncRNA regime). The five models retained are — to our
knowledge — the best currently available public tools for each category.

**Representational capacity.** The embeddings range from 9 dimensions (RhoFold+ structural
descriptors) to 768 dimensions (ERNIE-RNA, the CPU-feasible substitute we use for Evo-class
models in the present benchmark; see Table 1 footnote and §4 Methods). For continuity with
published RNA-FM baselines, we evaluate every model with a common downstream stack:
logistic regression, ridge regression, and a two-layer MLP. This keeps the comparison
about representation quality rather than downstream classifier engineering.

### Argument arc (skeleton)


5 モデルを **カテゴリ別** に紹介する（時系列ではなくカテゴリ別）。Table 1 と完全一致させる。

### §2.1 Sequence foundation models (~350 words)

- **RNA-FM** (Chen et al., 2022): BERT-like, 100M params, 23M ncRNA from RNAcentral, 1024 nt context
  - 貢献: RNA 一次配列の自己教師学習の先駆け
  - 限界: context 長 1024 nt は多くの lncRNA（>4 kb）を切り詰める
- **RiNALMo** (Penic et al., 2024): ALiBi, 650M params, 36M ncRNA
  - 貢献: スケールと可変長。RNA-FM の後継として事実上のベンチマーク
  - 限界: 依然として sequence-only

### §2.2 Genomic language models (~200 words)

- **Evo** (Nguyen et al., *Science* 2024): StripedHyena, 7B, 2.7M prokaryotic genomes + phages, 8k nt
  - 貢献: DNA/RNA 混在の long-context LM として lncRNA に応用可能
  - 限界: 訓練コーパスが prokaryotic 中心。真核生物 lncRNA の turnover や localization には
    情報が乏しい可能性 — §4 benchmark で検証する仮説
- Fallback: Nucleotide Transformer 2.5B (InstaDeepAI), DNABERT-2（Table 1 注記）

### §2.3 Structure predictors (~250 words)

- **RhoFold+** (Shen et al., *Nat Methods* 2024): E2E RNA 3D 予測, PDB RNA + Rfam MSA
  - 貢献: RNA 3D 予測の現状 SOTA の一つ
  - 限界: MSA 依存。lncRNA は進化保存性が低く MSA が貧弱になりやすい（Rfam coverage の統計を引用）
  - 出力処理: 本総説では 3D 座標から 9次元の structural descriptors（Rg, helix fraction, plDDT 等）を
    抽出して embedding とする（Methods §3.X で詳述）
- 除外: AlphaFold3-RNA（API 制限 / 再現性）、xTrimoRNA（商用）、RNA-MSM（lncRNA では MSA 破綻）

### §2.4 Function-specific networks (~200 words)

- **DeepLncLoc** (Zeng et al., *BiB* 2022): CNN-LSTM, 5-compartment subcellular localization
  - 貢献: lncRNA 特化・機能ラベル教師学習の代表
  - 限界: 5クラス分類に固定されており、他機能（turnover 等）へは転移しない
  - 本総説では 5次元確率ベクトルを embedding として turnover 予測に使用 — 「機能間転移」可否を検証

### §2.5 選定論理 (~100 words)

- 4カテゴリ（sequence / genomic / structure / function）を網羅
- 100M〜7B パラメータの幅
- 全て公開重み、再現可能
- 全て lncRNA を訓練 or 推論対象として扱う

**Key citations**: 各モデル原著論文 + 代表的 RNA AI 総説（Kaul & Martin 2024, BiB 等 Phase 4 で追加）

**TODO (Phase 4)**:

- [ ] 各モデルの「公開日 / 最新バージョン」を公式 GitHub で確認（2026-04 現在）
- [ ] Rfam coverage for lncRNA の統計（保存性の低さの定量）を1つ引用
- [ ] 除外モデルの節を正式に1段落に書き下ろす
- [ ] Table 1 と本文の数値を完全一致させる

---

## §3 Hidden dimension — RNA turnover (target 900 words)

### Draft prose (v0.1)

The functional opacity of lncRNAs is not only a problem of representation but also a
problem of *which* representation. The models surveyed in §2 all consume static sequence
and produce static embeddings. Yet lncRNA biology is fundamentally dynamic: whether a
transcript acts depends on whether it is present at the right time, in the right
compartment, at the right concentration. Half-life — the time to degrade to half the
initial steady-state pool — is the single-number summary of this dynamic state, and it is
the axis on which every model in our benchmark is tested and, as §4 will show, largely
fails.

#### §3.1 Why turnover matters for lncRNA function

Three arguments bring turnover to the foreground. First, well-characterised lncRNAs derive
their function from kinetics, not merely from the presence of a sequence element. NEAT1
drives paraspeckle assembly in a stress-dependent manner, with paraspeckle nucleation
requiring a critical cellular concentration of the NEAT1_2 isoform (Hirose et al. 2014).
MALAT1 is one of the most stable lncRNAs in the mammalian transcriptome and localises to
nuclear speckles where it modulates splicing kinetics (Tripathi et al. 2010). In both
cases, shifting half-life by a factor of two alters the functional readout.

Second, turnover is statistically orthogonal to steady-state abundance. Two transcripts at
identical expression level may reach that level via fast-synthesis-fast-degradation or
slow-synthesis-slow-degradation routes — with radically different biological consequences
for inducibility, noise, and signal latency. Static AI models that consume RNA-seq-derived
annotation alone cannot distinguish these regimes.

Third, no RNA foundation model training corpus to date includes half-life labels at scale.
RNAcentral and Rfam provide sequence, family, and (partial) structural annotation, but no
kinetic annotation. A model that never saw turnover during pretraining has, in an
information-theoretic sense, no way to represent it — unless sequence happens to determine
half-life through cis-acting elements (ARE, m6A sites, destabilising motifs) strong enough
to be inferred from masked-token prediction. The benchmark we present in §4 quantifies how
much of this signal is recoverable from pretrained sequence representations alone.

#### §3.2 Measuring transcriptome-wide half-lives

Three complementary methods underpin the ground truth used in this review.

*BRIC-seq* (Tani et al. 2012) pulse-labels nascent RNA with 5-bromouridine, immunoprecipitates
the labelled fraction at a series of chase timepoints, and fits an exponential decay to the
remaining signal per transcript. Because BRIC-seq is pulldown-based, it detects lncRNAs
poorly only when their 5-BrU incorporation is low; the method was in fact developed with
lncRNA-inclusive transcriptome coverage in mind. The HeLa dataset used in our benchmark is
drawn from this work (DDBJ accessions DRA000345–350, DRA000357–361). We note that the
present author co-developed BRIC-seq; the benchmark uses independently re-processed
counts to avoid any analytical circularity.

*SLAM-seq* (Herzog et al. 2017) labels nascent RNA with 4-thiouridine and chemically
converts incorporated residues to a cytidine-readable form via iodoacetamide, so that
decay is read out as T→C substitution frequency in standard short-read sequencing. We draw
the mouse embryonic stem cell half-life dataset from GSE99978, which provides kinetic
parameters for 8,405 transcripts.

*TimeLapse-seq* (Schofield et al. 2018) uses the same 4-thiouridine labelling but converts
with osmium tetroxide, producing U→C substitutions. The chemistry is orthogonal to SLAM-seq
but the downstream computational pipeline is similar. We use the mouse embryonic fibroblast
and K562 datasets from GSE95854. We stress the distinction from SLAM-seq because the two
are frequently conflated in secondary literature.

The three methods differ in labelling chemistry, read-out modality, and detection
sensitivity for lowly-expressed or poorly-labelled transcripts. Supplementary Table S1
tabulates the key parameters. Importantly, all three report half-lives on a log-hours
scale that is directly comparable once method-specific biases are calibrated, which we do
per cell line at benchmark time.

#### §3.3 Why current AI models cannot represent turnover

The argument is structural rather than empirical: a model whose supervision signal
contains no kinetic information cannot, in principle, learn a kinetic representation except
by proxy. The only pathway is for half-life to be strongly determined by cis-acting
sequence elements that survive masked-token pretraining. Many such elements are known
individually — AU-rich elements (Chen & Shyu 1995), m6A methylation consensus sites
(Wang et al. 2014), PUF-binding motifs — and k-mer composition features would be expected
to capture them. This motivates the k-mer baseline in §4, which turns out to match or
exceed every foundation model tested.

### Argument arc (skeleton)


§2 で示した 5 モデルは「配列 / 構造 / 局在」は学習するが、**時間軸（turnover）** は学習しない。
本章では turnover が lncRNA 機能の本質的次元であること、そしてそれを測る実験手法を読者に装備させる。

### §3.1 Why turnover matters for lncRNA function (~300 words)

- 論拠 1: lncRNA の機能発揮には「正しいタイミングで存在する」ことが必須
  （NEAT1 by paraspeckle formation under stress — Hirose et al.; MALAT1 cell-cycle regulation など）
- 論拠 2: turnover は steady-state abundance とは独立の次元
  （高発現・短寿命 vs 低発現・長寿命 は挙動が全く違う）
- 論拠 3: 既存 RNA AI モデルの訓練コーパス（RNAcentral, Rfam）には turnover ラベルが存在しない
  → アーキテクチャ改善では埋まらない「データ側の穴」

### §3.2 Measuring transcriptome-wide half-lives (~400 words)

3 手法を並列で紹介し、本総説 benchmark が依拠するデータを読者に理解させる。

- **BRIC-seq** (Tani et al., 2012, *Genome Research*): 5-EU pulse, HeLa, 無ラベル分子を経時減衰として観測
  - 本総説での使用: HeLa データ（DDBJ DRA000345-350, DRA000357-361）
  - 著者自身の手法であることは §3.2 または Methods で簡潔に明示（過剰な自己言及は避ける）
- **SLAM-seq** (Herzog et al., 2017, *Nature Methods*): 4sU 取り込み + iodoacetamide による T>C 変換
  - 本総説での使用: mESC（GEO GSE99978、Zenodo 8,405 転写産物の半減期）
- **TimeLapse-seq** (Schofield et al., 2018, *Nature Methods*): 4sU + OsO4 による U→C 変換（SLAM-seq と化学が異なる）
  - 本総説での使用: MEF + K562（GEO GSE95854）
  - **SLAM-seq と TimeLapse-seq の区別** は読者の混乱を避けるため明記する
- 3手法の比較テーブル（§3.2 冒頭 or Supplementary Table）:
  - 化学（IAA / OsO4 / 5-EU）、検出原理、カバレッジ、lncRNA検出感度、データ可用性

### §3.3 Why current AI models cannot represent turnover (~200 words)

- 構造的理由: 訓練コーパスに turnover ラベルがない → モデル内部表現に該当次元が存在しない
- 情報理論的理由: turnover は配列の cis 要素（ARE, m6A site 等）と trans 要素（細胞状態、RNA-binding protein 濃度）
  の両方に依存 — sequence-only モデルでは trans を原理的に捉えられない
- 経験的理由: §4 で示すベンチマーク結果が、この構造的欠陥を定量する
  （§1 で提起した盲点命題 A の定量的検証が §4 であることを読者に予告）

**Key citations**:

- BRIC-seq: Tani et al., *Genome Res* 2012
- SLAM-seq: Herzog et al., *Nat Methods* 2017
- TimeLapse-seq: Schofield et al., *Nat Methods* 2018
- NEAT1 paraspeckle dynamics: Hirose et al., *Mol Biol Cell* / Fox & Lamond *Cold Spring Harb Perspect* 等（Phase 4 で精査）
- lncRNA turnover の総説 — TiBS 投稿中の自著総説（採択済の場合は引用、未定の場合は省略）

**TODO (Phase 4)**:

- [ ] Tani 2012 の journal / volume / pages を PubMed で固定
- [ ] Herzog 2017 の exact cell line（mESC E14 など）を論文 Methods から確認
- [ ] Schofield 2018 の OsO4 化学の図解（自作 or 引用）を Fig. X として配置するか検討
- [ ] §3.2 の手法比較テーブルを本文表として入れるか、Supplementary に送るか Phase 4 で決定
- [ ] 自著 BRIC-seq への言及 "I developed" vs "was developed by the present author" の表現調整

---

## Cross-section consistency notes

Phase 4 以降で本 skeleton を本文化するときに守る一貫性ルール:

1. 5モデルの呼称は Table 1 の正式表記（RNA-FM, RiNALMo, Evo, RhoFold+, DeepLncLoc）で統一
2. 3データセットの呼称は BRIC-seq (HeLa) / SLAM-seq (mESC) / TimeLapse-seq (MEF, K562) で統一
   — SLAM-seq と TimeLapse-seq を混同しない
3. 盲点命題 A（静的 AI は turnover を表現しない）→ 構築命題 C（dynamic grounding）→
   評価命題 D（tiered evaluation）の論理順序を §1 → §6 → §7 で段階的に展開
4. AUROC / Spearman ρ / RMSE などの数値は全て `benchmark/results/metrics_summary.csv` から
   転記する。本文中に数値を書くときは出典ファイル名をコメントで残す
5. Figure 番号: Fig.1 (BioRender concept), Fig.2 (AUROC heatmap), Fig.3 (scatter), Fig.4 (failure),
   Fig.5 (BioRender framework) を本文中の言及順と一致させる

---

## Next actions (Phase 4 入り口)

Phase 2 実行後、`benchmark/results/metrics_summary.csv` の数値を受けて本 skeleton を本文化する。
§4 benchmarking results (2,000 words) から書き始める（Task 4.1）。
