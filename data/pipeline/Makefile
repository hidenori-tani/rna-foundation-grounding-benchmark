# Phase 1 Data Pipeline Makefile
#
# 使い方:
#   cd ~/claude-work/research/review-AI/data/pipeline
#   make all       # Phase 1 一式を順序通りに実行
#   make clean     # processed/ を空にして再実行
#   make check     # 依存コマンド・必要ファイルの存在を先に確認
#
# 実行時間目安: 30 分〜1 時間（GENCODE 人間版 GTF が重い）
# ディスク使用量: ~5 GB（GENCODE FASTA 含む）

PROCESSED := ../processed
RAW       := ../raw
PY        := python3

.PHONY: all check clean fetch normalize testset qc

all: check fetch normalize testset qc

check:
	@echo "==> Checking Python packages..."
	@$(PY) -c "import pandas, numpy, requests, openpyxl; print('OK')"
	@echo "==> Checking directory layout..."
	@mkdir -p $(PROCESSED) $(RAW)
	@echo "==> Ready."

# ---- fetch (Step 1.1–1.4) ----
fetch: $(PROCESSED)/bricseq_halflife.csv \
       $(PROCESSED)/slamseq_herzog_halflife.csv \
       $(PROCESSED)/timelapseseq_schofield_halflife.csv \
       $(PROCESSED)/gencode_v44_lncrna.gtf \
       $(PROCESSED)/gencode_vM33_lncrna.gtf

$(PROCESSED)/bricseq_halflife.csv:
	@echo "==> Task 1.1: Tani 2012 BRIC-seq"
	$(PY) fetch_bricseq.py --mode supp

$(PROCESSED)/slamseq_herzog_halflife.csv:
	@echo "==> Task 1.2: Herzog 2017 SLAM-seq (mESC)"
	$(PY) fetch_slamseq_herzog.py --mode zenodo

$(PROCESSED)/timelapseseq_schofield_halflife.csv:
	@echo "==> Task 1.3: Schofield 2018 TimeLapse-seq (MEF + K562)"
	$(PY) fetch_timelapseseq_schofield.py --mode supp

$(PROCESSED)/gencode_v44_lncrna.gtf $(PROCESSED)/gencode_vM33_lncrna.gtf:
	@echo "==> Task 1.4: GENCODE (human v44 + mouse vM33)"
	$(PY) fetch_gencode.py --species both

# ---- cross mapping + normalize (Step 1.5–1.6) ----
normalize: $(PROCESSED)/halflife_merged.csv

$(PROCESSED)/halflife_merged.csv: fetch
	@echo "==> Task 1.5: Cross-species gene ID mapping"
	$(PY) cross_mapping.py --source-csv $(PROCESSED)/bricseq_halflife.csv
	$(PY) cross_mapping.py --source-csv $(PROCESSED)/slamseq_herzog_halflife.csv
	$(PY) cross_mapping.py --source-csv $(PROCESSED)/timelapseseq_schofield_halflife.csv
	@echo "==> Task 1.6: Normalize half-lives (quantile + winsorize)"
	$(PY) normalize_halflife.py

# ---- test set (Step 1.7) ----
testset: $(PROCESSED)/test_set_final.csv

$(PROCESSED)/test_set_final.csv: normalize
	@echo "==> Task 1.7: Build final test set (binary + tertile labels)"
	$(PY) build_test_set.py

# ---- QC report (Step 1.8) ----
qc: ../QC_report.md

../QC_report.md: testset
	@echo "==> Task 1.8: QC report"
	$(PY) qc_report.py

clean:
	@echo "==> Cleaning $(PROCESSED)/ and ../QC_report.md"
	rm -rf $(PROCESSED)/*.csv $(PROCESSED)/*.gtf $(PROCESSED)/*.fa ../QC_report.md

# 個別実行例:
#   make fetch                                    # Step 1.1-1.4 のみ
#   make $(PROCESSED)/bricseq_halflife.csv        # BRIC-seq のみ
#   make clean && make all                        # 全やり直し
