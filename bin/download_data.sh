# =============
# Download data
# =============


set -e
mkdir -p data/raw
cd data/raw


# EURLEX: get data from https://archive.org/download/EURLEX57K
alternative: https://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K
wget https://archive.org/download/EURLEX57K/dataset.zip
unzip dataset.zip -d eurlex
rm -f dataset.zip
wget https://archive.org/download/EURLEX57K/eurovoc_concepts.jsonl -P eurlex
poetry run python ./scripts/eurlex_process.py


# PUBMED200k
git clone -n --depth=1 --filter=tree:0 https://github.com/Franck-Dernoncourt/pubmed-rct.git
cd pubmed-rct
git sparse-checkout set --no-cone PubMed_200k_RCT
git checkout
7z x PubMed_200k_RCT/train.7z
mv PubMed_200k_RCT/dev.txt ./
mv PubMed_200k_RCT/test.txt ./
rm -rf PubMed_200k_RCT/
poetry run python ./scripts/pubmed_process.py