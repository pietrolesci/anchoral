# =============
# Download data
# =============


set -e


cwd=$(pwd)
data_dir=$cwd/data


echo Creating files in $data_dir


mkdir -p $data_dir


# EURLEX: get data from https://archive.org/download/EURLEX57K
# alternative: https://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K
wget https://archive.org/download/EURLEX57K/dataset.zip
unzip dataset.zip -d $data_dir/raw/eurlex-57k
rm -f dataset.zip
wget https://archive.org/download/EURLEX57K/eurovoc_concepts.jsonl -P $data_dir/raw/eurlex-57k


AMAZONCAT13K
gdown https://drive.google.com/u/0/uc?id=17rVRDarPwlMpb3l5zof9h34FlwbpTu4l&export=download; sleep 20  # since gdown is non-blocking
unzip AmazonCat-13K.raw.zip
rm -f AmazonCat-13K.raw.zip 
gzip -d AmazonCat-13K.raw/trn.json.gz
gzip -d AmazonCat-13K.raw/tst.json.gz
mv AmazonCat-13K.raw $data_dir/raw/amazoncat-13k


# PUBMED200k
git clone -n --depth=1 --filter=tree:0 https://github.com/Franck-Dernoncourt/pubmed-rct.git
cd pubmed-rct
git sparse-checkout set --no-cone PubMed_200k_RCT
git checkout
7z x PubMed_200k_RCT/train.7z
mv PubMed_200k_RCT/dev.txt ./; mv PubMed_200k_RCT/test.txt ./
rm -rf PubMed_200k_RCT/
cd $cwd
mv pubmed-rct $data_dir/raw/pubmed-200k-rct


# # AMAZON670K
# gdown https://drive.google.com/u/0/uc?id=16FIzX3TnlsqbrwSJJ2gDih69laezfZWR&export=download
# unzip Amazon670K.raw.zip
# rm -f Amazon670K.raw.zip
# gzip -d trn.raw.json.gz
# gzip -d trn.raw.json.gz
