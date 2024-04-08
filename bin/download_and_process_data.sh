# =============
# Download data
# =============


set -e


cwd=$(pwd)
data_dir=$cwd/data


echo Creating files in $data_dir


# saves data into the `./data/raw` folder
mkdir -p $data_dir/raw


# AGNEWS
echo Processing Agnews
poetry run python ./scripts/process_data.py --data_dir $data_dir --dataset ag_news


# WIKITOXIC
echo Processing WikiToxic
poetry run python ./scripts/process_data.py --data_dir $data_dir --dataset OxAISH-AL-LLM/wiki_toxic


# AMAZONCAT13K
echo Processing AmazonCat -- this might fail, consider running the steps manually
gdown 17rVRDarPwlMpb3l5zof9h34FlwbpTu4l; sleep 20  # since gdown is non-blocking
unzip AmazonCat-13K.raw.zip
rm -f AmazonCat-13K.raw.zip 
gzip -d AmazonCat-13K.raw/trn.json.gz
gzip -d AmazonCat-13K.raw/tst.json.gz
mv AmazonCat-13K.raw $data_dir/raw/amazoncat-13k
poetry run python ./scripts/process_data.py --data_dir $data_dir --dataset amazoncat-13k