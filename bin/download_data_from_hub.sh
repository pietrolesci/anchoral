# =============
# Download data
# =============


set -e


# saves data into the `./data/processed` folder directly
cwd=$(pwd)
data_dir=$cwd/data/processed
echo Creating files in $data_dir
mkdir -p $data_dir


echo Processing AmazonCat-13k
poetry run python ./scripts/download_data_from_hub.py --data_dir $data_dir --dataset pietrolesci/amazoncat-13k --embedding embedding_all-mpnet-base-v2

echo Processing Agnews
poetry run python ./scripts/download_data_from_hub.py --data_dir $data_dir --dataset pietrolesci/agnews --embedding embedding_all-mpnet-base-v2

echo Processing WikiToxic
poetry run python ./scripts/download_data_from_hub.py --data_dir $data_dir --dataset pietrolesci/wikitoxic --embedding embedding_all-mpnet-base-v2

echo Remove the `.data_cache` folder to free memory since shutil sometimes does not do the job
rm -rf .data_cache