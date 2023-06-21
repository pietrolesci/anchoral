# =============
# Download data
# =============


set -e


cwd=$(pwd)
data_dir=$cwd/data


echo Creating files in $data_dir


mkdir -p $data_dir


poetry run python ./scripts/load_data_from_hub.py --data_dir $data_dir --dataset pietrolesci/agnews
poetry run python ./scripts/load_data_from_hub.py --data_dir $data_dir --dataset pietrolesci/pubmed-200k-rct
poetry run python ./scripts/load_data_from_hub.py --data_dir $data_dir --dataset pietrolesci/eurlex-57k
poetry run python ./scripts/load_data_from_hub.py --data_dir $data_dir --dataset pietrolesci/amazoncat-13k
poetry run python ./scripts/load_data_from_hub.py --data_dir $data_dir --dataset pietrolesci/wiki_toxic