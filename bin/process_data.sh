# ============
# Process data
# ============


set -e


data_dir=$(pwd)/data


# echo Processing AmazonCat-13k
# poetry run python ./scripts/process_data.py --data_dir $data_dir --dataset amazoncat-13k


echo Processing Pubmed-200k-RCT
poetry run python ./scripts/process_data.py --data_dir $data_dir --dataset pubmed-200k-rct


# echo Processing Eurlex-57k
# poetry run python ./scripts/process_data.py --data_dir $data_dir --dataset eurlex-57k
