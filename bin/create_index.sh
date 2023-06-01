# ============
# Process data
# ============


set -e


data_dir=$(pwd)/data


# echo Processing AmazonCat-13k
# poetry run python ./scripts/create_index.py --data_dir $data_dir --dataset amazoncat-13k --index_metric cosine


# echo Processing Pubmed-200k-RCT
# poetry run python ./scripts/create_index.py --data_dir $data_dir --dataset pubmed-200k-rct --index_metric cosine


# echo Processing Eurlex-57k
# poetry run python ./scripts/create_index.py --data_dir $data_dir --dataset eurlex-57k --index_metric cosine


# echo Processing Agnews
# poetry run python ./scripts/create_index.py --data_dir $data_dir --dataset agnews --index_metric cosine


echo Processing WikiToxic
poetry run python ./scripts/create_index.py --data_dir $data_dir --dataset wiki_toxic --index_metric l2
