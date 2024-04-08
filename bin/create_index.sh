# ============
# Process data
# ============


set -e


data_dir=$(pwd)/data


echo Creating HSNW index for AmazonCat-13k
poetry run python ./scripts/create_index.py --data_dir $data_dir --dataset amazoncat-13k --index_metric cosine --embedding embedding_all-mpnet-base-v2


echo Creating HSNW index for Agnews
poetry run python ./scripts/create_index.py --data_dir $data_dir --dataset agnews --index_metric cosine --embedding embedding_all-mpnet-base-v2


echo Creating HSNW index for WikiToxic
poetry run python ./scripts/create_index.py --data_dir $data_dir --dataset wikitoxic --index_metric cosine --embedding embedding_all-mpnet-base-v2
