# ============
# Process data
# ============


set -e


poetry run python ./scripts/create_data.py


# ################################
# # ========== AGNEWS ========== #
# ################################
# echo 'Processing AG-News'

# poetry run python ./scripts/process_agnews.py \
#     --output_dir='./data/processed/agnews' \
#     --seed=1994

# poetry run python ./scripts/embed_data.py \
#     --data_path='./data/processed/agnews' \
#     --model_name='all-mpnet-base-v2'


# ########################################
# # ========== Civil Comments ========== #
# ########################################
# echo 'Processing Civil Comments'

# poetry run python ./scripts/process_civil_comments.py \
#     --input_dir='./data/raw/civil_comments' \
#     --output_dir='./data/processed/civil_comments' \
#     --seed=1994 \
#     --min_chars=10 \
#     --train_samples=100_000 \
#     --test_samples=50_000

# poetry run python ./scripts/embed_data.py \
#     --data_path='./data/processed/civil_comments' \
#     --model_name='all-mpnet-base-v2'


# ##############################
# # ========== IMDB ========== #
# ##############################
# echo 'Processing IMDB'

# poetry run python ./scripts/process_imdb.py \
#     --output_dir='./data/processed/imdb' \
#     --seed=1994

# poetry run python ./scripts/embed_data.py \
#     --data_path='./data/processed/imdb' \
#     --model_name='all-mpnet-base-v2'