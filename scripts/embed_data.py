from pathlib import Path
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import numpy as np
import srsly
import argparse
import hnswlib as hb
from tqdm.auto import tqdm

from src.enums import SpecialColumns



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2")
    args = parser.parse_args()
    pbar = tqdm(total=4)

    data_path = Path(args.data_path)

    # # load model
    # pbar.set_description(f"Loading embedding model {args.model_name}")
    # sentence_encoder = SentenceTransformer(args.model_name)
    # pbar.update(1)

    # load data and metadata
    pbar.set_description(f"Loading data and metadata from {data_path}")
    train_df = load_from_disk(data_path)["train"].to_pandas()
    meta = srsly.read_yaml(data_path / "metadata.yaml")
    pbar.update(1)

    # # compute embeddings
    # pbar.set_description(f"Embedding")
    # texts = train_df[SpecialColumns.TEXT].tolist()
    # embeddings = sentence_encoder.encode(texts, show_progress_bar=True, batch_size=512, device="cuda")
    
    # # save data
    # np.save(data_path / f"{data_path.name}_index.npy", embeddings, fix_imports=False)
    # pbar.update(1)
    
    embeddings = np.load(data_path / f"{data_path.name}_index.npy")

    # create hnsw index
    pbar.set_description(f"Creating HNSW index with seed {meta.get('seed', 1994)}")
    p = hb.Index(space="cosine", dim=embeddings.shape[1])
    p.set_ef(200)
    p.init_index(max_elements=embeddings.shape[0], M=64, ef_construction=200, random_seed=meta.get("seed", 1994))
    
    unique_ids = train_df[SpecialColumns.ID].values
    p.add_items(embeddings, unique_ids)
    
    # save index
    p.save_index(str(data_path / f"{data_path.name}_index.bin"))
    pbar.update(1)

    # update metadata
    meta = {
        **meta, 
        "embedding_model": args.model_name,
        "embedding_dim": embeddings.shape[1],
        "num_elements": embeddings.shape[0],
        "numpy_embeddings_path": str((data_path / f"{data_path.name}_index.npy").absolute()),
        "hnsw_index_path": str((data_path / f"{data_path.name}_index.bin").absolute()),
    }
    srsly.write_yaml(data_path / "metadata.yaml", meta)

