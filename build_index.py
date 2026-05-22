"""
Download corpus and build BM25 index using Pyserini (same as search-r1).
Requires Java JDK installed.

Mode 1: Download corpus from HuggingFace + build BM25 index
    python build_index.py --mode download --save_dir ./index

Mode 2: Build BM25 index from your own corpus
    python build_index.py --mode build --corpus_path ./data/wiki.jsonl --save_dir ./index
"""
import argparse
import os
import shutil
import subprocess

# Workaround: pyserini imports OpenAI client on startup
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("OPENAI_ORGANIZATION", "")


HF_CORPUS_REPO = "PeterJinGo/wiki-18-corpus"


def download_corpus(save_dir: str) -> str:
    """Download wiki-18 corpus from HuggingFace and return JSONL path."""
    import gzip
    from huggingface_hub import hf_hub_download

    os.makedirs(save_dir, exist_ok=True)

    print("[1/3] Downloading corpus from " + HF_CORPUS_REPO + " ...")
    hf_hub_download(
        repo_id=HF_CORPUS_REPO,
        filename="wiki-18.jsonl.gz",
        repo_type="dataset",
        local_dir=save_dir,
    )

    gz = os.path.join(save_dir, "wiki-18.jsonl.gz")
    jsonl = os.path.join(save_dir, "wiki-18.jsonl")

    if os.path.exists(gz):
        print("  Decompressing ...")
        with gzip.open(gz, "rb") as fin, open(jsonl, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        os.remove(gz)

    print(f"  Corpus -> {jsonl}")
    return jsonl


def build_bm25_index(corpus_path: str, save_dir: str) -> str:
    """Build BM25 index using Pyserini/Lucene."""
    bm25_dir = os.path.join(save_dir, "bm25")
    temp_dir = os.path.join(save_dir, "_temp")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(bm25_dir, exist_ok=True)

    # Copy corpus to temp dir
    temp_file = os.path.join(temp_dir, "temp.jsonl")
    shutil.copyfile(corpus_path, temp_file)

    print("[2/3] Building BM25 index with Pyserini ...")
    subprocess.run([
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", temp_dir,
        "--index", bm25_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8",
    ], check=True)

    shutil.rmtree(temp_dir)
    print(f"  BM25 index -> {bm25_dir}")
    return bm25_dir


def main():
    p = argparse.ArgumentParser(description="Download corpus + build BM25 index (Pyserini)")
    p.add_argument("--mode", required=True, choices=["download", "build"])
    p.add_argument("--save_dir", default="./index")
    p.add_argument("--corpus_path", help="Corpus JSONL path (build mode only)")
    args = p.parse_args()

    if args.mode == "download":
        jsonl = download_corpus(args.save_dir)
        index_path = build_bm25_index(jsonl, args.save_dir)
        corpus_path = jsonl
    else:
        if not args.corpus_path:
            p.error("--corpus_path required for build mode")
        index_path = build_bm25_index(args.corpus_path, args.save_dir)
        corpus_path = args.corpus_path

    print(f"[3/3] Done!")
    print(f"\nUsage:")
    print(f"  --bm25_index_path {index_path}")
    print(f"  --bm25_corpus_path {corpus_path}")


if __name__ == "__main__":
    main()
