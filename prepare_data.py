"""
Prepare QA datasets for Search-R1-TRL (BM25).

Usage:
    python prepare_data.py --dataset nq --local_dir ./data/nq_search
    python prepare_data.py --dataset hotpotqa --local_dir ./data/hotpotqa_search
    python prepare_data.py --dataset custom --data_file questions.jsonl --local_dir ./data/custom
"""
import argparse
import json
import os

import datasets


PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
    "Question: {question}\n"
)


def build_record(question, golden_answers, data_source, idx):
    question = question.strip()
    if not question.endswith("?"):
        question += "?"
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": PROMPT.format(question=question)}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": golden_answers}},
        "extra_info": {"split": "train", "index": idx},
    }


def extract_qa(raw_ds):
    qs, ans = [], []
    for item in raw_ds:
        q = item.get("question", "").strip()
        ga = item.get("golden_answers", item.get("answer", []))
        if isinstance(ga, str):
            ga = [ga]
        if isinstance(ga, list) and ga:
            qs.append(q)
            ans.append(ga)
    return qs, ans


def process(raw_train, raw_test, data_source):
    tq, ta = extract_qa(raw_train)
    eq, ea = extract_qa(raw_test)
    train = [build_record(q, a, data_source, i) for i, (q, a) in enumerate(zip(tq, ta))]
    test = [build_record(q, a, data_source, i) for i, (q, a) in enumerate(zip(eq, ea))]
    return train, test


LOADERS = {
    "nq": lambda: (
        datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")["train"],
        datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")["test"],
    ),
    "hotpotqa": lambda: (
        datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa")["train"],
        datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa")["dev"],
    ),
    "triviaqa": lambda: (
        datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "triviaqa")["train"],
        datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "triviaqa")["test"],
    ),
}


def load_custom(path):
    qs, ans = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            qs.append(item["question"].strip())
            a = item.get("answer", item.get("answers", ""))
            if isinstance(a, str):
                a = [a]
            ans.append(a)
    n = int(len(qs) * 0.9)
    return ({"question": qs[:n], "golden_answers": ans[:n]},
            {"question": qs[n:], "golden_answers": ans[n:]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["nq", "hotpotqa", "triviaqa", "custom"])
    p.add_argument("--data_file", help="JSONL path (for custom)")
    p.add_argument("--local_dir", default="./data/search_r1")
    p.add_argument("--max_samples", type=int)
    args = p.parse_args()

    if args.dataset == "custom":
        if not args.data_file:
            p.error("--data_file required for custom")
        raw_train, raw_test = load_custom(args.data_file)
        data_source = "custom"
    else:
        raw_train, raw_test = LOADERS[args.dataset]()
        data_source = args.dataset

    train, test = process(raw_train, raw_test, data_source)
    if args.max_samples:
        train, test = train[:args.max_samples], test[:args.max_samples]

    os.makedirs(args.local_dir, exist_ok=True)
    datasets.Dataset.from_list(train).to_parquet(os.path.join(args.local_dir, "train.parquet"))
    datasets.Dataset.from_list(test).to_parquet(os.path.join(args.local_dir, "test.parquet"))
    print(f"Saved: {args.local_dir}/train.parquet ({len(train)}), test.parquet ({len(test)})")


if __name__ == "__main__":
    main()
