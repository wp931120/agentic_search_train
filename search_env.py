"""
BM25 retriever using Pyserini (same as search-r1).
Requires Java JDK and pyserini installed.
"""
import json
import os
import warnings
from typing import List, Tuple

# Workaround: pyserini imports OpenAI client on startup
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("OPENAI_ORGANIZATION", "")

import datasets
from pyserini.search.lucene import LuceneSearcher


def load_corpus(corpus_path: str):
    return datasets.load_dataset(
        "json", data_files=corpus_path, split="train", num_proc=4
    )


def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]


def passages_to_string(retrieval_result) -> str:
    """Format retrieval results into a string like search-r1."""
    text = ""
    for idx, doc in enumerate(retrieval_result):
        content = doc.get("contents", "")
        title = content.split("\n")[0]
        body = "\n".join(content.split("\n")[1:])
        text += f"Doc {idx + 1}(Title: {title}) {body}\n"
    return text


class BM25Retriever:
    def __init__(self, index_path: str, corpus_path: str = None, topk: int = 3):
        self.searcher = LuceneSearcher(index_path)
        self.topk = topk
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            assert corpus_path is not None, "corpus_path required when index has no docs"
            self.corpus = load_corpus(corpus_path)
        else:
            self.corpus = None

    def _check_contain_doc(self):
        try:
            return self.searcher.doc(0).raw() is not None
        except Exception:
            return False

    def _search(self, query: str, num: int = None) -> Tuple[list, list]:
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if not hits:
            return [], []
        scores = [h.score for h in hits]
        if len(hits) > num:
            hits = hits[:num]

        if self.contain_doc:
            contents_list = [
                json.loads(self.searcher.doc(h.docid).raw())["contents"] for h in hits
            ]
            results = [
                {
                    "title": c.split("\n")[0].strip('"'),
                    "text": "\n".join(c.split("\n")[1:]),
                    "contents": c,
                }
                for c in contents_list
            ]
        else:
            results = load_docs(self.corpus, [h.docid for h in hits])

        return results, scores

    def batch_search(self, queries: List[str], num: int = None) -> Tuple[list, list]:
        results, scores = [], []
        for q in queries:
            r, s = self._search(q, num)
            results.append(r)
            scores.append(s)
        return results, scores
