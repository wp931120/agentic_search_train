"""
Search-R1-TRL: Multi-turn GRPO training with BM25 retrieval.

Flow (same as search-r1):
  1. Model generates until <search> or <answer> tag
  2. If <search>, execute BM25 search, inject <information>...</information>
  3. Repeat until <answer> or max_turns
  4. EM reward on extracted answer
  5. GRPO policy update with KL penalty
"""
import argparse
import os
import random
import re
import string
from typing import List, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from tqdm import tqdm

from search_env import BM25Retriever, passages_to_string


# ===========================================================================
#  Reward
# ===========================================================================

def normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t):
        return " ".join(t.split())
    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))
    def lower(t):
        return t.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(pred: str, golds) -> int:
    if isinstance(golds, str):
        golds = [golds]
    pred = normalize_answer(pred)
    return any(normalize_answer(g) == pred for g in golds)


def extract_answer(text: str) -> str | None:
    """Extract answer from <answer>...</answer> tags in the response text."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def compute_reward(response_text: str, ground_truth, format_score=0.1) -> float:
    """Compute reward on the model's response text only (no prompt).

    - 1.0 if extracted answer matches ground truth (EM)
    - format_score (0.1) if model produced valid <answer>...</answer> but wrong content
    - Partial bonus for attempting correct tag format
    - 0.0 if no valid format found
    """
    answer = extract_answer(response_text)
    if random.randint(1, 64) == 1:
        targets = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
        print(f"--- reward sample ---\n  golden={targets}\n  answer={answer}\n  tail={response_text[-300:]}")

    if answer is None:
        # Exact format matches
        search_fmt = re.search(r"<search>.*?</search>", response_text, re.DOTALL)
        answer_fmt = re.search(r"<answer>.*?</answer>", response_text, re.DOTALL)
        # Lenient matches: model attempting the format but slightly off
        search_partial = re.search(r"<search\b.*?>.*?</search>", response_text, re.DOTALL)
        answer_partial = re.search(r"<answer\b.*?>.*?</answer>", response_text, re.DOTALL)

        bonus = 0.0
        if answer_fmt:
            bonus += 0.05
        elif answer_partial:
            bonus += 0.03
        if search_fmt:
            bonus += 0.05
        elif search_partial:
            bonus += 0.03
        return bonus

    targets = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
    if em_check(answer, targets):
        return 1.0
    return format_score


# ===========================================================================
#  Prompt
# ===========================================================================

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


# ===========================================================================
#  Dataset
# ===========================================================================

class SearchR1Dataset(Dataset):
    def __init__(self, data_path, tokenizer, max_prompt_length=4096):
        if data_path.endswith(".parquet"):
            self.ds = load_dataset("parquet", data_files=data_path, split="train")
        elif os.path.isdir(data_path):
            self.ds = load_dataset("parquet", data_files={
                "train": os.path.join(data_path, "train.parquet")
            }, split="train")
        else:
            self.ds = load_dataset("parquet", data_files=data_path, split="train")

        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.items = []

        for ex in self.ds:
            # The prepared parquet has 'prompt' field (list of chat messages)
            # but we need the raw question to reconstruct the prompt with our template
            # Try multiple possible field names for the question
            q = ex.get("question", "")
            if not q:
                # Fall back: extract from prompt field
                prompt_field = ex.get("prompt", "")
                if isinstance(prompt_field, list) and prompt_field:
                    # Extract question from the user message
                    content = prompt_field[0].get("content", "")
                    # The prompt ends with "Question: {question}\n"
                    m = re.search(r"Question:\s*(.+?)\n?$", content, re.DOTALL)
                    if m:
                        q = m.group(1).strip()

            if not q:
                continue  # skip invalid samples

            q = q.strip()
            if not q.endswith("?"):
                q += "?"

            user_content = (
                "Answer the given question. "
                "You must conduct reasoning inside <think> and </think> first every time you get new information. "
                "After reasoning, if you find you lack some knowledge, you can call a search engine by "
                "<search> query </search> and it will return the top searched results between <information> and </information>. "
                "You can search as many times as your want. "
                "If you find no further external knowledge needed, you can directly provide the answer inside "
                "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
                f"\nQuestion: {q}"
            )
            messages = [{"role": "user", "content": user_content}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = self.tokenizer(
                prompt_text, truncation=True, max_length=max_prompt_length,
                add_special_tokens=False, return_tensors="pt",
            )["input_ids"][0]

            gt = ex.get("reward_model", {}).get("ground_truth", {})
            if isinstance(gt, str):
                gt = {"target": [gt]}
            elif isinstance(gt, dict):
                t = gt.get("target", [])
                if isinstance(t, str):
                    t = [t]
                gt = {"target": t}
            else:
                gt = {"target": []}

            self.items.append({
                "prompt_ids": prompt_tokens,
                "prompt_text": prompt_text,
                "ground_truth": gt,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    ids_list = [x["prompt_ids"] for x in batch]
    lengths = [len(x) for x in ids_list]
    max_len = max(lengths)
    padded_ids = []
    attn_mask = []
    for t in ids_list:
        p = max_len - len(t)
        padded_ids.append(F.pad(t, (p, 0), value=0))
        attn_mask.append(F.pad(torch.ones(len(t)), (p, 0), value=0))
    return {
        "prompt_ids": torch.stack(padded_ids),
        "prompt_lengths": lengths,
        "attention_mask": torch.stack(attn_mask),
        "ground_truth": [x["ground_truth"] for x in batch],
    }


# ===========================================================================
#  Multi-turn generation
# ===========================================================================

def _find_action(text: str) -> Tuple[str | None, str | None]:
    search_m = re.search(r'<search>(.*?)</search>', text, re.DOTALL)
    answer_m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if search_m and answer_m:
        if search_m.start() <= answer_m.start():
            return "search", search_m.group(1).strip()
        return "answer", answer_m.group(1).strip()
    if search_m:
        return "search", search_m.group(1).strip()
    if answer_m:
        return "answer", answer_m.group(1).strip()
    return None, None


def _truncate_at_tag(text: str) -> str:
    for tag in ["</search>", "</answer>"]:
        pos = text.find(tag)
        if pos >= 0:
            return text[:pos + len(tag)]
    return text


def run_multi_turn(model, tokenizer, prompt_ids, retriever: BM25Retriever,
                   max_turns=2, max_response_length=500, max_obs_length=500,
                   temperature=1.0) -> Tuple[str, str]:
    """Returns (full_text, response_text).

    full_text: complete context with special tokens (for log-prob computation).
    response_text: model generation + observations only (for reward computation).
    """
    device = model.device
    context_ids = prompt_ids.unsqueeze(0).to(device)
    prompt_len = len(context_ids[0])
    response_parts = []

    for turn in range(max_turns):
        outputs = model.generate(
            input_ids=context_ids, attention_mask=torch.ones_like(context_ids),
            max_new_tokens=max_response_length, do_sample=True,
            temperature=temperature, top_p=0.9, top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

        new_ids = outputs[0, prompt_len:]
        new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        action, content = _find_action(new_text)

        if action == "answer":
            truncated = _truncate_at_tag(new_text)
            response_parts.append(truncated)
            trunc_ids = tokenizer.encode(truncated, add_special_tokens=False,
                                         return_tensors="pt").to(device)
            context_ids = torch.cat([context_ids, trunc_ids], dim=1)
            break
        elif action == "search":
            truncated = _truncate_at_tag(new_text)
            response_parts.append(truncated)
            results, _ = retriever.batch_search([content])
            if results and results[0]:
                obs_text = passages_to_string(results[0]).strip()
            else:
                obs_text = "No relevant information found."
            obs = f"\n\n<information>{obs_text}</information>\n\n"
            obs_ids = tokenizer.encode(obs, add_special_tokens=False)
            if len(obs_ids) > max_obs_length:
                obs_ids = obs_ids[:max_obs_length]
                obs = tokenizer.decode(obs_ids, skip_special_tokens=True)
            response_parts.append(obs)
            suffix_text = truncated + obs
            suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False,
                                          return_tensors="pt").to(device)
            context_ids = torch.cat([context_ids, suffix_ids], dim=1)
            prompt_len = context_ids.shape[1]
        else:
            hint = (
                "\n\nMy previous action is invalid. "
                "If I want to search, I should put the query between <search> and </search>. "
                "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
                "Let me try again.\n\n"
            )
            response_parts.append(new_text)
            suffix_text = new_text + hint
            suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False,
                                          return_tensors="pt").to(device)
            context_ids = torch.cat([context_ids, suffix_ids], dim=1)
            prompt_len = context_ids.shape[1]
    else:
        outputs = model.generate(
            input_ids=context_ids, attention_mask=torch.ones_like(context_ids),
            max_new_tokens=max_response_length, do_sample=True,
            temperature=temperature, top_p=0.9, top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
        final_text = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)
        response_parts.append(final_text)
        context_ids = torch.cat([
            context_ids,
            tokenizer.encode(final_text, add_special_tokens=False,
                             return_tensors="pt").to(device)
        ], dim=1)

    full_text = tokenizer.decode(context_ids[0], skip_special_tokens=False)
    response_text = "".join(response_parts)
    return full_text, response_text


# ===========================================================================
#  GRPO loss (only on response tokens, not prompt)
# ===========================================================================

def compute_token_log_probs(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    shift_labels = input_ids[:, 1:].unsqueeze(-1)
    return torch.gather(log_probs[:, :-1, :], -1, shift_labels).squeeze(-1)


def build_response_mask(tokenizer, all_texts, prompt_lengths, device) -> torch.Tensor:
    """Build a mask that is 1 for response tokens and 0 for prompt tokens."""
    enc = tokenizer(all_texts, padding=True, return_tensors="pt", add_special_tokens=False)
    B, T = enc["input_ids"].shape

    mask = torch.zeros(B, T, dtype=torch.float32)
    for i, (text, prompt_len) in enumerate(zip(all_texts, prompt_lengths)):
        # Find where the prompt ends in the full text
        full_ids = enc["input_ids"][i]
        # The prompt was tokenized separately, so we need to find the boundary
        # Approximate: find the first position where full_ids matches the prompt pattern
        # Simple approach: response starts after the prompt length
        mask[i, prompt_len:] = 1.0

    return mask


def grpo_loss(model, ref_model, all_texts, prompt_lengths, advantages,
              tokenizer, kl_coef, device):
    enc = tokenizer(all_texts, padding=True, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    B, T = input_ids.shape

    cur_lp = compute_token_log_probs(model, input_ids, attn)       # (B, T-1)
    with torch.no_grad():
        ref_lp = compute_token_log_probs(ref_model, input_ids, attn)

    kl = cur_lp - ref_lp.detach()
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    adv_tok = adv_t.unsqueeze(1).expand(B, T - 1)

    # Response mask: only compute loss on response tokens (not prompt)
    resp_mask = torch.zeros(B, T - 1, dtype=torch.float32, device=device)
    for i, plen in enumerate(prompt_lengths):
        # Response starts after prompt length (capped at T-1)
        start = min(plen, T - 1)
        resp_mask[i, start:] = 1.0

    # Apply both response mask and attention mask
    effective_mask = resp_mask * attn[:, 1:]

    per_token_loss = -adv_tok * cur_lp + kl_coef * kl
    loss = (per_token_loss * effective_mask).sum() / (effective_mask.sum() + 1e-8)

    with torch.no_grad():
        kl_val = (kl * effective_mask).sum() / (effective_mask.sum() + 1e-8)

    return loss, kl_val.item()


# ===========================================================================
#  Training
# ===========================================================================

def train(args):
    set_seed(args.seed)

    print(f"[1/5] Loading model: {args.model_name_or_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation=args.attn_implementation,
    )
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation=args.attn_implementation,
    )
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Verify model is on GPU
    print(f"  Model device: {next(model.parameters()).device}")
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({mem:.1f} GB)")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/5] Loading BM25 retriever: {args.bm25_index_path}")
    retriever = BM25Retriever(
        index_path=args.bm25_index_path,
        corpus_path=args.bm25_corpus_path,
        topk=args.retriever_topk,
    )

    print(f"[3/5] Loading dataset: {args.data_path}")
    dataset = SearchR1Dataset(args.data_path, tokenizer, args.max_prompt_length)
    if args.max_samples:
        dataset.items = dataset.items[:args.max_samples]
    print(f"  Dataset size: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(loader) * args.num_epochs // args.gradient_accumulation_steps
    warmup = int(total_steps * args.lr_warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    print(f"[4/5] Training  epochs={args.num_epochs}  steps_per_epoch={len(loader)}  "
          f"generations={args.num_generations}  lr={args.learning_rate}")

    global_step = 0
    model.zero_grad()

    for epoch in range(args.num_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        epoch_rewards, epoch_kls = [], []

        for bi, batch in enumerate(pbar):
            prompt_ids = batch["prompt_ids"]
            prompt_lengths = batch["prompt_lengths"]
            gts = batch["ground_truth"]
            B = prompt_ids.shape[0]

            all_texts, batch_rewards, batch_prompt_lens = [], [], []
            model.eval()
            with torch.no_grad():
                for i in range(B):
                    for g in range(args.num_generations):
                        full_text, response_text = run_multi_turn(
                            model, tokenizer, prompt_ids[i], retriever,
                            max_turns=args.max_turns,
                            max_response_length=args.max_response_length,
                            max_obs_length=args.max_obs_length,
                            temperature=args.temperature,
                        )
                        all_texts.append(full_text)
                        batch_rewards.append(compute_reward(response_text, gts[i]))
                        batch_prompt_lens.append(prompt_lengths[i])

                        if i == 0 and g == 0:
                            print(f"\n{'='*60}")
                            print(f"[SAMPLE] Step {bi}")
                            gt_targets = gts[i].get("target", [])
                            print(f"[GOLD]   {gt_targets}")
                            print(f"[GEN]    ...{response_text[-500:]}")
                            print(f"[REWARD] {batch_rewards[-1]}")
                            print(f"{'='*60}")
            model.train()

            # GRPO advantages
            advantages = []
            for g_start in range(0, len(batch_rewards), args.num_generations):
                grp = batch_rewards[g_start: g_start + args.num_generations]
                mean_r = sum(grp) / len(grp)
                std_r = (sum((r - mean_r) ** 2 for r in grp) / len(grp)) ** 0.5
                advantages.extend((r - mean_r) / (std_r + 1e-8) for r in grp)

            loss, kl = grpo_loss(
                model, ref_model, all_texts, batch_prompt_lens, advantages,
                tokenizer, args.kl_loss_coef, model.device,
            )
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (bi + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_rewards.extend(batch_rewards)
            epoch_kls.append(kl)
            pbar.set_postfix({
                "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                "reward": f"{sum(batch_rewards)/len(batch_rewards):.3f}",
                "kl": f"{kl:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if global_step > 0 and global_step % args.save_steps == 0:
                d = os.path.join(args.output_dir, f"ckpt-{global_step}")
                os.makedirs(d, exist_ok=True)
                model.save_pretrained(d)
                tokenizer.save_pretrained(d)
                print(f"\n  Saved -> {d}")

        avg_r = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        avg_kl = sum(epoch_kls) / max(len(epoch_kls), 1)
        print(f"\n  Epoch {epoch+1} done | avg_reward={avg_r:.4f} | avg_kl={avg_kl:.4f}")

    final = os.path.join(args.output_dir, "final")
    os.makedirs(final, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"\nDone. Final model -> {final}")


def parse_args():
    p = argparse.ArgumentParser(description="Search-R1 GRPO with BM25")
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--attn_implementation", type=str, default="sdpa",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Data
    p.add_argument("--data_path", type=str, required=True)

    # BM25
    p.add_argument("--bm25_index_path", type=str, required=True)
    p.add_argument("--bm25_corpus_path", type=str, required=True)
    p.add_argument("--retriever_topk", type=int, default=3)

    # Training
    p.add_argument("--output_dir", type=str, default="output/search-r1-trl-bm25")
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--lr_warmup_ratio", type=float, default=0.1)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_prompt_length", type=int, default=4096)
    p.add_argument("--max_response_length", type=int, default=500)
    p.add_argument("--max_obs_length", type=int, default=500)
    p.add_argument("--max_turns", type=int, default=2)
    p.add_argument("--num_generations", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kl_loss_coef", type=float, default=0.001)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit training to first N samples")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
