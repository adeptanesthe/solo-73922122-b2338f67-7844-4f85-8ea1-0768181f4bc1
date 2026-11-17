"""反例搜索：少量字符替换是否会翻转目标模型的下一 token 预测。

输入一段 1~32 个可打印 ASCII 字符的提示词，并在其中 2~5 个不同位置上各
指定 1~3 个「非原字符」替换候选；搜索穷尽所有「替换 0/1/2 处位置」的候选，
判定目标模型对下一 token 的贪心预测是否改变。

选择规则：
  1. 修改位置数最少优先（先穷尽 1 处替换，再穷尽 2 处）；
  2. 修改数相同时，替换后的完整提示词字典序最小优先。

KV 缓存策略（trie DFS）：
  - 所有候选与原提示词构成一棵共享前缀的 trie；DFS 沿树前进时只做单 token
    增量前向，回溯时把外部 KV 缓存截断回分叉点，兄弟分叉互不污染；
  - 每条搜索路径使用独立的 external_cache，模型自身的 _cache / cache_len /
    forward_count 完全不被触碰，因此与推测解码服务可安全并发共存。

正确性闸门：任何候选在发布前都要与「无缓存整句推理」的 argmax 比对；
基线或候选的计算/核对一旦失败，抛出 SearchError，绝不发布反例或稳定结论。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

from . import tokenizer
from .model import CausalTransformer, greedy_argmax

PROMPT_MIN_LEN = 1
PROMPT_MAX_LEN = 32
SPEC_MIN_POSITIONS = 2
SPEC_MAX_POSITIONS = 5
MAX_CHOICES_PER_POSITION = 3
MAX_REPLACEMENTS = 2
_PRINTABLE_ASCII = {chr(code) for code in range(0x20, 0x7F)}  # 95 个字符


class SearchError(Exception):
    """搜索参数非法或计算/核对失败；此时不发布任何结论。"""


@dataclass(frozen=True)
class EditSpec:
    """单个位置的替换规格：position 处可替换为 chars 中的任一字符。"""

    position: int
    chars: tuple[str, ...]


@dataclass
class SearchResult:
    found: bool
    baseline_prompt: str
    baseline_prediction: int
    candidate_prompt: str | None = None
    candidate_prediction: int | None = None
    changed_positions: list[int] = field(default_factory=list)
    replacement_chars: list[str] = field(default_factory=list)
    edit_count: int = 0
    candidates_checked: int = 0          # 实际完成核对的候选数（不含原句）
    cache_forwards: int = 0              # 搜索路径上的无状态增量前向次数
    full_verifications: int = 0          # 无缓存整句核对次数
    exhausted: bool = True               # 是否穷尽了允许范围
    error: str | None = None


def _validate(prompt: str, specs: list[EditSpec]) -> list[int]:
    if not isinstance(prompt, str) or not (
        PROMPT_MIN_LEN <= len(prompt) <= PROMPT_MAX_LEN
    ):
        raise SearchError(
            f"提示词长度须为 {PROMPT_MIN_LEN}~{PROMPT_MAX_LEN} 个字符。"
        )
    bad = [c for c in prompt if c not in _PRINTABLE_ASCII]
    if bad:
        raise SearchError("提示词只能包含可打印 ASCII 字符（0x20~0x7E）。")
    if not (SPEC_MIN_POSITIONS <= len(specs) <= SPEC_MAX_POSITIONS):
        raise SearchError(
            f"须指定 {SPEC_MIN_POSITIONS}~{SPEC_MAX_POSITIONS} 个不同替换位置。"
        )
    positions = [s.position for s in specs]
    if len(set(positions)) != len(positions):
        raise SearchError("替换位置必须互不相同。")
    for spec in specs:
        if not (0 <= spec.position < len(prompt)):
            raise SearchError(f"位置 {spec.position} 超出提示词范围。")
        if not (1 <= len(spec.chars) <= MAX_CHOICES_PER_POSITION):
            raise SearchError(
                f"位置 {spec.position} 须指定 1~{MAX_CHOICES_PER_POSITION} 个替换字符。"
            )
        original = prompt[spec.position]
        if len(set(spec.chars)) != len(spec.chars):
            raise SearchError(f"位置 {spec.position} 的替换字符存在重复。")
        for ch in spec.chars:
            if ch not in _PRINTABLE_ASCII:
                raise SearchError("替换字符只能是可打印 ASCII 字符。")
            if ch == original:
                raise SearchError(
                    f"位置 {spec.position} 的替换字符 {ch!r} 与原字符相同，"
                    "替换字符必须是非原字符。"
                )
    return tokenizer.encode(prompt)


def _full_sentence_prediction(model: CausalTransformer,
                              ids: list[int]) -> int:
    """无缓存整句推理的下一 token argmax（权威核对路径）。"""
    logits = model.forward(ids, use_cache=False, count_forward=False)
    return greedy_argmax(logits[-1])


def _generate_level(ids: list[int], specs: list[EditSpec],
                    level: int) -> list[tuple[list[int], list[int], list[str]]]:
    """生成恰好替换 level 处位置的全部候选。

    返回 (候选 token 序列, 替换位置列表, 替换字符列表) 的无序列表。
    候选数上界：C(5,2)*3^2 = 90，直接物化排序即可。
    """
    out: list[tuple[list[int], list[int], list[str]]] = []
    for combo in itertools.combinations(range(len(specs)), level):
        chosen = [specs[i] for i in combo]
        for chars in itertools.product(*(s.chars for s in chosen)):
            cand = list(ids)
            positions: list[int] = []
            repl: list[str] = []
            for spec, ch in zip(chosen, chars):
                cand[spec.position] = tokenizer.char_to_id(ch)
                positions.append(spec.position)
                repl.append(ch)
            out.append((cand, positions, repl))
    return out


def search_counterexample(target: CausalTransformer, prompt: str,
                          specs: list[EditSpec]) -> SearchResult:
    """穷尽允许候选，返回修改最少、同数字典序最小的反例；否则报告稳定。

    任何计算/核对失败都抛 SearchError，不发布结论。
    """
    ids = _validate(prompt, specs)
    # 按位置排序，便于 trie DFS 与结果解释
    specs = sorted(specs, key=lambda s: s.position)

    with torch.inference_mode():
        # ---- 1. 原句基线：缓存路径与无缓存整句必须给出同一 argmax ----
        base_cache = target.new_external_cache()
        base_cached_logits = target.stateless_prefill(ids, base_cache)
        base_cached = greedy_argmax(base_cached_logits)
        try:
            base_full = _full_sentence_prediction(target, ids)
        except RuntimeError as exc:  # 计算失败：不发布结论
            raise SearchError(f"基线整句推理失败：{exc}") from exc
        if base_cached != base_full:
            raise SearchError("基线 KV 缓存预测与无缓存整句推理不一致，搜索中止。")

        cache_forwards = 1
        full_verifications = 1
        candidates_checked = 0

        # ---- 2. 先 1 处替换、再 2 处替换；同级按完整候选字典序 ----
        for level in (1, MAX_REPLACEMENTS):
            level_candidates = _generate_level(ids, specs, level)
            # 完整提示词字典序（同级长度相同，等价于 token 序列字典序）
            level_candidates.sort(key=lambda item: tokenizer.decode(item[0]))

            # trie DFS：每个候选仅在「与上一候选的公共前缀」之外做增量前向，
            # 共享前缀复用同一份 external_cache，回溯截断保证分叉互不污染。
            # 缓存保留到公共前缀的倒数第二个 token（最后一个公共 token 重算，
            # 以便取得当前候选末端位置的 logits），KV 复用收益不受实质影响。
            shared_cache = target.new_external_cache()
            shared_len = 0
            prev_cand: list[int] | None = None

            for cand_ids, positions, repl in level_candidates:
                if prev_cand is None:
                    common = 0
                else:
                    common = 0
                    for a, b in zip(prev_cand, cand_ids):
                        if a != b:
                            break
                        common += 1
                # 回滚到公共前缀末 token 之前（分叉互不污染，末端 logits 重取）
                keep = max(common - 1, 0)
                if keep < shared_len:
                    target.truncate_external_cache(shared_cache, keep)
                    shared_len = keep
                # 增量推进差异后缀；保留最后一次增量前向的 logits
                last_logits: torch.Tensor | None = None
                while shared_len < len(cand_ids):
                    last_logits = target.stateless_next(
                        cand_ids[shared_len], shared_cache)
                    shared_len += 1
                    cache_forwards += 1

                cached_pred = greedy_argmax(last_logits)
                # 无缓存整句权威核对；不一致或失败都不发布结论
                try:
                    full_pred = _full_sentence_prediction(target, cand_ids)
                except RuntimeError as exc:
                    raise SearchError(
                        f"候选 {tokenizer.decode(cand_ids)!r} 整句推理失败：{exc}"
                    ) from exc
                full_verifications += 1
                if cached_pred != full_pred:
                    raise SearchError(
                        "候选 KV 缓存预测与无缓存整句推理不一致，搜索中止。"
                    )
                candidates_checked += 1
                prev_cand = list(cand_ids)

                if full_pred != base_full:
                    return SearchResult(
                        found=True,
                        baseline_prompt=prompt,
                        baseline_prediction=base_full,
                        candidate_prompt=tokenizer.decode(cand_ids),
                        candidate_prediction=full_pred,
                        changed_positions=positions,
                        replacement_chars=repl,
                        edit_count=level,
                        candidates_checked=candidates_checked,
                        cache_forwards=cache_forwards,
                        full_verifications=full_verifications,
                        exhausted=True,
                    )

        # ---- 3. 全部候选预测均未翻转：仅在完整核对后报告稳定 ----
        return SearchResult(
            found=False,
            baseline_prompt=prompt,
            baseline_prediction=base_full,
            candidates_checked=candidates_checked,
            cache_forwards=cache_forwards,
            full_verifications=full_verifications,
            exhausted=True,
        )
