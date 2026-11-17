"""反例搜索功能测试：

  1. 搜索使用的外部 KV 缓存与无缓存整句推理 argmax 完全一致（含分叉回滚）；
  2. 穷尽性与选择规则：修改数最少优先，同级完整提示词字典序最小优先
     （与独立暴力枚举逐候选核对）；
  3. 稳定结论只在全部候选核对完成后给出；
  4. 失败闸门：计算失败或缓存/整句不一致时抛 SearchError，不发布任何结论；
  5. 隔离性：搜索不改变模型权重、内部 KV 缓存与 forward_count，
     且搜索前后的推测解码输出与前向次数完全一致；
  6. 参数校验：长度、位置数、替换字符数、非原字符等约束；
  7. HTTP 端点行为（FastAPI TestClient）。
"""

from __future__ import annotations

import itertools
import os
import random
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.search as search_mod
from app import tokenizer
from app.model import get_models, greedy_argmax
from app.search import (
    MAX_REPLACEMENTS,
    EditSpec,
    SearchError,
    search_counterexample,
)

torch.set_num_threads(2)
DRAFT, TARGET = get_models()
_ALPHABET = [chr(c) for c in range(0x20, 0x7F)]


def _full_pred(text: str) -> int:
    return greedy_argmax(
        TARGET.forward(tokenizer.encode(text), use_cache=False,
                       count_forward=False)[-1]
    )


def test_external_cache_matches_full_sentence_with_branching() -> None:
    """trie 式分叉：共享前缀增量前向 + 截断回滚，预测必须与整句推理一致。"""
    ids = tokenizer.encode("the quick brown fox")
    cache = TARGET.new_external_cache()
    # 推进到分叉点
    TARGET.stateless_prefill(ids[:8], cache)
    assert cache[0][0].shape[2] == 8

    branch_a = ids[8:]
    for t in branch_a:
        logits_a = TARGET.stateless_next(t, cache)
    full_a = TARGET.forward(ids, use_cache=False, count_forward=False)[-1]
    assert greedy_argmax(logits_a) == greedy_argmax(full_a)

    # 回滚到分叉点，换一条续写（兄弟分叉互不污染）
    TARGET.truncate_external_cache(cache, 8)
    other = tokenizer.encode("ZZZZZZZ")[:len(ids) - 8]
    for t in other:
        logits_b = TARGET.stateless_next(t, cache)
    full_b = TARGET.forward(ids[:8] + other, use_cache=False,
                            count_forward=False)[-1]
    assert greedy_argmax(logits_b) == greedy_argmax(full_b)

    # 回滚后再走第一条分支，结果仍与整句一致（无污染）
    TARGET.truncate_external_cache(cache, 8)
    for t in branch_a:
        logits_a2 = TARGET.stateless_next(t, cache)
    assert greedy_argmax(logits_a2) == greedy_argmax(full_a)
    print("ok: 外部缓存分叉/回滚与无缓存整句推理逐位一致，分叉互不污染")


def test_search_matches_bruteforce_and_selection_rule() -> None:
    """多组随机场景下，搜索结果与独立暴力枚举的最优反例完全一致。"""
    rng = random.Random(20261117)
    checked_cases = 0
    for _ in range(40):
        n = rng.randint(3, 12)
        p = "".join(rng.choice(_ALPHABET) for _ in range(n))
        n_specs = rng.randint(2, min(5, n))
        positions = rng.sample(range(n), n_specs)
        specs = []
        for pos in positions:
            choices = rng.sample([c for c in _ALPHABET if c != p[pos]],
                                 rng.randint(1, 3))
            specs.append(EditSpec(pos, tuple(choices)))
        base = _full_pred(p)

        def brute(level: int, specs=specs, p=p, base=base) -> list[str]:
            flips = []
            for combo in itertools.combinations(specs, level):
                for chars in itertools.product(*(s.chars for s in combo)):
                    cand = list(p)
                    for s, ch in zip(combo, chars):
                        cand[s.position] = ch
                    text = "".join(cand)
                    if _full_pred(text) != base:
                        flips.append(text)
            return sorted(flips)

        flips1 = brute(1)
        flips2 = brute(MAX_REPLACEMENTS)
        r = search_counterexample(TARGET, p, specs)

        if not (flips1 or flips2):
            assert r.found is False
            expected_total = (
                sum(len(s.chars) for s in specs)
                + sum(
                    len(a.chars) * len(b.chars)
                    for a, b in itertools.combinations(specs, 2)
                )
            )
            assert r.candidates_checked == expected_total
        else:
            expected_text = flips1[0] if flips1 else flips2[0]
            expected_level = 1 if flips1 else 2
            assert r.found
            assert r.edit_count == expected_level
            assert r.candidate_prompt == expected_text
            assert r.baseline_prediction == base
            assert r.candidate_prediction == _full_pred(expected_text)
        checked_cases += 1
    print(f"ok: {checked_cases} 组随机场景下搜索与暴力枚举结论及选择规则完全一致")


def test_stable_only_after_full_verification() -> None:
    """稳定结论必须建立在全部候选核对完成之上（候选数统计正确）。"""
    p = "abcdefgh"
    base = _full_pred(p)
    safe_specs = []
    for pos in (1, 3, 5):
        safe = []
        for ch in _ALPHABET:
            if ch == p[pos]:
                continue
            cand = list(p)
            cand[pos] = ch
            if _full_pred("".join(cand)) == base:
                safe.append(ch)
            if len(safe) == 3:
                break
        assert len(safe) == 3
        safe_specs.append(EditSpec(pos, tuple(safe)))
    r = search_counterexample(TARGET, p, safe_specs)
    assert r.found is False
    # 一级 3*3=9，二级 C(3,2)*3*3=27，合计 36
    assert r.candidates_checked == 9 + 27
    assert r.full_verifications == 1 + r.candidates_checked
    print("ok: 稳定结论仅在全部候选无缓存整句核对完成后给出")


def test_failure_gates_publish_nothing() -> None:
    """计算失败与缓存/整句不一致都必须中止搜索且不发布结论。"""
    prompt = "abcdef"
    specs = [EditSpec(0, ("z", "!")), EditSpec(2, ("q",))]
    base_ids = tokenizer.encode(prompt)
    original = search_mod._full_sentence_prediction

    def raising(model, ids):
        raise RuntimeError("simulated failure")

    def tamper_candidate(model, ids):
        gid = original(model, ids)
        return (gid + 1) % tokenizer.VOCAB_SIZE if ids != base_ids else gid

    def tamper_baseline(model, ids):
        gid = original(model, ids)
        return (gid + 1) % tokenizer.VOCAB_SIZE if ids == base_ids else gid

    for fn in (raising, tamper_candidate, tamper_baseline):
        search_mod._full_sentence_prediction = fn
        try:
            search_counterexample(TARGET, prompt, specs)
            raise AssertionError("应当抛出 SearchError")
        except SearchError:
            pass
        finally:
            search_mod._full_sentence_prediction = original
    print("ok: 计算失败/核对不一致时均不发布反例或稳定结论")


def test_search_does_not_touch_model_state_or_decode() -> None:
    """搜索与原解码共存：权重、内部 KV、forward_count 与解码结果均不受影响。"""
    from app.decoding import compare

    ids = tokenizer.encode("the quick brown fox jumps over ")
    r1 = compare(DRAFT, TARGET, ids, 3, 16)
    assert r1.match
    target_snap = [(k.clone(), v.clone()) for k, v in TARGET._cache]
    draft_snap = [(k.clone(), v.clone()) for k, v in DRAFT._cache]
    weight_snap = [p.clone() for p in TARGET.parameters()]
    t_fwd, d_fwd = TARGET.forward_count, DRAFT.forward_count

    search_counterexample(TARGET, "the quick brown", [
        EditSpec(0, ("a", "z")),
        EditSpec(4, ("Q", "x")),
        EditSpec(10, ("z", "!")),
    ])

    assert TARGET.forward_count == t_fwd and DRAFT.forward_count == d_fwd
    for wb, wa in zip(weight_snap, TARGET.parameters()):
        assert torch.equal(wb, wa)
    for (kb, vb), (ka, va) in zip(target_snap, TARGET._cache):
        assert torch.equal(kb, ka) and torch.equal(vb, va)
    for (kb, vb), (ka, va) in zip(draft_snap, DRAFT._cache):
        assert torch.equal(kb, ka) and torch.equal(vb, va)

    r2 = compare(DRAFT, TARGET, ids, 3, 16)
    assert r2.match
    assert r2.speculative.new_tokens == r1.speculative.new_tokens
    assert (r2.speculative.target_forward_count
            == r1.speculative.target_forward_count)
    assert (r2.speculative.draft_forward_count
            == r1.speculative.draft_forward_count)
    print("ok: 搜索不影响模型权重、内部 KV、前向计数与原解码结果")


def test_input_validation() -> None:
    good = [EditSpec(0, ("z", "!")), EditSpec(2, ("q",))]
    bad_cases = [
        ("", good, "长度"),
        ("a" * 33, good, "长度"),
        ("ab", [EditSpec(0, ("z",))], "替换位置"),                 # 位置数不足
        ("ab", [EditSpec(0, ("z",)), EditSpec(0, ("q",))], "互不相同"),
        ("ab", [EditSpec(0, ("a",)), EditSpec(1, ("z",))], "非原字符"),
        ("ab", [EditSpec(0, ("z", "z")), EditSpec(1, ("q",))], "重复"),
        ("ab", [EditSpec(0, (), ), EditSpec(1, ("q",))], "1~3"),
        ("ab", [EditSpec(0, ("z",)), EditSpec(5, ("q",))], "超出"),
    ]
    for prompt, specs, needle in bad_cases:
        try:
            search_counterexample(TARGET, prompt, specs)
            raise AssertionError(f"应拒绝: {prompt!r} {specs}")
        except SearchError as exc:
            assert needle in str(exc), (needle, str(exc))
    print("ok: 参数越界/非原字符/重复等输入全部被拒绝")


def test_http_endpoint() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    body = {
        "prompt": "the quick brown",
        "edits": [
            {"position": 0, "chars": ["a", "z"]},
            {"position": 4, "chars": ["Q", "x"]},
            {"position": 10, "chars": ["z", "!"]},
        ],
    }
    resp = client.post("/api/counterexample", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["exhausted"] is True
    assert "baseline" in data and "stats" in data
    if data["found"]:
        ce = data["counterexample"]
        assert ce["prediction"]["id"] != data["baseline"]["prediction"]["id"]
        assert 1 <= ce["edit_count"] <= 2

    # 非法入参：同字符替换 -> 400，且不携带结论
    bad = dict(body)
    bad["edits"] = [
        {"position": 0, "chars": ["t"]},  # 与原字符相同
        {"position": 4, "chars": ["Q"]},
    ]
    r2 = client.post("/api/counterexample", json=bad)
    assert r2.status_code == 400
    assert "error" in r2.json() and "found" not in r2.json()

    # pydantic 层越界 -> 422
    r3 = client.post("/api/counterexample",
                     json={"prompt": "ab", "edits": [{"position": 0, "chars": ["z"]}]})
    assert r3.status_code == 422

    # 健康检查不受影响
    assert client.get("/api/health").status_code == 200
    print("ok: HTTP 端点正常发布结论/拒绝非法请求，健康检查可用")


if __name__ == "__main__":
    test_external_cache_matches_full_sentence_with_branching()
    test_search_matches_bruteforce_and_selection_rule()
    test_stable_only_after_full_verification()
    test_failure_gates_publish_nothing()
    test_search_does_not_touch_model_state_or_decode()
    test_input_validation()
    test_http_endpoint()
    print("\n反例搜索全部测试通过 ✔")
