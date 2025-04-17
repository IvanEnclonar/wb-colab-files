try:
    import part4_rlhf.tests as tests
    print("test is installed!")
except ImportError:
    print("Module is NOT installed.")
from . import submit
import torch as t
import torch.nn as nn
import math

mid = 9
eid = 1

def tester():
    required_tests = [
        "test_compute_advantages",
        "test_calc_kl_penalty",
        "test_calc_kl_penalty_stability",
        "test_calc_entropy_bonus",
        "test_calc_entropy_bonus_stability",
        "test_get_logprobs"
    ]
    
    missing_tests = []
    for test in required_tests:
        if not hasattr(tests, test):
            missing_tests.append(test)
    
    if missing_tests:
        print("Missing tests:", ", ".join(missing_tests))
    else:
        print("All tests are available.")


def test1(model, HookedTransformer, device):
    # Test your value head's architecture
    assert isinstance(model.base_model, HookedTransformer)
    assert isinstance(model.value_head, nn.Module)
    d_model = model.base_model.cfg.d_model
    n_params_expected = (d_model + 1) * 4 * d_model + (4 * d_model + 1)
    assert len(model.value_head) == 3, "Your value head should be a `nn.Sequential` with 3 layers."
    assert sum(p.numel() for p in model.value_head.parameters()) == n_params_expected, "Unexpected param count"

    # Test your class's forward pass
    batch_size, seq_len = 2, 10
    input_ids = t.randint(0, 1000, (batch_size, seq_len)).to(device)
    logits, values = model(input_ids)
    assert logits.shape == (batch_size, seq_len, model.base_model.cfg.d_vocab), "logits should be (batch, seq, d_vocab)"
    assert values.shape == (batch_size, seq_len), "value head output should be (batch, seq)"

    print("All tests for `TransformerWithValueHead` passed!")

    submit.test_submit(1, mid=mid, eid=eid, test_result=True)

def test2(reward_fn_char_count, device, A, B, C):
    t.testing.assert_close(reward_fn_char_count([A]), t.tensor([1.0], device=device))
    t.testing.assert_close(reward_fn_char_count([A, B, C]), t.tensor([1.0, 6.0, 0.0], device=device))
    t.testing.assert_close(reward_fn_char_count([A], " "), t.tensor([3.0], device=device))
    print("All tests for `reward_fn_char_count` passed!")
    submit.test_submit(2, mid=mid, eid=eid, test_result=True)

def test3(normalize_reward):
    reward = 10 + 5 * t.randn(10_000)
    reward_normalized = normalize_reward(reward)
    assert reward_normalized.mean().abs() < 1e-4
    assert (reward_normalized.std() - 1).abs() < 1e-4
    # Test edge case of zero reward
    reward = t.zeros(5)
    reward_normalized = normalize_reward(reward)
    assert reward_normalized.abs().sum() < 1e-4

    print("All tests for `normalize_reward` passed!")
    submit.test_submit(3, mid=mid, eid=eid, test_result=True)

def test4(compute_advantages):
    tests.test_compute_advantages(compute_advantages)
    print("All tests for `compute_advantages` passed!")
    submit.test_submit(4, mid=mid, eid=eid, test_result=True)


def test5(calc_kl_penalty):
    tests.test_calc_kl_penalty(calc_kl_penalty)
    tests.test_calc_kl_penalty_stability(calc_kl_penalty)

    print("All tests for `calc_kl_penalty` passed!")
    submit.test_submit(5, mid=mid, eid=eid, test_result=True)


def test6(calc_entropy_bonus):
    tests.test_calc_entropy_bonus(calc_entropy_bonus)
    tests.test_calc_entropy_bonus_stability(calc_entropy_bonus)

    print("All tests for `calc_entropy_bonus` passed!")
    submit.test_submit(6, mid=mid, eid=eid, test_result=True)


def test7(get_logprobs):
    tests.test_get_logprobs(get_logprobs)

    print("All tests for `get_logprobs` passed!")
    submit.test_submit(7, mid=mid, eid=eid, test_result=True)


def test8(optimizer, model, base_lr, head_lr):
    assert len(optimizer.param_groups) == 2, "Your optimizer should have two parameter groups."
    for param_group in optimizer.param_groups:
        assert param_group["maximize"], "Should be maximize=True."
        if len(param_group["params"]) <= 4:
            assert param_group["lr"] == head_lr, "LR for value head should be `head_lr`."
        else:
            assert param_group["lr"] == base_lr, "LR for base should be `base_lr`."

    total_params = sum(len(param_group["params"]) for param_group in optimizer.param_groups)
    assert total_params == len(
        list(model.parameters())
    ), "Your optimizer should have the same number of parameters as the model."

    print("All tests for `get_optimizer` passed!")

    submit.test_submit(8, mid=mid, eid=eid, test_result=True)

def test9():
    submit.test_submit(9, mid=mid, eid=eid, test_result=False)

def test10():
    submit.test_submit(10, mid=mid, eid=eid, test_result=False)

    