import tests
import submit
import torch as t


def test1(attn_patterns_from_shorthand: t.Tensor, attn_patterns_from_full_name: t.Tensor):
    assert t.testing.assert_close(attn_patterns_from_shorthand, attn_patterns_from_full_name)
    submit.test_submit(1)
    print("All tests in `test1` passed!")


def test2(layer0_pattern_from_cache, layer0_pattern_from_q_and_k):
    assert t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
    submit.test_submit(2)
    print("All tests in `test1` passed!")

def test3(logit_attr, correct_token_logits): 

    assert t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    submit.test_submit(3)
    print("All tests in `test3` passed!")

def test4(ablation_scores, model, rep_tokens):
    assert tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)
    submit.test_submit(4)

def test5(AB_unfactored, AB):

    assert t.testing.assert_close(AB_unfactored, AB)
    submit.test_submit(5)
    print("All tests in `test5` passed!")

def test6(full_OV_circuit, model, layer, head_index):

    assert tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)
    submit.test_submit(6)

def test7(decomposed_qk_input, decomposed_q, decomposed_k, rep_cache, ind_head_index):

    assert t.testing.assert_close(
    decomposed_qk_input.sum(0), rep_cache["resid_pre", 1] + rep_cache["pos_embed"], rtol=0.01, atol=1e-05
)
    assert t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
    assert t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)
    print("All tests in `test7` passed!")
    submit.test_submit(7)

def test8(decompose_attn_scores, decomposed_q, decomposed_k, model):
    assert tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k, model)
    submit.test_submit(8)

def test9(find_K_comp_full_circuit, model):
    assert tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)
    submit.test_submit(9)

def test10(get_comp_score):

    assert tests.test_get_comp_score(get_comp_score)
    submit.test_submit(10)

def test11(composition_scores_batched, composition_scores):

    assert t.testing.assert_close(composition_scores_batched["Q"], composition_scores["Q"])
    assert t.testing.assert_close(composition_scores_batched["K"], composition_scores["K"])
    assert t.testing.assert_close(composition_scores_batched["V"], composition_scores["V"])

    submit.test_submit(11)
    print("All tests in `test11` passed!")