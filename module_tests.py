import part31_superposition_and_saes.tests as tests
from . import submit
import torch as t

mid = 8
eid = 1

def tester():
    print("Test connection for module tests 8.1")

def test1(logits_without_sae_recon, logits_no_saes):
    t.testing.assert_close(logits_no_saes, logits_without_sae_recon)
    print("All tests passed!")

    submit.test_submit(1, mid=mid, eid=eid)

def test2(show_top_logits, gpt2, gpt2_sae):
    tests.test_show_top_logits(show_top_logits, gpt2, gpt2_sae)
    print("All tests passed!")
    submit.test_submit(2, mid=mid, eid=eid)

def test3(prompts):
    assert prompts["system"].startswith("We're studying neurons in a neural network.")
    assert "<< new>>" in prompts["user"]
    assert prompts["assistant"] == "this neuron fires on"

    print("All tests passed!")
    submit.test_submit(3, mid=mid, eid=eid)

#Checkpoints are not required for this module
def checkpoint1():
    """1 - Completed Attention SAEs subsubsection"""
    submit.test_submit(4, mid=mid, eid=eid, is_checked=False)

def checkpoint2():
    """2 - Completed GemmaScope and SAE Circuits subsubsection"""
    submit.test_submit(5, mid=mid, eid=eid, is_checked=False)

def checkpoint3():
    """3 - Completed Transcoders subsubsection"""
    submit.test_submit(6, mid=mid, eid=eid, is_checked=False)

def checkpoint4():
    """2 - Completed section 3"""
    submit.test_submit(7, mid=mid, eid=eid, is_checked=False)  

def checkpoint5():
    """Completed section 4"""
    submit.test_submit(8, mid=mid, eid=eid, is_checked=False)
