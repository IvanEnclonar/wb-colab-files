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

    submit.test_submit(12, mid=mid, eid=eid)

def test2(show_top_logits, gpt2, gpt2_sae):
    tests.test_show_top_logits(show_top_logits, gpt2, gpt2_sae)
    print("All tests passed!")
    submit.test_submit(13, mid=mid, eid=eid)

def test3(prompts):
    assert prompts["system"].startswith("We're studying neurons in a neural network.")
    assert "<< new>>" in prompts["user"]
    assert prompts["assistant"] == "this neuron fires on"

    print("All tests passed!")
    submit.test_submit(14, mid=mid, eid=eid)

#Checkpoints are not required for this module
def checkpoint1():
    """Completed Attention SAEs subsubsection"""
    submit.test_submit(15, mid=mid, eid=eid, is_checked=False)

def checkpoint2():
    """Completed latents for features subsubsection"""
    submit.test_submit(16, mid=mid, eid=eid, is_checked=False)

def checkpoint3():
    """Completed GEMMASCOPE subsubsection"""
    submit.test_submit(17, mid=mid, eid=eid, is_checked=False)

def checkpoint4():
    """Completed Section 1 Bonus"""
    submit.test_submit(18, mid=mid, eid=eid, is_checked=False)

def checkpoint5():
    """Completed Section 2"""
    submit.test_submit(19, mid=mid, eid=eid, is_checked=False)

def checkpoint6():
    """Completed Section 3"""
    submit.test_submit(20, mid=mid, eid=eid, is_checked=False)

def checkpoint7():
    """Reached the end of the module"""
    submit.test_submit(21, mid=mid, eid=eid, is_checked=False)
