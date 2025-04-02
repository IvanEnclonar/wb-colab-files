import tests
from . import submit
import torch as t

mid = 8
eid = 1

def tester():
    print("Test connection for module tests 8.1")

def test1(ToyModel):
    tests.test_model(ToyModel)
    submit.test_submit(1, mid=mid, eid=eid)

def test2(ToyModel):
    tests.test_generate_batch(ToyModel)
    submit.test_submit(2, mid=mid, eid=eid)

def test3(ToyModel):
    tests.test_calculate_loss(ToyModel)
    submit.test_submit(3, mid=mid, eid=eid)

def test4(ToySAE):
    tests.test_sae_init(ToySAE)
    submit.test_submit(4, mid=mid, eid=eid)

def test5(ToySAE):
    tests.test_sae_W_dec_normalized(ToySAE)
    submit.test_submit(5, mid=mid, eid=eid)

def test6(ToySAE):
    tests.test_sae_generate_batch(ToySAE)
    submit.test_submit(6, mid=mid, eid=eid)

def test7(ToySAE):
    tests.test_sae_forward(ToySAE)
    submit.test_submit(7, mid=mid, eid=eid)

def test8(ToySAE):
    tests.test_resample_simple(ToySAE) 
    submit.test_submit(8, mid=mid, eid=eid)

def test9(ToySAE):
    tests.test_resample_advanced(ToySAE)
    submit.test_submit(9, mid=mid, eid=eid)

def test10(output, input):
    t.testing.assert_close(output, t.tensor(9.0))
    t.testing.assert_close(input.grad, t.tensor(6.0))
    print("All tests passed!")
    submit.test_submit(10, mid=mid, eid=eid)

def test11(output, theta, z):
    t.testing.assert_close(output, t.tensor([[0.0, 0.0, 1.6, 2.0]]))  # expect J(θ,z,ε) = z * 1[z > θ]
    t.testing.assert_close(theta.grad, t.tensor([0.0, -3.0, -3.0, 0.0]))  # expect dJ/dθ = -θ/ε * K((z-θ)/ε)
    t.testing.assert_close(z.grad, t.tensor([[0.0, 0.0, 1.0, 1.0]]))  # expect dJ/dz = 1[z > θ]

    print("All tests for `JumpReLU` passed!")
    submit.test_submit(11, mid=mid, eid=eid)

def test12(logits_without_sae_recon, logits_no_saes):
    t.testing.assert_close(logits_no_saes, logits_without_sae_recon)
    print("All tests passed!")

    submit.test_submit(12, mid=mid, eid=eid)

def test13(show_top_logits, gpt2, gpt2_sae):
    tests.test_show_top_logits(show_top_logits, gpt2, gpt2_sae)
    print("All tests passed!")
    submit.test_submit(13, mid=mid, eid=eid)

def test14(prompts):
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
