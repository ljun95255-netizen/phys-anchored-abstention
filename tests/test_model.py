"""test_model.py — 架构忠实性 + 参数断言 + 前向形状 + 确定性"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from aof.model import SONTRA_A, SeparationLayer, EncoderLayer, TemporalLayer, BeliefLayer

def test_parameter_counts():
    m = SONTRA_A()
    n = m.parameter_count()
    print(f"SONTRA-A params: {n:,}")
    assert 1.15e6 < n < 1.45e6, f"SONTRA-A ≈1.28M expected, got {n}"
    sep, enc, tmp, bel = (sum(p.numel() for p in mod.parameters())
                          for mod in (m.separation, m.encoder, m.temporal, m.belief))
    print(f"  sep={sep:,} enc={enc:,} temporal={tmp:,} belief={bel:,} ahead={m.ahead.mlp[0].weight.numel()*2+128*10+10:,}")
    # Swift 侧 [COMPUTED]: L1=111,705 L2=803,680 L3=264,288 L5=90,715 → 1,321,537(全栈)
    print(f"  Swift 参考: sep≈111K enc≈803K temporal≈264K belief≈90K")

def test_forward_shape():
    m = SONTRA_A().eval()
    x = torch.randn(2, 1, 24, 128)
    with torch.no_grad():
        out = m(x)
    assert out["event_probs"].shape == (2, 10), out["event_probs"].shape
    assert out["snr_db"].shape == (2, 10), out["snr_db"].shape
    assert out["open_set"].shape == (2, 1)
    print("forward shapes OK:", {k: tuple(v.shape) for k, v in out.items()})

def test_determinism():
    torch.manual_seed(0)
    m = SONTRA_A().eval()
    x = torch.randn(2, 1, 24, 128)
    with torch.no_grad():
        a = m(x)["event_probs"].numpy()
        b = m(x)["event_probs"].numpy()
    assert (a == b).all()
    print("determinism OK")

def test_mps_forward():
    if not torch.backends.mps.is_available():
        print("MPS unavailable, skip")
        return
    m = SONTRA_A().to("mps").eval()
    x = torch.randn(2, 1, 24, 128, device="mps")
    with torch.no_grad():
        out = m(x)
    assert out["event_probs"].shape == (2, 10)
    print("MPS forward OK")

if __name__ == "__main__":
    test_parameter_counts(); test_forward_shape(); test_determinism(); test_mps_forward()
    print("ALL MODEL TESTS PASSED")
