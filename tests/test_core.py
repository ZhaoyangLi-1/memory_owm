"""Unit tests that need no data / GPU:  pytest -q tests"""
import numpy as np
import pandas as pd
import pytest
import torch

from owm.analysis.bootstrap import bootstrap_scores
from owm.data import goal_parser as gp
from owm.data.decision_rules import assign_types
from owm.data.gt_state import match_target, project
from owm.data.sampling import future_indices, history_indices

# every distinct instruction that occurs in the training data of the six tasks (scripts/p1_labels.py re-checks the
# live data and must also report 100% coverage)
PICK = [(f"pick up the {c} cube and place it on the target, repeating this {v}action {w} times, then press the button to stop", c, n)
        for c in gp.COLORS for v in ("", "pick-and-place ") for w, n in (("two", 2), ("three", 3), ("four", 4), ("five", 5))]
PICK += [(f"pick up the {c} cube and place it on the target, then press the button to stop", c, 1) for c in gp.COLORS]


@pytest.mark.parametrize("text,color,count", PICK)
def test_goal_pickxtimes(text, color, count):
    assert gp.parse_goal("PickXtimes", text) == {"color": color, "count": count}
    v = gp.goal_vector("PickXtimes", text)
    assert v.shape == (gp.goal_dim("PickXtimes"),) and v.sum() == 2


@pytest.mark.parametrize("task,prefix", [("VideoUnmask", "watch the video carefully, then"),
                                          ("VideoUnmaskSwap", "watch the video carefully, then"),
                                          ("ButtonUnmaskSwap", "first press both buttons on the table, then")])
def test_goal_unmask(task, prefix):
    g = gp.parse_goal(task, f"{prefix} pick up the container hiding the blue cube")
    assert g == {"color": "blue", "color2": None}
    g = gp.parse_goal(task, f"{prefix} pick up the container hiding the red cube, finally pick up another container hiding the green cube")
    assert g == {"color": "red", "color2": "green"}
    assert gp.goal_vector(task, f"{prefix} pick up the container hiding the red cube").tolist() == [1, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("text,count", [
    ("watch the video carefully, then pick up the same block that was previously picked up again, finally put it down and press the button to stop", 1),
    ("watch the video carefully, then pick up the same cube that was previously picked up again, finally press the button to stop", 1),
    ("watch the video carefully, then pick up the same cube that was previously picked up for three times, finally press the button to stop", 3),
    ("watch the video carefully, then repeatedly pick up and put down the same block that was previously picked up for two times, finally put it down and press the button to stop", 2)])
def test_goal_videorepick(text, count):
    assert gp.parse_goal("VideoRepick", text)["count"] == count


def test_goal_failures_are_loud():
    with pytest.raises(gp.GoalParseError):
        gp.parse_goal("PickXtimes", "pick up the purple cube and place it on the target")
    with pytest.raises(gp.GoalParseError):
        gp.parse_goal("VideoRepick", "repeatedly pick it up a few times")
    assert gp.goal_vector("MoveCube", "watch the video carefully, then move the cube to the target in the same manner as before").tolist() == [0.0]


def test_history_indices():
    assert history_indices(168, 16, 80) == [8, 24, 40, 56, 72, 88, 104, 120, 136, 152, 168]
    assert history_indices(5, 16, 80) == [5]
    h = history_indices(2000, 16, 80)
    assert len(h) == 80 and h[-1] == 2000 and all(b - a == 16 for a, b in zip(h, h[1:]))
    assert future_indices(100, 120, [16, 32]) == ([116, 119], [False, True])


def test_rules():
    assert assign_types("PickXtimes", list("ababc")) == ["non_critical", "other", "memory_critical", "other", "memory_critical"]
    assert assign_types("VideoUnmaskSwap", list("aba")) == ["memory_critical", "other", "memory_secondary"]
    assert assign_types("ButtonUnmaskSwap", list("abcdc")) == ["other", "other", "memory_critical", "other", "memory_secondary"]
    assert assign_types("VideoRepick", list("ababc")) == ["memory_critical", "other", "non_critical", "other", "other"]
    assert assign_types("MoveCube", list("ab")) == ["memory_critical", "other"]


def test_projection_and_matching():
    K = np.array([[128, 0, 128], [0, 128, 128], [0, 0, 1.0]])
    E = np.array([[0, 1, 0, 0], [0.8944, 0, -0.4472, -0.0894], [-0.4472, 0, -0.8944, 0.4919]])  # RoboMME front camera
    uv, z = project(np.array([[0.0, 0.0, 0.0]]), K, E)
    assert z[0] > 0 and 0 < uv[0, 0] < 1 and 0 < uv[0, 1] < 1
    cand = np.array([[0.5, 0.25], [0.2, 0.8]])                 # (u, v)
    assert match_target((64, 128), cand)[0] == 0               # point is [row, col] = [v, u] * 256
    assert match_target((205, 51), cand)[0] == 1
    assert match_target((10, 10), cand)[0] == -1               # farther than 20 px -> no match


def test_bootstrap_scores():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(dict(key=[f"e{i}" for i in range(n)], floor=rng.random(n) < 0.33, wm=rng.random(n) < 0.33,
                           ceil=np.ones(n, bool))).astype({"floor": float, "wm": float, "ceil": float})
    s = bootstrap_scores(df, n=2000)
    assert s["ci_low"]["S"] < 0 < s["ci_high"]["S"] and s["acc_ceil"] == 1.0
    df["wm"] = 1.0
    assert bootstrap_scores(df, n=500)["S"] == pytest.approx(1.0)


def test_readout_shapes_and_permutation_equivariance():
    from types import SimpleNamespace
    from owm.readout.model import Readout, count_parameters, decode, is_correct, readout_loss
    torch.manual_seed(0)
    m = Readout(d_cand=15, d_goal=6, d_evid=128, n_options=3).eval()
    assert 1.5e6 < count_parameters(m) < 3e6
    B, Nc, Ne = 2, 5, 24
    b = SimpleNamespace(cand=torch.randn(B, Nc, 15), cand_id=torch.randint(0, 32, (B, Nc)), cand_uv=torch.rand(B, Nc, 2),
                        cand_type=torch.tensor([[0, 0, 0, 0, 1]] * B), cand_pad=torch.zeros(B, Nc, dtype=torch.bool),
                        goal=torch.randn(B, 6), option_avail=torch.ones(B, 3, dtype=torch.bool), evid=torch.randn(B, Ne, 128),
                        evid_id=torch.full((B, Ne), 33), evid_uv=torch.rand(B, Ne, 2),
                        evid_step=torch.tensor([[1] * 12 + [2] * 12] * B), evid_pad=torch.zeros(B, Ne, dtype=torch.bool))
    ll, pl = m(b)
    assert ll.shape == (B, 3) and pl.shape == (B, Nc) and (pl[:, -1] < -1e8).all()      # the arm is never a target
    perm = torch.tensor([2, 0, 3, 1, 4])
    b2 = SimpleNamespace(**{**vars(b), **{k: getattr(b, k)[:, perm] for k in ("cand", "cand_id", "cand_uv", "cand_type", "cand_pad")}})
    ll2, pl2 = m(b2)
    assert torch.allclose(ll, ll2, atol=1e-4) and torch.allclose(pl[:, perm], pl2, atol=1e-4)  # list order carries no info
    need = torch.tensor([True, False, False])
    loss = readout_loss(ll, pl, torch.tensor([0, 1]), torch.tensor([2, -1]), need)
    assert torch.isfinite(loss)
    lab, obj = decode(ll.softmax(-1), pl.softmax(-1), need)
    assert is_correct(lab, obj, lab, obj, need).all()
