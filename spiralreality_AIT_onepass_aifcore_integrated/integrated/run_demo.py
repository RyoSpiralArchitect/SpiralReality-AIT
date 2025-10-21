import json
import time
from typing import List

from .aif_core import ActionSpace, ActiveInferenceAgent, AgentConfig
from .checkpoint import save_checkpoint
from .gwm_bridge import AITGWMBridge
from .onepass_ait import GateDiagnostics, OnePassAIT, StudentTrainingConfig

# Seed tiny training data for Student (just to get a working head)
train_texts = [
    "Bob re-examined Carol's motives and updated his provisional evaluation.",
    "Avoid premature closure; maintain hypotheses and update them with evidence.",
    "ボブはキャロルの動機を再検討し、第三者の証拠で暫定評価を更新した。",
    "結論を急ぎ過ぎないこと。内部対話で仮説を維持し、証拠で更新する。"
]
# Make pseudo teacher segments (simple space/punct split for demo)
def naive_segments(t: str):
    seg=[]; buf=""
    for ch in t:
        buf += ch
        if ch.isspace() or ch in ",.;。、「」…！？!?:;—‑-":
            if buf.strip(): seg.append(buf.strip())
            if ch.isspace()==False: seg.append(ch)
            buf=""
    if buf.strip(): seg.append(buf.strip())
    return seg

teacher_segments = [naive_segments(t) for t in train_texts]

# Build One‑Pass AIT and train Student
ait = OnePassAIT(latent_dim=24, seed=4242)
train_summary = ait.train_student(
    train_texts,
    teacher_segments,
    cfg=StudentTrainingConfig(
        lr=0.05,
        epochs=72,
        batch_size=2,
        validation_split=0.25,
        patience=8,
        hidden_dim=32,
        emb_dim=20,
        window=3,
        phase_lr=0.6,
        encoder_lr=1e-3,
    ),
)
print("Student boundary head summary:", json.dumps(train_summary, ensure_ascii=False, indent=2))


def boundary_f1(text: str, gold_segments: List[str]) -> float:
    gold_cuts = set()
    idx = 0
    for tok in gold_segments:
        idx += len(tok)
        if idx < len(text):
            gold_cuts.add(idx)
    pred_segments = ait.segment_text(text)
    pred_cuts = set()
    idx = 0
    for tok in pred_segments:
        idx += len(tok)
        if idx < len(text):
            pred_cuts.add(idx)
    tp = len(gold_cuts & pred_cuts)
    fp = len(pred_cuts - gold_cuts)
    fn = len(gold_cuts - pred_cuts)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


seg_scores = [boundary_f1(text, seg) for text, seg in zip(train_texts, teacher_segments)]
print("Segmentation F1 per sample:", seg_scores)

prompt = "Bob re‑examined Carol's motives and updated his provisional evaluation. ボブはキャロルの動機を再検討し、第三者の証拠で暫定評価を更新した。"

bench_iters = 12
start = time.perf_counter()
for _ in range(bench_iters):
    ait.encode(prompt)
bench_time = (time.perf_counter() - start) / bench_iters
print(f"Average encode latency: {bench_time*1000:.2f} ms")
diag: GateDiagnostics = ait.gate_diagnostics()
print("Gate trace preview:", diag.gate_trace[:10])
print("Attention strength per layer:", [round(v, 3) for v in diag.attention_strength])

# Encode prompt once (single pass)
enc = ait.encode(prompt)

# Define action space from policies
policies = ait.policies
vecs = ait.policy_vecs
A = ActionSpace(names=policies, vecs=vecs)

# Build bridge (GWM replaced by AIT dynamics)
bridge = AITGWMBridge(ait, enc, obs_sigma=0.6)

# Agent wiring (AIF Core v2)
goal_vec = ait.goal_vec  # reuse AIT goal
cfg = AgentConfig(horizon=3, planner="cem", cem_K=6, cem_elite_frac=0.25, cem_iters=1, obs_sigma=0.6)

agent = ActiveInferenceAgent(
    dim=ait.latent_dim, action_space=A, goal_vec=goal_vec,
    step_fn=bridge.step_fn, ctx_provider=bridge.ctx_from_prefix, r3_provider=bridge.r3_from_prefix, cfg=cfg
)

# Run a short interaction: plan -> execute first action -> update, repeat
log = {"train_summary": train_summary, "steps": []}
for step in range(1, 5):
    plan = agent.plan()
    a0 = plan.actions[0]
    upd = agent.act_and_update(a0)
    log["steps"].append({
        "step": step,
        "chosen_action": a0,
        "rollout_actions": plan.actions,
        "efe_agg": plan.aggregate,
        "upd": upd,
        "R3_mix": float(ait.R3_mix)
    })

with open(str(__file__).replace("run_demo.py","integrated_log.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            **log,
            "segmentation_f1": seg_scores,
            "encode_latency_ms": bench_time * 1000,
            "gate_trace": diag.gate_trace,
            "attention_strength": diag.attention_strength,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

save_checkpoint(str(__file__).replace("run_demo.py", "checkpoint.json"), ait.state_dict())

print("Integrated demo finished. Actions picked:", [e["chosen_action"] for e in log["steps"]])
