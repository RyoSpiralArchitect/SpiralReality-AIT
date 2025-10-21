import json, numpy as np, math
from integrated.aif_core import ActionSpace, ActiveInferenceAgent, AgentConfig
from integrated.onepass_ait import OnePassAIT, BoundaryStudent, seeded_vector
from integrated.gwm_bridge import AITGWMBridge

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
ait = OnePassAIT(latent_dim=64, seed=4242)
ait.student.train(train_texts, teacher_segments, lr=0.25, epochs=220, reg=1e-3)

# Encode prompt once (single pass)
prompt = "Bob re‑examined Carol's motives and updated his provisional evaluation. ボブはキャロルの動機を再検討し、第三者の証拠で暫定評価を更新した。"
enc = ait.encode(prompt)

# Define action space from policies
policies = ait.policies
vecs = ait.policy_vecs
A = ActionSpace(names=policies, vecs=vecs)

# Build bridge (GWM replaced by AIT dynamics)
bridge = AITGWMBridge(ait, enc, obs_sigma=0.6)

# Agent wiring (AIF Core v2)
goal_vec = ait.goal_vec  # reuse AIT goal
cfg = AgentConfig(horizon=3, planner="cem", cem_K=36, cem_elite_frac=0.25, cem_iters=2, obs_sigma=0.6)

agent = ActiveInferenceAgent(
    dim=ait.latent_dim, action_space=A, goal_vec=goal_vec,
    step_fn=bridge.step_fn, ctx_provider=bridge.ctx_from_prefix, r3_provider=bridge.r3_from_prefix, cfg=cfg
)

# Run a short interaction: plan -> execute first action -> update, repeat
log = []
for step in range(1, 5):
    plan = agent.plan()
    a0 = plan.actions[0]
    upd = agent.act_and_update(a0)
    log.append({
        "step": step,
        "chosen_action": a0,
        "rollout_actions": plan.actions,
        "efe_agg": plan.aggregate,
        "upd": upd,
        "R3_mix": float(ait.R3_mix)
    })

with open(str(__file__).replace("run_demo.py","integrated_log.json"), "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2)

print("Integrated demo finished. Actions picked:", [e["chosen_action"] for e in log])
