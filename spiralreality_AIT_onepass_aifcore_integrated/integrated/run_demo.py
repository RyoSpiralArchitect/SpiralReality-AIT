import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

from .aif_core import ActionSpace, ActiveInferenceAgent, AgentConfig
from .checkpoint import save_checkpoint
from .corpus import TRAIN_TEXTS, teacher_segments as build_teacher_segments
from .gwm_bridge import AITGWMBridge
from .onepass_ait import GateDiagnostics, OnePassAIT, StudentTrainingConfig

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
except Exception:  # pragma: no cover
    _TorchSummaryWriter = None


class _JSONSummaryWriter:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file = (log_dir / "events.jsonl").open("a", encoding="utf-8")

    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None = None) -> None:
        record = {
            "type": "scalar",
            "tag": tag,
            "value": float(scalar_value),
            "step": global_step,
            "timestamp": time.time(),
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _summary_writer() -> object:
    log_root = Path(__file__).resolve().parent / "logs"
    run_dir = log_root / datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    if _TorchSummaryWriter is not None:
        return _TorchSummaryWriter(log_dir=str(run_dir))
    return _JSONSummaryWriter(run_dir)


def boundary_f1(ait: OnePassAIT, text: str, gold_segments: List[str]) -> float:
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


def main() -> None:
    train_texts = list(TRAIN_TEXTS)
    teacher_segments = build_teacher_segments(train_texts)

    ait = OnePassAIT(latent_dim=24, seed=4242)
    writer = _summary_writer()
    try:
        train_summary = ait.train_student(
            train_texts,
            teacher_segments,
            cfg=StudentTrainingConfig(
                lr=0.05,
                epochs=84,
                batch_size=4,
                validation_split=0.2,
                patience=10,
                hidden_dim=32,
                emb_dim=20,
                window=3,
                phase_lr=0.6,
                encoder_lr=1e-3,
            ),
        )
        print("Student boundary head summary:", json.dumps(train_summary, ensure_ascii=False, indent=2))

        history = train_summary.get("history", []) if isinstance(train_summary, dict) else []
        for step, metrics in enumerate(history, start=1):
            if "train_loss" in metrics:
                writer.add_scalar("boundary/train_loss", metrics["train_loss"], step)
            if "val_loss" in metrics:
                writer.add_scalar("boundary/val_loss", metrics["val_loss"], step)
            if "val_f1" in metrics:
                writer.add_scalar("boundary/val_f1", metrics["val_f1"], step)

        seg_scores = [boundary_f1(ait, text, seg) for text, seg in zip(train_texts, teacher_segments)]
        print("Segmentation F1 per sample:", seg_scores)
        for idx, score in enumerate(seg_scores):
            writer.add_scalar("boundary/seg_f1", score, idx)

        prompt = (
            "Bob re‑examined Carol's motives and updated his provisional evaluation. "
            "ボブはキャロルの動機を再検討し、第三者の証拠で暫定評価を更新した。"
        )

        bench_iters = 12
        start = time.perf_counter()
        for _ in range(bench_iters):
            ait.encode(prompt)
        bench_time = (time.perf_counter() - start) / bench_iters
        writer.add_scalar("latency/encode_ms", bench_time * 1000, 0)
        print(f"Average encode latency: {bench_time*1000:.2f} ms")
        diag: GateDiagnostics = ait.gate_diagnostics()
        writer.add_scalar("gate/mask_energy", diag.mask_energy, 0)
        print("Gate trace preview:", diag.gate_trace[:10])
        print("Attention strength per layer:", [round(v, 3) for v in diag.attention_strength])

        enc = ait.encode(prompt)
        if enc["phase_local"].size:
            phase_vals = enc["phase_local"].tolist() if hasattr(enc["phase_local"], "tolist") else enc["phase_local"]
            flat_phase = [float(val) for row in phase_vals for val in row]
            phase_energy = sum(flat_phase) / max(1, len(flat_phase))
        else:
            phase_energy = 0.0
        writer.add_scalar("phase/mean_energy", phase_energy, 0)

        policies = ait.policies
        vecs = ait.policy_vecs
        action_space = ActionSpace(names=policies, vecs=vecs)

        bridge = AITGWMBridge(ait, enc, obs_sigma=0.6)
        goal_vec = ait.goal_vec
        cfg = AgentConfig(horizon=3, planner="cem", cem_K=6, cem_elite_frac=0.25, cem_iters=1, obs_sigma=0.6)

        agent = ActiveInferenceAgent(
            dim=ait.latent_dim,
            action_space=action_space,
            goal_vec=goal_vec,
            step_fn=bridge.step_fn,
            ctx_provider=bridge.ctx_from_prefix,
            r3_provider=bridge.r3_from_prefix,
            cfg=cfg,
        )

        log = {"train_summary": train_summary, "steps": []}
        for step in range(1, 5):
            plan = agent.plan()
            a0 = plan.actions[0]
            upd = agent.act_and_update(a0)
            log["steps"].append(
                {
                    "step": step,
                    "chosen_action": a0,
                    "rollout_actions": plan.actions,
                    "efe_agg": plan.aggregate,
                    "upd": upd,
                    "R3_mix": float(ait.R3_mix),
                }
            )

        out_path = Path(str(__file__).replace("run_demo.py", "integrated_log.json"))
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    **log,
                    "segmentation_f1": seg_scores,
                    "encode_latency_ms": bench_time * 1000,
                    "gate_trace": diag.gate_trace,
                    "attention_strength": diag.attention_strength,
                    "gate_mask_energy": diag.mask_energy,
                    "phase_local_mean": phase_energy,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        save_checkpoint(str(__file__).replace("run_demo.py", "checkpoint.json"), ait.state_dict())
        print("Integrated demo finished. Actions picked:", [e["chosen_action"] for e in log["steps"]])
    finally:
        if hasattr(writer, "close"):
            writer.close()


if __name__ == "__main__":
    main()
