import cProfile
import io
import json
import pstats
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

from .aif_core import ActionSpace, ActiveInferenceAgent, AgentConfig
from .checkpoint import save_checkpoint
from .gwm_bridge import AITGWMBridge
from .multilingual import AVAILABLE_LANGUAGES
from .onepass_ait import GateDiagnostics, OnePassAIT, StudentTrainingConfig


class _ScalarLogWriter:
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
    return _ScalarLogWriter(run_dir)


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
    ait = OnePassAIT(latent_dim=24, seed=4242)
    writer = _summary_writer()
    scalar_steps: dict[str, int] = defaultdict(int)

    def log_scalar(tag: str, value: float, *, step: int | None = None) -> None:
        if step is None:
            step = scalar_steps[tag]
            scalar_steps[tag] += 1
        writer.add_scalar(tag, value, step)
    try:
        train_summary = ait.train_student(
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
                cache_sequences=False,
            ),
            languages=AVAILABLE_LANGUAGES,
            include_reflective=True,
            shuffle=True,
            seed=4242,
        )
        print("Student boundary head summary:", json.dumps(train_summary, ensure_ascii=False, indent=2))
        backend = train_summary.get("backend") if isinstance(train_summary, dict) else None
        encoder_backend = train_summary.get("encoder_backend") if isinstance(train_summary, dict) else None
        available_devices = train_summary.get("available_devices", {}) if isinstance(train_summary, dict) else {}
        if backend:
            log_scalar(
                "backends/boundary_jit",
                1.0 if str(backend).startswith(("julia", "compiled")) else 0.0,
            )
            print("Boundary backend:", backend)
        if available_devices:
            log_scalar(
                "backends/has_gpu",
                1.0
                if any(
                    any(dev.lower().startswith(prefix) for prefix in ("cuda", "gpu", "metal"))
                    for devs in available_devices.values()
                    for dev in devs
                )
                else 0.0,
            )
            print("Backend device inventory:", available_devices)
        if encoder_backend:
            log_scalar(
                "backends/encoder_external",
                1.0 if str(encoder_backend).startswith(("julia", "r")) else 0.0,
            )
            print("Encoder backend:", encoder_backend)
        encoder_devices = []
        if isinstance(train_summary, dict):
            encoder_devices = train_summary.get("encoder_devices", [])
        if encoder_devices:
            print("Encoder devices:", encoder_devices)

        train_texts: List[str] = []
        train_segments: List[List[str]] = []
        if isinstance(train_summary, dict):
            if "cache_sequences" in train_summary:
                log_scalar(
                    "boundary/cache_sequences",
                    float(bool(train_summary["cache_sequences"])),
                )
                print("Sequence cache enabled:", bool(train_summary["cache_sequences"]))
            if "cached_sequences" in train_summary:
                log_scalar(
                    "boundary/cached_sequences",
                    float(train_summary["cached_sequences"]),
                )
                print("Cached sequences used:", train_summary["cached_sequences"])
            train_texts = train_summary.get("dataset_texts", [])
            train_segments = train_summary.get("dataset_segments", [])
        else:
            train_texts = []
            train_segments = []
        language_tags = (
            train_summary.get("dataset_tags", []) if isinstance(train_summary, dict) else []
        )
        lang_hist = train_summary.get("dataset_languages", {}) if isinstance(train_summary, dict) else {}
        lang_stats = (
            train_summary.get("dataset_language_stats", {})
            if isinstance(train_summary, dict)
            else {}
        )
        if hasattr(lang_hist, "items"):
            for idx, (lang, count) in enumerate(lang_hist.items()):
                log_scalar(f"data/language/{lang}", count, step=idx)
        print("Training language histogram:", lang_hist)
        if hasattr(lang_stats, "items"):
            for lang_idx, (lang, stats) in enumerate(lang_stats.items()):
                if not isinstance(stats, dict):
                    continue
                mean_chars = stats.get("mean_chars")
                mean_tokens = stats.get("mean_tokens")
                mean_cpt = stats.get("mean_chars_per_token")
                if mean_chars is not None:
                    log_scalar(
                        f"data/mean_chars/{lang}",
                        float(mean_chars),
                        step=lang_idx,
                    )
                if mean_tokens is not None:
                    log_scalar(
                        f"data/mean_tokens/{lang}",
                        float(mean_tokens),
                        step=lang_idx,
                    )
                if mean_cpt is not None:
                    log_scalar(
                        f"data/mean_chars_per_token/{lang}",
                        float(mean_cpt),
                        step=lang_idx,
                    )
        print("Training language statistics:", lang_stats)

        history = train_summary.get("history", []) if isinstance(train_summary, dict) else []
        for step, metrics in enumerate(history, start=1):
            if "train_loss" in metrics:
                writer.add_scalar("boundary/train_loss", metrics["train_loss"], step)
            if "val_loss" in metrics:
                writer.add_scalar("boundary/val_loss", metrics["val_loss"], step)
            if "val_f1" in metrics:
                writer.add_scalar("boundary/val_f1", metrics["val_f1"], step)

        seg_scores = [boundary_f1(ait, text, seg) for text, seg in zip(train_texts, train_segments)]
        print("Segmentation F1 per sample:", seg_scores)
        for idx, score in enumerate(seg_scores):
            writer.add_scalar("boundary/seg_f1", score, idx)

        prompt = (
            "Bob re‑examined Carol's motives and updated his provisional evaluation. "
            "ボブはキャロルの動機を再検討し、第三者の証拠で暫定評価を更新した。"
        )

        bench_iters = 12
        profiler = cProfile.Profile()
        profiler.enable()
        start = time.perf_counter()
        for _ in range(bench_iters):
            ait.encode(prompt)
        elapsed = time.perf_counter() - start
        profiler.disable()
        bench_time = elapsed / bench_iters
        log_scalar("latency/encode_ms", bench_time * 1000)
        print(f"Average encode latency: {bench_time*1000:.2f} ms")
        profile_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=profile_stream).sort_stats("cumulative")
        stats.print_stats(30)
        profile_output = profile_stream.getvalue()
        print("Encode profile (top 30 by cumulative time):\n", profile_output)
        profile_path = Path(writer.log_dir) / "encode_profile.txt"
        profile_path.write_text(profile_output, encoding="utf-8")

        diag: GateDiagnostics = ait.gate_diagnostics()
        log_scalar("gate/mask_energy", diag.mask_energy)
        print("Gate trace preview:", diag.gate_trace[:10])
        print("Attention strength per layer:", [round(v, 3) for v in diag.attention_strength])

        enc = ait.encode(prompt)
        if enc["phase_local"].size:
            phase_vals = enc["phase_local"].tolist() if hasattr(enc["phase_local"], "tolist") else enc["phase_local"]
            flat_phase = [float(val) for row in phase_vals for val in row]
            phase_energy = sum(flat_phase) / max(1, len(flat_phase))
        else:
            phase_energy = 0.0
        log_scalar("phase/mean_energy", phase_energy)

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
                    "language_tags": language_tags,
                    "language_histogram": lang_hist,
                    "language_statistics": lang_stats,
                    "boundary_backend": backend,
                    "encoder_backend": encoder_backend,
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
