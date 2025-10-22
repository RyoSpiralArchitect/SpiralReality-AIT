# Event Operations Plan

## Governance Structure
- **Steering Committee:** Maintains competition roadmap, approves rule changes,
  and coordinates partnerships. Meets bi-weekly.
- **Technical Leads:** Own CI infrastructure, benchmark curation, and security
  reviews. Provide daily triage for submission pipeline alerts.
- **Community Liaisons:** Moderate discussion forums, ensure documentation is up
  to date, and surface participant feedback to the committee.
- **Review Board:** Evaluates finalist submissions for compliance, fairness, and
  reproducibility before leaderboard publication.

## Operational Schedule
| Week | Activities |
| --- | --- |
| -4 to -2 | Finalise rules, dry-run CI pipeline, publish warm-up materials. |
| -1 | Open registration, host onboarding workshop, release baseline notebook. |
| 0 | Competition launch, enable automated submission endpoint. |
| 1-5 | Weekly office hours, leaderboard refresh each Friday, publish patch
notes for infrastructure updates. |
| 6 | Submission freeze, run final evaluation sweep, collect judge reviews. |
| 7 | Winners announcement, post-mortem with stakeholders, archive datasets. |

## Participant Guidelines
- Adhere to the evaluation metrics defined in `event_design.md`.
- Submit patches via pull requests or managed uploads; all code must include
  unit tests exercising new behaviour.
- Respect latency budget and resource quotas defined by the CI system.
- Document any third-party assets and confirm licensing compatibility.
- Engage respectfully in community channels; violations may result in
  disqualification.

## Automation and CI Requirements
- **Continuous Integration:** Extend `.github/workflows/` to execute the full
  `pytest` suite and latency profiling on each submission.
- **Automated Judging:** Implement a job that loads submitted checkpoints via
  `checkpoint.py` and runs the multilingual evaluation harness.
- **Security Scans:** Integrate dependency and secret scanning prior to running
  user code.
- **Artifact Retention:** Store evaluation outputs (metrics JSON, logs,
  gate diagnostics) for at least 90 days.
- **Leaderboard Sync:** Push verified metrics to the public leaderboard store
  after CI success, with manual override for exceptional cases.

## Risk and Incident Response
- Establish on-call rotations for technical leads with 1-hour response targets
  for CI outages.
- Maintain rollback scripts for infrastructure changes impacting evaluation.
- Document incident reports and retrospective actions in `docs/incidents/`.

## Communication Channels
- **Announcements:** RSS feed and mailing list for rule updates.
- **Support:** Issue tracker with labelled templates (`help-wanted`,
  `clarification`).
- **Workshops:** Recorded sessions hosted weekly; slides archived in
  `docs/community/workshops/`.

