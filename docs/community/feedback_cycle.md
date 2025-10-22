# Feedback and Continuous Improvement Plan

## Objectives
- Collect structured feedback from participants, reviewers, and organisers after
  each major milestone.
- Translate insights into actionable roadmap updates and documentation changes.
- Maintain transparency by publishing aggregated outcomes and follow-up actions.

## Feedback Channels
1. **Post-sprint surveys:** Short questionnaires circulated at the end of each
   sprint cycle to capture satisfaction, blockers, and feature requests.
2. **Issue templates:** Dedicated GitHub issue forms for bug reports,
   infrastructure incidents, and general suggestions.
3. **Office hour notes:** Summaries from weekly sessions stored in
   `docs/community/workshops/notes/`.
4. **Retrospective board:** Shared whiteboard for the organising team to record
   keep/stop/start items.

## Survey Workflow
- Publish surveys using privacy-friendly tooling (e.g., LimeSurvey, Google
  Forms) with anonymised response collection.
- Store raw exports under `surveys/<YYYY-MM-DD>.csv` with restricted access.
- Summaries are committed to the repository as Markdown tables and referenced in
  the weekly update posts.

## Issue Labelling
- `feedback`: General suggestions, documentation gaps, UX improvements.
- `incident`: Unexpected outages or performance regressions requiring immediate
  action.
- `data-quality`: Reports on corpus inconsistencies or multilingual coverage
  gaps.
- `evaluation`: Questions or problems related to metrics, leaderboards, or
  benchmark scripts.

## Improvement Cycle
1. **Collect:** Gather survey responses and triage incoming issues weekly.
2. **Synthesize:** Community liaisons produce a summary noting trends and
   priority areas.
3. **Plan:** Steering committee reviews summaries and logs actionable items in
   the roadmap tracker.
4. **Execute:** Technical leads implement approved changes, referencing the
   original feedback ticket.
5. **Report:** Publish outcome updates in the community portal and close the
   feedback issues with a link to the relevant patches.

## Templates and Automation
- Add automated reminders in CI to prompt submitters to complete surveys after
  successful evaluations.
- Use GitHub Actions to tag stale feedback items after 30 days without updates.
- Provide a default `ISSUE_TEMPLATE` that pre-fills context (submission ID,
  benchmark target, logs) to reduce back-and-forth.

