# Whitepaper Release Checklist

## Pre-release Validation
- [ ] Regenerate metrics with `python scripts/run_evaluation.py --seed <release-seed>` and record the commit hash.
- [ ] Refresh figures via `python docs/whitepaper/generate_figures.py` and confirm SVG timestamps match the release tag.
- [ ] Inspect `docs/whitepaper/data/evaluation_metrics.json` for anomalous latency spikes or NaN values.
- [ ] Rebuild the PDF locally with `make whitepaper` and verify the output checksum is captured in the release notes.

## Versioning & Artefact Management
- [ ] Bump the repository version in `README.md` and any package manifests to match the release tag.
- [ ] Create a git tag following the `whitepaper-v<major>.<minor>` convention.
- [ ] Upload `docs/whitepaper/whitepaper.pdf`, the JSON metrics, SVG figures, and the release checklist to GitHub Releases.

## DOI & Link Governance
- [ ] Mint or update a DOI via Zenodo (or the chosen archival service) using the release tag and capture the DOI in the release notes.
- [ ] Update the DOI badge and canonical download link in `README.md`.
- [ ] Ensure archived artefacts include a `CITATION.cff` or `codemeta.json` referencing the DOI.

## Final Publication Gate
- [ ] Run the regression suite (`pytest`) to guarantee compatibility with the release artefact.
- [ ] Sanity-check that `docs/whitepaper/whitepaper.pdf` renders figures correctly in common PDF viewers.
- [ ] Publish the announcement with links to both the GitHub Release and the DOI landing page.
