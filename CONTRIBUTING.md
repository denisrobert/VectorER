# Contributing to vector-er

Thank you for your interest in contributing! This document describes how
contributions — code, docs, examples, benchmarks — are accepted into this
project. Please read it before opening a pull request; the process below is
deliberate and is the result of lessons learned from other projects.

## Governance

**The maintainer retains full control of the project.** Specifically:

- All decisions about the roadmap, architecture, public API, and what gets
  merged rest with the maintainer.
- Maintainers may close, defer, or request changes to any contribution for any
  reason, including scope, style, or prioritization.
- Being acknowledged as a contributor does **not** confer commit access or a
  vote on project direction.

This project is licensed under the MIT License (see
[`LICENSE`](LICENSE)); by contributing you agree that your contribution is
offered under that license.

---

## The discussion-first workflow (this is a hard gate)

**Every change — code, documentation, or otherwise — must begin as a
[GitHub Discussion][discussions], and the change must be approved there before
any pull request is opened.**

This is a hard requirement, not a suggestion. It mirrors a lesson many
open-source projects have learned the hard way: without a discussion-first
gate, pull requests arrive for features nobody has agreed on, and maintainers
spend review time explaining why an idea does not fit instead of reviewing
well-scoped, agreed work.

Consequences of the rule:

1. **No pull requests without an approved discussion.** A PR that does not
   link to an approved, project-relevant discussion is **closed** (not merely
   commented on). If you are not sure a change is wanted yet, open a discussion
   first and ask.
2. **Small, unambiguous fixes.** Trivial one-line typo fixes, build-broken
   repairs, and CI restorations may skip the discussion stage at the sole
   discretion of a maintainer, if they are obviously correct and
   uncontroversial. When in doubt, open the discussion anyway.
3. **The discussion comes first, in either direction.** Technology permitting,
   we prefer discussions and pull requests where a pull request can just refer
   to the discussion by number. When that is not possible (e.g. a fork-only
   workflow), state the discussion number in the PR description.

How to start a discussion:

- Search [existing discussions][discussions] first — an approved discussion
  may already cover your idea.
- Open a discussion with a clear title and a concise description: the problem
  or motivation, the proposed change, and any alternatives you considered.
- Keep discussions focused. Fork threads to a new discussion rather than
  letting one balloon.

### What "approved" means

An approval is an explicit statement by a maintainer in the discussion (e.g.
"please go ahead and open a PR for this" or "this fits; open the PR"). A
silent discussion, or one with only community replies, is **not** approval.
If a discussion has been open a while without a maintainer response, gently
mention a maintainer rather than assuming approval.

---

## The pull-request process

1. **Get approval** for your idea in a [discussion][discussions].
2. **Open the pull request** and:
   - Link the discussion in the PR description: `Closes discussion #NNN` or
     `Companion to discussion #NNN`.
   - Keep the PR focused on the agreed scope. If unrelated changes are needed,
     they belong in their own discussion + PR.
   - Prefer a short branch, clearly named (e.g. `fix/blocking-order`,
     `feat/union-merge`, `bench/yancey-em`).
3. **CI runs automatically** on every PR (see [Testing](#testing)).
4. **A maintainer reviews.** Expect at least one review round. Respond to
   feedback, push fixes, and keep the conversation in the PR thread.
5. **Merge.** Merge is performed by a maintainer only — squashing or rebasing
   at the maintainer's discretion, targeting `master`.

### Commit guidelines

- Write descriptive commit messages (a short summary line, then a body when
  useful). Match the style of the repository's existing history.
- Do not force-push a shared branch without reason; prefer additive commits
  during review.
- Do not include unrelated refactors, formatting churn of files you did not
  touch, or secrets/credentials (ever).

---

## AI-generated contributions

AI-assisted or AI-generated contributions are welcome **provided they are
declared and engineered by a human**, per the following policy:

1. **Declare it.** If any substantial part of a contribution (code, design, or
   text) was produced with the assistance of an AI system, state this clearly
   in the PR description (e.g. "implemented with an AI coding assistant and
   human review"). Sounding "human" is not a reason to omit the declaration.
2. **A human takes responsibility.** A named human contributor must review,
   verify, and stand behind every line of AI-produced code before it is
   offered for merge. This includes: understanding the code, checking it
   against the project's architecture and the approved discussion, and running
   the test suite and any relevant benchmarks.
3. **The normal review applies.** AI-contributed code has no special status in
   review: it must satisfy the same tests, documentation, style, and
   correctness bars as any other contribution — and it will be reviewed at
   least as carefully, because AI systems can produce plausible but subtly
   wrong code, licensing mismatches, or unbounded dependencies.
4. **License concerns are the human's responsibility.** Ensure any model
   outputs or libraries brought in have compatible licenses and are
   lawfully usable in this MIT-licensed project.
5. **The hard gate still applies.** An AI-generated PR with no approved
   discussion and no human engineer named as responsible is subject to the
   same closure rule as any other PR.

We do not accept fully-automated bots filing pull requests with no human
engineer engaged.

---

## Testing: required for every change

- **New features / behaviors** must ship with tests that exercise the new
  behavior, added to the existing `tests/` suite.
- **Bug fixes** must ship with a regression test that reproduces the bug and
  fails without the fix.
- **CI runs the full suite** on every PR and push to `master` across Python
  3.10–3.13 (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
- Changes to the scoring engine, comparison set, or training/EM paths are
  expected to keep the existing suite green, including the coverage-sensitive
  tests around the refactored `vectorer.scoring` and `vectorer.comparisons`
  packages.
- The test suite is run with:

  ```bash
  python -m pytest -q
  ```

---

## Developer setup

```bash
git clone https://github.com/denisrobert/VectorER.git
cd VectorER

# create a virtualenv (example with the standard library):
python -m venv .venv
# activate it — on Windows PowerShell: .\.venv\Scripts\Activate.ps1
#                         POSIX shells: source .venv/bin/activate

pip install -e ".[test]"        # core + test dependencies (numpy + faiss-cpu)
pip install -e ".[embedding]"   # optional: sentence-transformers models
pip install -e ".[openai]"      # optional: OpenAI embeddings

python -m pytest -q             # run the test suite
```

The benchmark suite lives in `benchmarks/` and the generators produce
git-ignored population data files; see the docstrings in each benchmark script
and the `.docs/` guides for how to run them and what parameters to use.

---

## Documentation

- Architecture and design rationale: `.docs/architecture.md`
- Hands-on user guide (all three modes, calibration, production notes):
  `.docs/user_guide.md`
- API reference (Sphinx, `docs/`): build with
  `pip install -r docs/requirements.txt` then `make html` in `docs/`.
- Changes to public behavior should update the relevant `.docs/` page and the
  `CHANGELOG.md` entry under `[Unreleased]`.

---

## What happens to contributions that don't follow this process

- A PR with no approved discussion is closed with a link to this document and
  an invitation to open a discussion.
- A discussion without maintainer resolution will sit until a maintainer
  responds; please nudge rather than working around it with a PR.
- A PR that is closed this way is not a reflection on you or your work. The
  discussion-first gate is how scope and architecture stay coherent across a
  project with a single controlling maintainer.

## Code of conduct

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

[discussions]: https://github.com/denisrobert/VectorER/discussions