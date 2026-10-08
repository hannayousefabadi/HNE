---
name: update-docs
description: Keep CLAUDE.md, README.md, the review/*.md design notes, and the review/00_glossary.md vocabulary in sync with the repo. Use after any change to code, config, thresholds, file or function names, pipeline behaviour, outputs, or plans, and whenever a design decision is made in conversation, without waiting to be asked. Also use when asked to write or edit any of these three kinds of document.
---

# Update docs

This repo has three kinds of documentation. Each has a different reader and a different job. When something changes, update the documents it affects in the same turn as the change. The user should never have to ask for it.

## The three documents

| | `CLAUDE.md` | `README.md` | `review/NN_<topic>.md` |
|---|---|---|---|
| Reader | Claude, at the start of every session | A person opening the repo for the first time | The user, coming back to a pipeline after weeks away |
| Question it answers | What must I know to work here without making a mistake? | What is this, and how do I install and run it? | How does this pipeline work, and why is it built this way? |
| Depth | A map and a list of rules | Surface | Full |
| Length | Short. Aim for under 60 lines | One or two screens | As long as the pipeline needs |

### `CLAUDE.md`

A working brief, not a readme. Every line should change how Claude acts in this repo. Test each line: if it were deleted, would Claude be more likely to do something wrong? If not, it does not belong.

Belongs here:
- One short paragraph on what the project is.
- The stage map: which script runs each stage, where its logic lives, what it writes.
- Conventions to follow (where paths and thresholds are defined, how scripts are configured).
- Traps that cause silent mistakes (naming mismatches, `NaN` rules, split rules).
- Pointers to the review notes.

Does not belong here:
- Installation, setup, or usage instructions. Those are README content.
- Results, metrics, patient counts.
- Explanations of how an algorithm works or why a decision was made. Those go in `review/`, and `CLAUDE.md` links to them.
- History of any kind.

### `README.md`

For someone who has never seen the repo. They want to know what it does and how to run it.

Belongs here:
- What the project is, in plain language.
- Install, configure, run, in order, with commands that work when copied.
- Where outputs land.
- The directory layout at folder level.
- A short status line.

Does not belong here:
- Design rationale, thresholds, alternatives, internal gotchas. Those go in `review/`.
- Instructions addressed to Claude.
- Function-level detail.

### `review/NN_<topic>.md`

One file per pipeline, sub-pipeline, product, or milestone. It describes the thing **as it is now**, in enough depth that the user can pick it up again without rereading the code.

Each file should cover, where relevant:
- What it does, in one or two sentences.
- Where the code lives, and how the modules and functions relate (who calls whom, what feeds what).
- The flow, step by step.
- Decisions and the reason for each.
- Rules that must not be broken.
- Parameters and thresholds with their current values.
- Alternatives that were considered or remain open.
- Future plans.
- Challenges and known limitations.
- Critical things to remember.

**It is not a changelog.** A bug that was found and fixed is not mentioned. The one exception: when the fix leaves behind something the user still needs to know, such as a constraint, an invariant, or an input format that will break again if ignored. In that case write it as a present-tense rule, not as a story.

- Keep: "`HE_MAP` entries must be bare filenames. A full `s3://` path gets double-prefixed."
- Drop: "Previously the parser crashed with `NoSuchBucket`; this was fixed by switching to `urlsplit`."

Words such as "previously", "now", "was fixed", "no longer" are a sign that history is leaking in. Rewrite the sentence to state the current behaviour, or delete it.

A known problem that is **not** fixed is current state, and belongs in the file.

### `review/00_glossary.md`

A short list of the terms this project uses in one specific sense, so the user, Claude, and the documents all mean the same thing by the same word. It lives in `review/` but is not a review note: it holds definitions only.

**Add a term only when necessary.** The bar is high. A term earns an entry when at least one of these is true:

- It has caused a real misunderstanding in conversation, or the user and Claude were found to be using it differently.
- It means different things in different fields, and this project has to pick one. Example: "validation set" is an untouched, independent cohort in clinical papers, but in machine learning it is the set used during training to choose the checkpoint, and the untouched one is the "test set".
- The project uses an everyday word in a narrow sense of its own. Example: "tile" (the ~1 mm² region that carries a signature score) against "patch" (the 112 µm crop fed to the foundation model).
- Two names are in use for one thing, or one name for two things.

Do not add:

- Standard terms used in their standard sense (Pearson r, ssGSEA, embedding).
- A term just because it appears in the code or was mentioned once.
- Anything "for completeness". Most changes add nothing to the glossary, and a session that adds no term is the normal case.

**What an entry contains.** The term, the meaning this project uses in one or two sentences, and, only if it is the source of the confusion, a single line on the other meaning it is not. Keep entries in alphabetical order.

```markdown
**Validation set.** The patients held out from weight updates but used during training to pick the checkpoint and stop early. Not an untouched cohort; that is the test set.
```

**What does not go in an entry.** Reasons, decisions, how something is implemented, parameter values, consequences, plans. Those belong in the review note for that pipeline. If an entry needs more than three sentences, the extra material is review-note content: put it there and keep the entry to the definition.

**Maintaining it.**

- If the file does not exist, create it when the first term qualifies. Do not create it empty.
- When a term's meaning in the project changes, or a renamed concept makes an entry wrong, fix the entry. Remove entries for terms the project no longer uses.
- Once a term is in the glossary, use it in that sense everywhere: in the other documents, in code comments, and in replies. When a document uses the term loosely, correct the document.
- When adding or changing an entry, say so in the end-of-reply report, in one line.

## Where a fact goes

| Fact | `CLAUDE.md` | `README.md` | `review/` |
|---|---|---|---|
| What the project is | One paragraph | One paragraph, plain language | No |
| `pip install`, `.env` setup | No | Yes | No |
| Command to run a stage | Script path in the stage map | Full command | Full command, with its settings |
| Directory layout | Stage map only | Folder level | File and function level |
| A threshold's value | No. Say where thresholds are defined | No | Yes, with its role and reason |
| Why a decision was made | No | No | Yes |
| A trap that causes silent errors | One line | No | Yes, with explanation |
| Alternatives, future plans | No | One-line status at most | Yes |
| Results and metrics | No | One-line status at most | Yes, marked as the current run |

When a fact appears in more than one document, the full version lives in `review/`. The others carry one line and stay consistent with it.

## Procedure

After a change, or when a decision is made:

1. **Work out what the change touches.** Use this mapping, and check the `review/` directory for files added since this was written.

   | Changed | Review file |
   |---|---|
   | `src/hne/core/`, `src/hne/preprocessing/`, `src/hne/preprocessing_qc/`, `scripts/preprocessing/`, `scripts/cohort_inventory/` | `review/00_preprocessing_pipeline.md` |
   | `src/hne/feature_extraction/`, `scripts/feature_extraction/` | `review/01_feature_extraction_pipeline.md` |
   | `src/hne/models/mlp.py`, `scripts/model_train/phase1/train_phase1.py`, `plots/phase1/` | `review/02_early_phase1_modeling_pipeline.md` |
   | `src/hne/models/evaluation.py`, `src/hne/models/ridge_cv.py`, `src/hne/models/data.py`, `scripts/model_train/phase1/train_phase1_ridge.py` | `review/04_phase1_ridge_baseline.md` |
   | `src/hne/feature_extraction/registration_audit.py`, `scripts/audit/` | `review/06_registration_audit.md` |
   | Plans and decisions not tied to one pipeline | `review/plan_decisions.md` |

2. **Decide which documents are affected.**
   - `review/`: almost any change to behaviour, names, parameters, outputs, decisions, or plans.
   - `README.md`: only when installing, configuring, running, the output locations, or the folder layout change.
   - `CLAUDE.md`: only when the stage map, a convention, or a trap changes.

   - `review/00_glossary.md`: only when a term meets the bar in its section above. Rarely.

   Most changes touch one review file and nothing else.

3. **Edit in place.** Change the sentences, tables, and diagrams that are now wrong. Do not append a dated note or an "update" section. Remove text that no longer describes the code.

4. **Check against the code, not memory.** Every file path, function name, column name, and default value written into a document must match the source. Read the source to confirm.

5. **A new pipeline, sub-pipeline, or milestone gets a new file**: `review/NN_<topic>.md`, with the next free number. Add it to the mapping above and to the pointers in `CLAUDE.md`.

6. **Report it.** End the reply with one line naming the documents that changed and what changed in them.

`review/` is gitignored and must stay that way. Never remove it from `.gitignore`, and never commit or force-add a file from it. Its edits do not appear in `git status`; say so when it matters.

## When not to update

- Formatting, comments, or a refactor that changes no behaviour and no name that a document mentions.
- Exploratory or debugging scripts in `tests/`.
- An idea that was discussed and rejected with nothing learned. If the rejection taught something, record it under alternatives.

## Writing style for all three

- State what is true now, in the present tense.
- Be specific: real paths, real names, real values.
- Prefer a table for parameters and a code block for layout or call flow.
- Mark anything uncertain or unverified as such. Do not present a guess as fact.
