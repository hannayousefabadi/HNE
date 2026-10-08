---
name: update-graphify
description: Refresh the Graphify knowledge graph in graphify-out/ after code changes. Use whenever a code file in this repo is created, edited, renamed, moved, or deleted, at the end of the turn that made the change, without waiting to be asked. Also use when asked to rebuild, query, or check the graph.
---

# Update Graphify

`graphify-out/` holds a knowledge graph of this repo's code: files, classes, functions, and the calls and imports between them. It goes stale as soon as code changes. Refresh it after every code change.

## Refresh

Run from the repo root, once, after all the edits for the turn are done:

```bash
graphify update .
```

This re-parses the code locally. It needs no API key and no login, and takes a few seconds. It rewrites `graphify-out/graph.json`, `graph.html`, and `GRAPH_REPORT.md`.

After a change that deletes or merges code, the command may refuse to overwrite because the new graph has fewer nodes than the old one. If the shrink is expected, rerun with:

```bash
graphify update . --force
```

`graphify-out/` is gitignored in full and is never committed. It is a local by-product that anyone can regenerate. `.graphifyignore` is configuration and stays tracked.

Mention the refresh in one short line of the reply, with the node and edge counts the command prints.

## Git hooks

`post-commit` and `post-checkout` hooks (installed with `graphify hook install`) also rebuild the graph on every commit and branch switch, which covers code the user edits by hand. The hooks live in `.git/hooks/`, so they are local to this clone; check them with `graphify hook status`. Still run the refresh after editing, since uncommitted changes are not picked up by the hooks.

## When to run it

- Any `.py` file under `src/`, `scripts/`, `plots/`, or `tests/` was created, edited, renamed, moved, or deleted.
- Not needed for changes to data, results, images, or Markdown only.

## If something is wrong

- `graphify: command not found`: install it with `pip install graphifyy` (the PyPI package has two y's).
- `graphify-out/` is missing: rebuild from scratch with `graphify extract . --code-only`, then `graphify cluster-only . --no-label`.
- Data directories showing up in the graph: add them to `.graphifyignore` in the repo root.

## Using the graph

Before a broad search through the code, the graph can answer structural questions faster:

```bash
graphify explain "preprocess_patient"       # a node and its neighbours
graphify affected "load_features_and_targets"   # what depends on it
graphify path "train" "S3DataLoader"        # how two nodes connect
graphify god-nodes                          # most connected nodes
```

`graphify-out/GRAPH_REPORT.md` is the readable summary. `graphify-out/graph.html` opens in a browser.

The graph covers code structure only. It was built without semantic extraction, so it does not capture meaning from docs or comments.
