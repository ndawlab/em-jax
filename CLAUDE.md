# pyem

Python/JAX port of [EM.jl](https://github.com/ndawlab/em). See `README.md`
for the actual package documentation (install, API, examples).

## Repo layout

This is the public branch (`main`) of a two-branch repo, pushed to
https://github.com/ndawlab/em-jax (**public**). There is also a `dev`
branch, checked out in a sibling worktree at `../em-jax-dev` (not pushed
anywhere), holding dev-only content not meant for the public package:
experimental alternative `em_fit` implementations, and the Julia scripts
that generate `tests/fixtures/`. See that worktree's own `CLAUDE.md` for
details.

`git worktree list` shows both. Public-package work belongs here on `main`;
anything experimental or Julia-fixture-generation-related belongs on `dev`.
Since `origin/main` is public and live, don't push here without being asked.

## Docstring/comment conventions

Docstrings follow standard Python (Google-style) structure: a one-line
summary, `Args:` (one line per parameter), `Returns:`, and only a brief
optional `Notes:` for how/why/limitations -- not prose-heavy, notes-only
docstrings. A single factual "Direct port of EM.jl's X" attribution per
docstring is fine, but avoid narrating the Julia-vs-Python comparison,
repeating "matches Julia exactly"-style qualifiers, or referencing how or
why something was written (session history, specific test files, informal
benchmark numbers) -- that belongs in commit messages, not in the code.
This applies to `pyem/`, `tests/`, `examples/`; it does not apply to
`experimental/` on the `dev` branch, which is intentionally rougher.
