# pyem

Python/JAX port of [EM.jl](https://github.com/ndawlab/em). See `README.md`
for the actual package documentation.

This is the public branch (`main`) of a two-branch repo. There is also a
`dev` branch, checked out in a sibling worktree at `../em-jax-dev`,
holding dev-only content not meant for the public package: experimental
alternative `em_fit` implementations, and the Julia scripts that generate
`tests/fixtures/`. See that worktree's own `CLAUDE.md` for details.

`git worktree list` shows both. Public-package work belongs here on `main`;
anything experimental or Julia-fixture-generation-related belongs on `dev`.
