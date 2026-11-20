# AGENTS.md

diffmpc-drones is a prototype: research and student code that changes quickly.
Nothing here is fixed.

## How to work here

- Write the simple, direct version. Add an abstraction, config layer or
  interface only when a present need requires it.
- Experiments are welcome. Replace what does not fit.
- `docs/` records where we stand, not rules. If a decision is in the way, change
  it and update the doc.
- Do not gold-plate: no exhaustive tests, no handling of cases that cannot
  happen, no compatibility shims. Keep a test if it catches a real mistake.
- State what was run and what was not. Everything so far has run only on a
  laptop CPU, so GPU behavior is unverified until it has run on one.
- Do not edit `third_party/diffmpc` or copy from it. The README says why.

## How to write here

Write docs and comments as a methods section is written: full sentences in
paragraphs, each opening with its claim, followed by the reason and then what
follows from it.

- State each fact once, in one place. Elsewhere, point to it.
- Give the measurement and what it was measured on, not an adjective.
- Mark what is measured, what is read from documentation, and what is untried.
- A comment says why the code is so. It does not restate the code or tell the
  story of how we found out.
- Use a heading only for a section a reader would look up. Within a section,
  open a paragraph with a bold phrase if it needs a label.
- Use a list only for items of one kind. Reasoning goes in prose.
- Wrap Markdown at 80 columns.

## Where we stand

- The repository holds DiffMPC's quadrotor example as a script,
  `src/diffmpc_drones/rl_quadrotor.py`. It follows the notebook step for step,
  except for one line that JAX 0.12.1 needs; the README says why.
- The drone model, the cost and the solver are DiffMPC's, imported from the
  submodule.
- Everything is JAX in double precision. The state is DiffMPC's 13 numbers:
  position, velocity, a `wxyz` quaternion and body rates, in metres and radians
  with Z up. The control is a thrust and three body torques.

## Practical

- `git submodule update --init` after cloning, without `--recursive`.
- `uv run pytest -m "not slow"` skips the test that compiles a training update.
  Run `uv run ruff check . && uv run ruff format .` before committing.
- Open work is in `docs/TODO.md`.
