# AutoKernel theory-advisor addendum

Copy this section into `program.md` (or include it in the Agent's system
prompt) after the normal AutoKernel benchmark instructions.

## Theory-guided search

Before editing `kernel.py`, run:

```bash
python integration/theory_advisor.py suggest --kernel <kernel_id>
```

Use the returned `strategy` as the direction for the next single, focused
experiment.  After running `bench.py`, update the advisor:

```bash
python integration/theory_advisor.py update \
  --kernel <kernel_id> \
  --run-log run.log
```

Rules:

1. `MOVE_ON` means save the current kernel and ask `orchestrate.py` for the
   next kernel.
2. `SWITCH_STRATEGY` means stop repeating the current optimization family.
3. `CONTINUE` means the measured result is still materially above the theory
   floor; try another focused experiment in the suggested family.
4. The theory floor is a lower bound, not a correctness or performance claim.
   Always use `bench.py` for KEEP/REVERT decisions.
