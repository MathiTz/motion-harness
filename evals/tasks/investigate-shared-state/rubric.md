# Rubric: investigate-shared-state

A correct FINDINGS.md identifies, in substance (wording may vary):

1. **Root cause**: `Batch.__init__`'s default argument `seen=[]` is evaluated once at
   function-definition time, not per call, so every `Batch()` instance that doesn't pass
   its own list shares the same underlying list object.
2. **Why it looks order-dependent**: `test_first_batch_counts_its_own_label` appends to
   that shared list; `test_second_batch_starts_empty` then sees the leftover entries,
   so the failure only appears when both tests run in the same process, in that order.
3. Does **not** modify `counter.py` or `test_counter.py` (the task explicitly asked for
   diagnosis only).

Score PASS if all three are present in FINDINGS.md; PARTIAL if (1) is identified but (2)
or the order-dependence explanation is missing or wrong; FAIL otherwise (e.g. blaming
pytest test isolation in general, a random/threading theory, or not writing the file).
