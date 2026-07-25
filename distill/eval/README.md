# Evaluation

The evaluator compares the base, distilled, and DPO checkpoints for either
branch.

It writes per-model response predictions plus a JSON summary containing exact
match, normalized exact match, token F1, non-empty rate, average output length,
and chosen-over-rejected preference accuracy. Comparisons are reported as
deltas from the first configured model.

The bundled configs use sampled published training rows for pipeline
validation. Use independent held-out data for final model claims.
