# Calibration data

`forget.txt` and `retain.txt` are the 128 forget and 128 retain calibration sequences used
by `frag/predictor.py` for the TOFU experiments (Appendix A.1: N_f = N_r = 128, 256 tokens).

Both are drawn from [locuslab/TOFU](https://huggingface.co/datasets/locuslab/TOFU) —
`forget10` and `retain90` respectively — and formatted as `Question: ... Answer: ...`.
TOFU consists entirely of synthetic biographies of fictitious authors; no real personal
data is involved.

One detail worth knowing if you compare numbers. These files hold one example per line, so
the newline that `load_tofu` (in `frp/data.py`) puts between the question and the answer is
flattened to a space here. `analysis/cache_norms.py` goes through `load_tofu` and therefore
sees the newline form; `frag/predictor.py` reading these files sees the space form. The
tokenization differs by one token per example, which moves FRAG in the fourth decimal — 
immaterial for ranking checkpoints, but it means the two paths are not bit-identical. The
reported table values come from the `load_tofu` path via cached norms.

For another benchmark or model, replace these with a sample of your own forget and retain
sets in the same format (plain text one example per line, or JSONL with a `text` field).
