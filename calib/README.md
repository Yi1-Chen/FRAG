# Calibration data

`forget.txt` and `retain.txt` are the 128 forget and 128 retain calibration sequences used
by `frag/predictor.py` for the TOFU experiments (Appendix A.1: N_f = N_r = 128, 256 tokens).

Both are drawn from [locuslab/TOFU](https://huggingface.co/datasets/locuslab/TOFU) —
`forget10` and `retain90` respectively — and formatted as `Question: ... Answer: ...`, one
example per line, matching `load_tofu` in `frp/data.py`. TOFU consists entirely of
synthetic biographies of fictitious authors; no real personal data is involved.

For another benchmark or model, replace these with a sample of your own forget and retain
sets in the same format (plain text one example per line, or JSONL with a `text` field).
