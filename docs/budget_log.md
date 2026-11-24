# Budget log (cap $10, interim report at $6, stop at $9)

Append rows with `python scripts/ext_budget.py add "event" --gpu A40 --rate <live $/h> --hours <planned h>` (refuses if over the cap) and
`... add "pod stopped" --spent <actual $>` when a pod ends. Prices are the LIVE console price at rental time.

| date | event | GPU | $/h | planned h | planned $ | spent | remaining |
|---|---|---|---|---|---|---|---|
