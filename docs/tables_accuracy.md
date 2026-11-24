## llama-3.1-8b

| task | ctx | format | ratio | n | score / ppl | change vs baseline [95% CI] | verdict | flips R>W / W>R | outputs differ | class |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | all | fp8 | 2.0x | 1319 | 80.3 | -1.59 pts [-2.96, -0.15] | worse | 58 / 37 | 775 | normal |
| gsm8k | all | kv4 | 3.2x | 1319 | 80.9 | -0.99 pts [-2.50, +0.45] | no clear difference | 55 / 42 | 870 | normal |
| gsm8k | all | int4_g32 | 3.2x | 1319 | 80.1 | -1.74 pts [-3.34, -0.15] | worse | 73 / 50 | 994 | normal |
| gsm8k | all | int8_row | 1.94x | 1319 | 82.3 | +0.45 pts [-0.76, +1.67] | no clear difference | 31 / 37 | 552 | strict |
| gsm8k | all | raw (baseline) | 1.0x | 1319 | 81.9 | | | | | |
| niah | 8192 | fp8 | 2.0x | 55 | 100.0 | +0.00 pts [+0.00, +0.00] | no clear difference (n<200: wide interval) | 0 / 0 | 0 | strict |
| niah | 8192 | kv4 | 3.2x | 55 | 100.0 | +0.00 pts [+0.00, +0.00] | no clear difference (n<200: wide interval) | 0 / 0 | 0 | strict |
| niah | 8192 | int8_row | 1.94x | 55 | 100.0 | +0.00 pts [+0.00, +0.00] | no clear difference (n<200: wide interval) | 0 / 0 | 0 | strict |
| niah | 8192 | raw (baseline) | 1.0x | 55 | 100.0 | | | | | |
| niah | 32768 | fp8 | 2.0x | 55 | 100.0 | +0.00 pts [+0.00, +0.00] | no clear difference (n<200: wide interval) | 0 / 0 | 0 | strict |
| niah | 32768 | kv4 | 3.2x | 55 | 100.0 | +0.00 pts [+0.00, +0.00] | no clear difference (n<200: wide interval) | 0 / 0 | 1 | strict |
| niah | 32768 | int8_row | 1.94x | 55 | 100.0 | +0.00 pts [+0.00, +0.00] | no clear difference (n<200: wide interval) | 0 / 0 | 0 | strict |
| niah | 32768 | raw (baseline) | 1.0x | 55 | 100.0 | | | | | |
| niah | 131072 | fp8 | 2.0x | 55 | 98.2 | +0.00 pts [+0.00, +0.00] | no clear difference (n<200: wide interval) | 0 / 0 | 0 | strict |
| niah | 131072 | kv4 | 3.2x | 55 | 98.2 | +0.00 pts [+0.00, +0.00] | no clear difference (n<200: wide interval) | 0 / 0 | 0 | strict |
| niah | 131072 | int8_row | 1.94x | 55 | 98.2 | +0.00 pts [+0.00, +0.00] | no clear difference (n<200: wide interval) | 0 / 0 | 0 | strict |
| niah | 131072 | raw (baseline) | 1.0x | 55 | 98.2 | | | | | |
| pg19_32768 | all | kv4 | 3.2x | 77 | 9.747 | +0.25% [+0.14, +0.36] | worse (n<200: wide interval) |  |  | strict |
| pg19_32768 | all | int8_row | 1.94x | 77 | 9.727 | +0.04% [+0.00, +0.08] | worse (n<200: wide interval) |  |  | strict |
| pg19_32768 | all | raw (baseline) | 1.0x | 77 | 9.723 | | | | | |
| pg19_8192 | all | kv4 | 3.2x | 95 | 10.117 | +0.15% [+0.07, +0.22] | worse (n<200: wide interval) |  |  | strict |
| pg19_8192 | all | int8_row | 1.94x | 95 | 10.104 | +0.02% [-0.02, +0.05] | no clear difference (n<200: wide interval) |  |  | strict |
| pg19_8192 | all | raw (baseline) | 1.0x | 95 | 10.102 | | | | | |
| wikitext | all | fp8 | 2.0x | 141 | 6.531 | +0.08% [+0.04, +0.11] | worse (n<200: wide interval) |  |  | strict |
| wikitext | all | kv4 | 3.2x | 141 | 6.538 | +0.19% [+0.14, +0.25] | worse (n<200: wide interval) |  |  | strict |
| wikitext | all | int4_g32 | 3.2x | 141 | 6.540 | +0.22% [+0.17, +0.28] | worse (n<200: wide interval) |  |  | strict |
| wikitext | all | int8_row | 1.94x | 141 | 6.525 | -0.01% [-0.04, +0.01] | no clear difference (n<200: wide interval) |  |  | strict |
| wikitext | all | raw (baseline) | 1.0x | 141 | 6.526 | | | | | |

## qwen3-4b

| task | ctx | format | ratio | n | score / ppl | change vs baseline [95% CI] | verdict | flips R>W / W>R | outputs differ | class |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | all | kv4 | 3.2x | 1319 | 90.0 | -0.61 pts [-1.52, +0.30] | no clear difference | 23 / 15 | 710 | normal |
| gsm8k | all | int4_g32 | 3.2x | 1319 | 90.1 | -0.45 pts [-1.44, +0.53] | no clear difference | 26 / 20 | 1007 | strict |
| gsm8k | all | int8_row | 1.94x | 1319 | 90.8 | +0.15 pts [-0.38, +0.68] | no clear difference | 6 / 8 | 351 | strict |
| gsm8k | all | raw (baseline) | 1.0x | 1319 | 90.6 | | | | | |
| wikitext | all | kv4 | 3.2x | 145 | 9.092 | +0.07% [-0.00, +0.14] | no clear difference (n<200: wide interval) |  |  | strict |
| wikitext | all | int4_g32 | 3.2x | 145 | 9.186 | +1.11% [+0.94, +1.29] | worse (n<200: wide interval) |  |  | normal |
| wikitext | all | int8_row | 1.94x | 145 | 9.088 | +0.03% [-0.00, +0.07] | no clear difference (n<200: wide interval) |  |  | strict |
| wikitext | all | raw (baseline) | 1.0x | 145 | 9.086 | | | | | |
