# DFlash Benchmark

| suite | prompts | prompt tok avg | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | TTFT | peak memory | acceptance | prefix saved | baseline prefill tok/s | dflash prefill physical tok/s | dflash prefill apparent tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 1 | 102.00 | 12.79 | 39.18 | 3.06x | n/a | n/a | 865.07 ms | 30.96 GB | 0.84 | n/a | 49.00 | 120.44 | 120.44 |

- mode: smoke
- suite: smoke
- model: mlx-community/Qwen3.6-27B-8bit
- draft: z-lab/Qwen3.6-27B-DFlash
- draft_quant: w8:gs64
- git_hash: e70c097
- max_tokens: 512
- block_tokens: 16
- repeat: 2
- cooldown: 10
- prompt_count: 1
- prompt_ids: smoke-default
- prompt_source: smoke
- prompt_tokenization_mode: chat_template
- use_chat_template: True
- target_fa_window: 0
- draft_window: 64+1024
- verify_len_cap: 0
- verify_mode: adaptive
- only_dflash: False

## Per Prompt

| prompt id | prompt tokens | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke-default | 102 | 12.79 | 39.18 | 3.06x | n/a | n/a | 0.84 |
