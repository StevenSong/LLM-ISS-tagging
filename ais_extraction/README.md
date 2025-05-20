# AIS extraction

## Notes

* `qwen2.5-vl-32b-instruct`
```bash
vllm serve \
Qwen/Qwen2.5-VL-32B-Instruct \
--max-model-len 4096 \
--max-num-seqs 1 \
--limit-mm-per-prompt image=1 \
--allowed-local-media-path /opt/gpudata/steven/ais-extraction/data
```