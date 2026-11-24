PY ?= python3   # if your checkout path has a space, build from a space-free symlink instead (D23)
.PHONY: test test-gpu ext smoke clean bench-baselines
test:            ## CPU tests (reference, wire format, C++ CPU extension, link emulator, planner)
	$(PY) -m pytest -q tests -m "not gpu"
test-gpu:        ## CUDA kernel tests (needs a GPU)
	$(PY) -m pytest -q tests -m gpu
ext:             ## build the extension once (JIT cache)
	$(PY) -c "from kvpick import ext; m = ext.kvq(verbose=True); print('kvq built, cuda =', m.with_cuda)"
bench-baselines: ## R6: KVPick vs KVServe's CUDA kernels vs KIVI's pack step (D36, needs a GPU + triton)
	$(PY) scripts/bench_baselines.py
smoke:           ## 1-minute end-to-end smoke of every GPU script
	cd feasibility/scripts && $(PY) kv_capture.py --smoke --tiny-random && $(PY) bench_quant.py --smoke
clean:
	rm -rf ~/.cache/torch_extensions/*/kvq_* .pytest_cache

.PHONY: ext-report ext-test
ext-test:        ## CPU tests of the extension (cost model, overlap, formats, stats, shapes)
	$(PY) -m pytest -q tests -m "not gpu"
ext-report:      ## rebuild profiles, sweep, tables and figures of the extension from results/ext/*.json (CPU only)
	$(PY) scripts/ext_build_profile.py --kernels results/ext/kernels_a40.json --prefill results/ext/prefill_llama-3.1-8b_a40.json --prefill results/ext/prefill_qwen3-4b_a40.json
	$(PY) scripts/ext_build_profile.py --kernels $(FAST_KERNELS) --prefill $(FAST_PREFILL)
	$(PY) scripts/ext_sweep.py --profile results/ext/profile_a40.json --profile $(FAST_PROFILE)
	$(PY) scripts/ext_tables.py && $(PY) scripts/ext_figures.py && $(PY) scripts/ext_acc_report.py
