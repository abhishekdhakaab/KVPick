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
