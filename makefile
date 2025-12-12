.PHONY: all_tests \
        test_naive_small test_naive_medium test_naive_big \
        test_chunked_small test_chunked_medium test_chunked_big

test_naive_small:
	@echo "===== Running Naive Shader: Small Matrix ====="
	python main.py --mode naive --size small
	@echo ""

test_naive_medium:
	@echo "===== Running Naive Shader: Medium Matrix ====="
	python main.py --mode naive --size medium
	@echo ""

test_naive_big:
	@echo "===== Running Naive Shader: Big Matrix ====="
	python main.py --mode naive --size big
	@echo ""

test_chunked_small:
	@echo "===== Running Chunked Shader: Small Matrix ====="
	python main.py --mode chunked --size small
	@echo ""

test_chunked_medium:
	@echo "===== Running Chunked Shader: Medium Matrix ====="
	python main.py --mode chunked --size medium
	@echo ""

test_chunked_big:
	@echo "===== Running Chunked Shader: Big Matrix ====="
	python main.py --mode chunked --size big
	@echo ""

all_tests: \
	test_naive_small \
	test_naive_medium \
	test_naive_big \
	test_chunked_small \
	test_chunked_medium \
	test_chunked_big
	@echo "===== All tests completed ====="
