# Every target is a one-line wrapper; behaviour lives in `ofl` (src/, Rust).
ENV ?= dry-run

.PHONY: setup dev check test fix build deploy smoke doctor clean help
setup dev check test fix build smoke doctor clean help:
	@cargo run -q -- $@

deploy:
	@cargo run -q -- deploy --env $(ENV)