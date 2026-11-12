# Every target is a one-line wrapper; behaviour lives in `ofl` (src/ofl/).
ENV ?= dry-run

.PHONY: setup dev check test fix build deploy smoke doctor clean help
setup dev check test fix build smoke doctor clean help:
	@uv run ofl $@

deploy:
	@uv run ofl deploy --env $(ENV)
