#!/usr/bin/env python3
"""Open Frontier Labs housekeeping bot. No models: GitHub API calls and string rules.

One pass per run (a user systemd timer runs it every 10 minutes), idempotent:
every comment it posts carries a hidden marker and is never posted twice.

Does: path labels on PRs, welcome comment for first-time contributors, `/assign` and
`/unassign` commands, `needs-triage` on new issues, a daily stale sweep and DCO hints.

Auth: a fine-grained token scoped to OpenFrontierLabs, read from the file named by
OFL_BOT_TOKEN_FILE (default ~/ofl-secrets/ofl-bot-token). Never printed or put in argv.
State and logs live in ~/.ofl/bot/ (state.json, bot.log).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ORG = "OpenFrontierLabs"
REPOS = ("openfrontier", "website")
DISCORD = "https://discord.gg/M4sD8T5M8D"
CONTRIBUTING = "https://github.com/OpenFrontierLabs/openfrontier/blob/main/CONTRIBUTING.md"
SIGNATURE = "<sub>Automated by ofl-bot (a plain script, no model). Source: `tools/ofl-bot/`.</sub>"
MARK = "<!-- ofl-bot:{} -->"

STALE_AFTER = timedelta(days=45)
CLOSE_AFTER = timedelta(days=14)
STALE_LABEL = "stale"
EXEMPT_LABELS = {"roadmap", "type/rfc", "rfc", "type/tracking", "good first issue", "security"}
MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
FIRST_TIME = {"FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER"}

# First match per file adds its labels; the PR gets the union.
# Labels missing from a repo are skipped.
PATH_LABELS: dict[str, tuple[str, ...]] = {
    "src/ofl/doctor.py": ("layer/1-hardware", "area/python"),
    "tests/test_doctor.py": ("layer/1-hardware", "area/python"),
    "docs/join-a-node.md": ("layer/2-node", "area/docs"),
    "src/ofl/swarm/*": ("layer/3-network", "area/python"),
    "src/ofl/smoke/*": ("layer/4-training", "area/ml", "area/python"),
    "src/ofl/run0.py": ("layer/4-training", "area/ml", "area/python"),
    "src/ofl/moe.py": ("layer/4-training", "area/ml", "area/python"),
    "src/ofl/phases.py": ("layer/4-training", "area/ml", "area/python"),
    "configs/*": ("layer/4-training", "area/ml"),
    "docs/run0.md": ("layer/4-training", "area/docs"),
    "docs/phase*.md": ("layer/4-training", "area/docs"),
    "manifests/*": ("layer/5-storage", "area/data"),
    "ledger/*": ("layer/5-storage", "area/data"),
    "src/ofl/ledger.py": ("layer/5-storage", "area/data", "area/python"),
    "CREDITS.md": ("layer/5-storage", "area/docs"),
    "src/ofl/deploy.py": ("layer/7-hosting", "area/python"),
    "rfcs/*": ("type/rfc", "area/docs"),
    "docs/*": ("area/docs",),
    "*.md": ("area/docs",),
    "*.py": ("area/python",),
    "*.rs": ("area/rust",),
    ".github/*": ("type/chore",),
    "tools/*": ("type/chore",),
}
REPO_LABELS: dict[str, tuple[str, ...]] = {"website": ("layer/6-console", "area/web")}

HOME = Path.home()
BOT_DIR = HOME / ".ofl" / "bot"
STATE_FILE = BOT_DIR / "state.json"
LOG_FILE = BOT_DIR / "bot.log"
TOKEN_FILE = Path(os.environ.get("OFL_BOT_TOKEN_FILE", HOME / "ofl-secrets" / "ofl-bot-token"))


def now() -> datetime:
    return datetime.now(UTC)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Bot:
    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.env = {**os.environ, "GH_TOKEN": TOKEN_FILE.read_text().strip()}
        self.state: dict[str, Any] = {"items": {}, "comments": {}, "last_stale_sweep": None}
        if STATE_FILE.exists():
            try:
                self.state.update(json.loads(STATE_FILE.read_text()))
            except ValueError:  # corrupt state: start fresh, markers keep comments idempotent
                self.state["items"] = {}
        self.label_cache: dict[str, set[str]] = {}

    # -- plumbing ---------------------------------------------------------------------------

    def log(self, event: str, **fields: object) -> None:
        line = json.dumps({"ts": now().isoformat(timespec="seconds"), "event": event, **fields})
        with LOG_FILE.open("a") as fh:
            fh.write(line + "\n")
        print(line)

    def gh(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        cmd = ["gh", "api", "-X", method, path]
        if method == "GET":
            cmd.insert(2, "--paginate")
        stdin = None
        if body is not None:
            cmd += ["--input", "-"]
            stdin = json.dumps(body)
        try:
            proc = subprocess.run(
                cmd, input=stdin, capture_output=True, text=True, env=self.env, timeout=120
            )
        except subprocess.TimeoutExpired as err:
            raise RuntimeError(f"{method} {path}: timed out") from err
        if proc.returncode != 0:
            raise RuntimeError(f"{method} {path}: {proc.stderr.strip()[:300]}")
        out = proc.stdout.strip()
        if not out:
            return None
        try:
            return self.parse(out)
        except ValueError as err:
            raise RuntimeError(f"{method} {path}: bad JSON: {err}") from err

    @staticmethod
    def parse(out: str) -> Any:
        """Decode every JSON document in `out` (`gh --paginate` prints one per page)."""
        docs: list[Any] = []
        pos, decoder = 0, json.JSONDecoder()
        while pos < len(out):
            doc, pos = decoder.raw_decode(out, pos)
            docs.append(doc)
            while pos < len(out) and out[pos].isspace():
                pos += 1
        if len(docs) == 1:
            return docs[0]
        if all(isinstance(d, list) for d in docs):
            return [row for page in docs for row in page]
        return docs

    def write(
        self, method: str, path: str, body: dict[str, Any] | None, what: str, **fields: object
    ) -> None:
        if self.dry_run:
            self.log("dry-run", action=what, **fields)
            return
        self.gh(method, path, body)
        self.log(what, **fields)

    def save(self) -> None:
        if self.dry_run:
            return
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(STATE_FILE)  # atomic: a crash never leaves a half-written state.json

    def labels(self, repo: str) -> set[str]:
        if repo not in self.label_cache:
            rows = self.gh("GET", f"repos/{ORG}/{repo}/labels?per_page=100")
            self.label_cache[repo] = {row["name"] for row in rows}
        return self.label_cache[repo]

    def ensure_label(self, repo: str, name: str, color: str, description: str) -> None:
        if name in self.labels(repo):
            return
        self.write(
            "POST",
            f"repos/{ORG}/{repo}/labels",
            {"name": name, "color": color, "description": description},
            "label-create",
            repo=repo,
            label=name,
        )
        self.label_cache[repo].add(name)

    def comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        return self.gh("GET", f"repos/{ORG}/{repo}/issues/{number}/comments?per_page=100") or []

    def comment(self, repo: str, number: int, marker: str, text: str) -> None:
        body = f"{MARK.format(marker)}\n{text}\n\n{SIGNATURE}"
        self.write(
            "POST",
            f"repos/{ORG}/{repo}/issues/{number}/comments",
            {"body": body},
            "comment",
            repo=repo,
            number=number,
            marker=marker,
        )

    @staticmethod
    def has_marker(comments: list[dict[str, Any]], marker: str) -> bool:
        needle = MARK.format(marker)
        return any(needle in (c.get("body") or "") for c in comments)

    @staticmethod
    def is_bot_comment(comment: dict[str, Any]) -> bool:
        return "<!-- ofl-bot:" in (comment.get("body") or "")

    # -- rules ------------------------------------------------------------------------------

    def path_labels(self, repo: str, files: list[str]) -> set[str]:
        wanted: set[str] = set(REPO_LABELS.get(repo, ()))
        for name in files:
            for pattern, labels in PATH_LABELS.items():
                if fnmatch.fnmatch(name, pattern):
                    wanted.update(labels)
                    break
        return wanted & self.labels(repo)

    def label_pr(self, repo: str, pr: dict[str, Any], seen: dict[str, Any]) -> None:
        sha = pr["head"]["sha"]
        if seen.get("labeled_sha") == sha:
            return
        files = [
            f["filename"]
            for f in self.gh("GET", f"repos/{ORG}/{repo}/pulls/{pr['number']}/files?per_page=100")
        ]
        have = {label["name"] for label in pr["labels"]}
        add = sorted(self.path_labels(repo, files) - have)
        if add:
            self.write(
                "POST",
                f"repos/{ORG}/{repo}/issues/{pr['number']}/labels",
                {"labels": add},
                "labels",
                repo=repo,
                number=pr["number"],
                add=add,
            )
        seen["labeled_sha"] = sha

    def triage(self, repo: str, issue: dict[str, Any], seen: dict[str, Any]) -> None:
        if seen.get("triaged") or "needs-triage" not in self.labels(repo):
            return
        names = {label["name"] for label in issue["labels"]}
        age = now() - parse_time(issue["created_at"])
        if names or age > timedelta(days=1):  # a maintainer already labelled it, or state was lost
            seen["triaged"] = True
            return
        self.write(
            "POST",
            f"repos/{ORG}/{repo}/issues/{issue['number']}/labels",
            {"labels": ["needs-triage"]},
            "labels",
            repo=repo,
            number=issue["number"],
            add=["needs-triage"],
        )
        seen["triaged"] = True  # only after the label request succeeded

    def welcome(self, repo: str, item: dict[str, Any], comments: list[dict[str, Any]]) -> None:
        if item["author_association"] not in FIRST_TIME or item["user"]["type"] == "Bot":
            return
        if self.has_marker(comments, "welcome"):
            return
        kind = "pull request" if "pull_request" in item else "issue"
        text = (
            f"Welcome, @{item['user']['login']}, and thanks for your first {kind} "
            "to Open Frontier Labs.\n\n"
            f"- How to contribute and what reviews a PR: [CONTRIBUTING.md]({CONTRIBUTING})\n"
            f"- Questions or a quick chat: [Discord]({DISCORD})\n"
            "- Every commit needs a `Signed-off-by` line (`git commit -s`)."
        )
        self.comment(repo, item["number"], "welcome", text)

    def commands(self, repo: str, item: dict[str, Any], comments: list[dict[str, Any]]) -> None:
        if "pull_request" in item:
            return
        done: dict[str, bool] = self.state["comments"]
        assigned = {a["login"] for a in item["assignees"]}
        path = f"repos/{ORG}/{repo}/issues/{item['number']}/assignees"
        for c in comments:
            key = f"{repo}#{c['id']}"
            if key in done or self.is_bot_comment(c):
                continue
            acted = False
            for line in (c.get("body") or "").splitlines():
                parts = line.strip().split()
                if not parts or parts[0] not in ("/assign", "/unassign"):
                    continue
                acted = True
                target = c["user"]["login"]
                if (
                    len(parts) > 1
                    and parts[1].startswith("@")
                    and c["author_association"] in MAINTAINER_ASSOCIATIONS
                ):
                    target = parts[1].lstrip("@")
                fields = {"repo": repo, "number": item["number"], "user": target}
                if parts[0] == "/assign" and target not in assigned:
                    self.write("POST", path, {"assignees": [target]}, "assign", **fields)
                    assigned.add(target)
                elif parts[0] == "/unassign" and target in assigned:
                    self.write("DELETE", path, {"assignees": [target]}, "unassign", **fields)
                    assigned.discard(target)
            if acted:
                self.write(
                    "POST",
                    f"repos/{ORG}/{repo}/issues/comments/{c['id']}/reactions",
                    {"content": "+1"},
                    "ack",
                    repo=repo,
                    comment=c["id"],
                )
            done[key] = True  # a failed write raises above, so the command is retried next run

    def dco_hint(self, repo: str, pr: dict[str, Any], comments: list[dict[str, Any]]) -> None:
        if self.has_marker(comments, "dco"):
            return
        runs = self.gh(
            "GET", f"repos/{ORG}/{repo}/commits/{pr['head']['sha']}/check-runs?per_page=100"
        )
        runs = (
            runs["check_runs"]
            if isinstance(runs, dict)
            else [r for page in runs for r in page["check_runs"]]
        )
        if not any(
            r["name"] == "DCO" and r["conclusion"] in ("failure", "action_required") for r in runs
        ):
            return
        text = (
            "The DCO check failed: every commit needs a `Signed-off-by` line "
            "matching its author.\n\n"
            "```bash\ngit rebase --signoff main\ngit push --force-with-lease\n```\n\n"
            f"Details: [CONTRIBUTING.md]({CONTRIBUTING}#sign-your-work-dco). "
            f"Stuck? Ask in [Discord]({DISCORD})."
        )
        self.comment(repo, pr["number"], "dco", text)

    def stale_sweep(self, repo: str) -> None:
        self.ensure_label(
            repo,
            STALE_LABEL,
            "ededed",
            "No activity for a while; closes automatically unless someone replies",
        )
        items = self.gh("GET", f"repos/{ORG}/{repo}/issues?state=open&per_page=100")
        for item in items:
            number = item["number"]
            names = {label["name"] for label in item["labels"]}
            if names & EXEMPT_LABELS:
                continue
            comments = self.comments(repo, number)
            stale_at = next(
                (
                    parse_time(c["created_at"])
                    for c in reversed(comments)
                    if MARK.format("stale") in (c.get("body") or "")
                ),
                None,
            )
            if STALE_LABEL in names and stale_at:
                replied = any(
                    parse_time(c["created_at"]) > stale_at and not self.is_bot_comment(c)
                    for c in comments
                )
                if replied:
                    self.write(
                        "DELETE",
                        f"repos/{ORG}/{repo}/issues/{number}/labels/{STALE_LABEL}",
                        None,
                        "unstale",
                        repo=repo,
                        number=number,
                    )
                elif now() - stale_at >= CLOSE_AFTER:
                    state = (
                        {"state": "closed", "state_reason": "not_planned"}
                        if "pull_request" not in item
                        else {"state": "closed"}
                    )
                    # close first: a failed close then never leaves a repeated close comment
                    self.write(
                        "PATCH",
                        f"repos/{ORG}/{repo}/issues/{number}",
                        state,
                        "close",
                        repo=repo,
                        number=number,
                    )
                    self.comment(
                        repo,
                        number,
                        "closed",
                        "Closing after 14 more days without activity. "
                        "Reopen it any time if this is still relevant.",
                    )
            elif now() - parse_time(item["updated_at"]) >= STALE_AFTER:
                self.comment(
                    repo,
                    number,
                    "stale",
                    "No activity for 45 days, so this is marked stale. Reply or push a change "
                    "to keep it open; it closes in 14 days otherwise.",
                )
                self.write(
                    "POST",
                    f"repos/{ORG}/{repo}/issues/{number}/labels",
                    {"labels": [STALE_LABEL]},
                    "labels",
                    repo=repo,
                    number=number,
                    add=[STALE_LABEL],
                )

    # -- driver -----------------------------------------------------------------------------

    def run(self) -> None:
        for repo in REPOS:
            for item in self.gh("GET", f"repos/{ORG}/{repo}/issues?state=open&per_page=100"):
                number = item["number"]
                seen = self.state["items"].setdefault(f"{repo}#{number}", {})
                is_pr = "pull_request" in item
                try:
                    comments = self.comments(repo, number)
                    self.welcome(repo, item, comments)
                    if is_pr:
                        pr = self.gh("GET", f"repos/{ORG}/{repo}/pulls/{number}")
                        self.label_pr(repo, pr, seen)
                        self.dco_hint(repo, pr, comments)
                    else:
                        self.triage(repo, item, seen)
                        self.commands(repo, item, comments)
                except RuntimeError as err:
                    self.log("error", repo=repo, number=number, error=str(err))
                self.save()
        last = self.state["last_stale_sweep"]
        if last is None or now() - parse_time(last) >= timedelta(days=1):
            for repo in REPOS:
                try:
                    self.stale_sweep(repo)
                except RuntimeError as err:
                    self.log("error", repo=repo, error=str(err))
            self.state["last_stale_sweep"] = now().isoformat()
        self.save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="log what would change, write nothing"
    )
    args = parser.parse_args()
    BOT_DIR.mkdir(parents=True, exist_ok=True)
    Bot(args.dry_run).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
