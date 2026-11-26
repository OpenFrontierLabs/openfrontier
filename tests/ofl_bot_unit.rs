#![cfg(test)]
#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]
//! The ofl-bot systemd unit runs everything from its session-writable clone inside agent-sandbox.

mod common;

/// The launcher every command line of the unit must start with.
const SANDBOX: &str = "%h/.local/bin/agent-sandbox";
/// Where the unit's bot clone lives; paths below it name files of this repository.
const CLONE: &str = "%h/.ofl/bot/openfrontier/";
/// Root-owned directories the unit's `PATH` may name (it reaches agent-sandbox, which runs outside).
const ROOT_OWNED: [&str; 4] =
    ["/run/current-system/sw/bin", "/run/wrappers/bin", "/usr/bin", "/bin"];

fn service() -> String {
    std::fs::read_to_string(common::root().join("tools/ofl-bot/ofl-bot.service")).unwrap()
}

/// `(key, value)` of every `Key=value` line in `[Service]`.
fn service_settings(text: &str) -> Vec<(String, String)> {
    let mut section = "";
    let mut out = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            section = line;
        } else if section == "[Service]"
            && !line.starts_with('#')
            && let Some((key, value)) = line.split_once('=')
        {
            out.push((key.trim().to_owned(), value.trim().to_owned()));
        }
    }
    out
}

/// `(key, prefix, command)` of every Exec line; `prefix` is systemd's `-@:+!` flags.
fn exec_lines(text: &str) -> Vec<(String, String, String)> {
    service_settings(text)
        .into_iter()
        .filter(|(key, _)| key.starts_with("Exec"))
        .map(|(key, value)| {
            let command = value.trim_start_matches(['-', '@', ':', '+', '!']);
            let prefix = value[..value.len() - command.len()].to_owned();
            (key, prefix, command.to_owned())
        })
        .collect()
}

/// Every command (the pull and the bot) runs inside agent-sandbox. An Exec line that starts git or
/// python directly runs the clone's code, or a program its `.git/config` names, unsandboxed.
#[test]
fn every_command_runs_inside_agent_sandbox() {
    let lines = exec_lines(&service());
    let shape: Vec<(&str, &str)> =
        lines.iter().map(|(key, prefix, _)| (key.as_str(), prefix.as_str())).collect();
    // Only the pull may fail: a `-` on ExecStart would report a refused sandbox (exit 126) as success.
    assert_eq!(shape, [("ExecStartPre", "-"), ("ExecStart", "")]);
    for (key, _, command) in &lines {
        let words = shlex::split(command).unwrap();
        assert_eq!(words.first().map(String::as_str), Some(SANDBOX), "{key}={command}");
        let sep = words.iter().position(|w| w == "--").unwrap();
        let program = words.get(sep + 1).unwrap();
        assert!(program.starts_with("/run/current-system/sw/bin/"), "{key}: {program}");
    }
}

/// The unit's `PATH` names root-owned directories only: agent-sandbox runs outside, and any
/// directory before the system profile (mise, ~/.nix-profile, ~/.bun/bin) is session-writable.
#[test]
fn path_names_only_root_owned_directories() {
    let settings = service_settings(&service());
    let paths: Vec<&str> = settings
        .iter()
        .filter(|(key, _)| key == "Environment")
        .filter_map(|(_, value)| value.strip_prefix("PATH="))
        .collect();
    assert_eq!(paths.len(), 1, "exactly one Environment=PATH=...");
    for dir in paths[0].split(':') {
        assert!(ROOT_OWNED.contains(&dir), "session-writable or unknown PATH entry: {dir}");
    }
}

/// The script the unit runs from its clone exists in this repository, so a moved script fails here
/// instead of on the host's next timer tick.
#[test]
fn the_bot_script_the_unit_runs_exists() {
    let lines = exec_lines(&service());
    let (_, _, bot) = lines.iter().find(|(key, _, _)| key == "ExecStart").unwrap();
    let words = shlex::split(bot).unwrap();
    let script = words.iter().find_map(|w| w.strip_prefix(CLONE)).unwrap();
    assert!(common::root().join(script).is_file(), "missing {script}");
}
