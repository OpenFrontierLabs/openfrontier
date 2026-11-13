//! The `ofl` binary: parse, dispatch, report. All behaviour lives in the modules.

fn main() -> std::process::ExitCode {
    ofl::cli::main()
}
