use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::process::Stdio;
use std::time::Duration;

use llm::builder::ParamBuilder;

use super::Tool;

pub struct ShellCommandTool;

impl Tool for ShellCommandTool {
    fn name(&self) -> &'static str {
        "run_shell"
    }
    fn description(&self) -> &'static str {
        #[cfg(target_os = "windows")]
        return "Execute a Windows cmd command on the user's machine. The machine runs Windows. The user can see the command output! Use for tasks that require terminal operations. Always prefer safe, idempotent commands and avoid destructive operations.";
        #[cfg(target_os = "linux")]
        return "Execute a Linux shell command on the user's machine. The machine runs Linux. The user can see the command output! Use for tasks that require terminal operations. Always prefer safe, idempotent commands and avoid destructive operations.";
        #[cfg(target_os = "macos")]
        return "Execute a Mac OS shell command on the user's machine. The machine runs Mac OS. The user can see the command output! Use for tasks that require terminal operations. Always prefer safe, idempotent commands and avoid destructive operations.";
    }
    fn required_params(&self) -> &'static [&'static str] {
        &["command"]
    }
    fn params(&self) -> Vec<ParamBuilder> {
        #[cfg(target_os = "windows")]
        let shell = "Using `cmd /C`";
        #[cfg(target_os = "linux")]
        let shell = "Using `sh -c`";
        #[cfg(target_os = "macos")]
        let shell = "Using `sh -c`";
        vec![
            ParamBuilder::new("command")
                .type_of("string")
                .description(format!("The exact shell command to execute ({shell})")),
            ParamBuilder::new("timeout_sec")
                .type_of("integer")
                .description("Optional timeout in seconds (defaults to 120)"),
        ]
    }
    fn execute_blocking(&self, args: Value) -> Result<Value> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'command'"))?
            .to_string();
        let timeout = args
            .get("timeout_sec")
            .and_then(|v| v.as_u64())
            .unwrap_or(120);

        // println!("> {}", command);
        print!("Do you want to execute this command? [Y/n/c] ");
        std::io::Write::flush(&mut std::io::stdout()).context("Failed to flush stdout")?;
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .context("Failed to read user input")?;
        let choice = input.trim().to_lowercase();
        if choice == "c" {
            if let Ok(mut cb) = arboard::Clipboard::new() {
                if let Err(e) = cb.set_text(&command) {
                    eprintln!("Failed to copy to clipboard: {}", e);
                } else {
                    println!("Command copied to clipboard");
                }
            } else {
                eprintln!("Failed to access clipboard");
            }
            return Ok(json!({
                "command": command,
                "executed": false,
                "copied": true
            }));
        }
        if choice == "n" {
            println!("Command execution cancelled");
            return Ok(json!({
                "command": command,
                "executed": false
            }));
        }

        print!("\x1B[1A\x1B[2K\r");
        print!("\x1B[2K\r");

        let mut child = if cfg!(target_os = "windows") {
            std::process::Command::new("cmd")
                .args(["/C", &command])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
        } else {
            std::process::Command::new("sh")
                .args(["-c", &command])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
        }
        .context("Failed to execute command")?;

        let start = std::time::Instant::now();
        let status_output = loop {
            if let Some(status) = child.try_wait().context("wait failed")? {
                let output = child.wait_with_output().context("output failed")?;
                break Ok((status, output));
            }
            if start.elapsed().as_secs() >= timeout {
                let _ = child.kill();
                break Err(anyhow!("timeout after {}s", timeout));
            }
            std::thread::sleep(Duration::from_millis(50));
        };

        // TODO: make this a textbox as well (like for regular output) but increase size a bit and
        // do not clear it after it finished.
        match status_output {
            Ok((status, output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let combined = if stderr.is_empty() {
                    stdout.clone()
                } else if stdout.is_empty() {
                    stderr.clone()
                } else {
                    format!("{}\n{}", stdout, stderr)
                };
                Ok(json!({
                    "command": command,
                    "executed": true,
                    "exit_status": status.code(),
                    "stdout": stdout,
                    "stderr": stderr,
                    "output": combined,
                }))
            }
            Err(e) => Ok(json!({
                "command": command,
                "executed": false,
                "error": e.to_string(),
            })),
        }
    }
}
