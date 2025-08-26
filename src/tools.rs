use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::process::Stdio;

use llm::builder::{FunctionBuilder, LLMBuilder, ParamBuilder};
use llm::ToolCall;

pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn required_params(&self) -> &'static [&'static str] {
        &[]
    }
    fn params(&self) -> Vec<ParamBuilder>;
    fn register_on(&self, builder: LLMBuilder) -> LLMBuilder {
        let mut fb = FunctionBuilder::new(self.name()).description(self.description());
        for p in self.params() {
            fb = fb.param(p);
        }
        let required = self
            .required_params()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let fb = if required.is_empty() {
            fb
        } else {
            fb.required(required)
        };
        builder.function(fb)
    }
    fn execute_blocking(&self, args: Value) -> Result<Value>;
}

pub struct ToolsRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl Default for ToolsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolsRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }
    pub fn with_default() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(ShellCommandTool));
        reg
    }
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }
    pub fn apply_to_builder(&self, mut builder: LLMBuilder) -> LLMBuilder {
        for t in &self.tools {
            builder = t.register_on(builder);
        }
        builder
    }
    pub fn find(&self, name: &str) -> Option<&dyn Tool> {
        for t in &self.tools {
            if t.name() == name {
                return Some(t.as_ref());
            }
        }
        None
    }
    pub fn handle_tool_call(&self, call: &ToolCall) -> Result<Value> {
        let name = &call.function.name;
        let args: Value = serde_json::from_str(&call.function.arguments)
            .with_context(|| format!("Failed parsing tool args for {}", name))?;
        let tool = self
            .find(name)
            .ok_or_else(|| anyhow!("Unknown tool: {}", name))?;
        tool.execute_blocking(args)
    }
}

pub struct ShellCommandTool;

impl Tool for ShellCommandTool {
    fn name(&self) -> &'static str {
        "run_shell"
    }
    fn description(&self) -> &'static str {
        "Execute a shell command on the user's machine. The user can see the command output! Use for tasks that require terminal operations. Always prefer safe, idempotent commands and avoid destructive operations."
    }
    fn required_params(&self) -> &'static [&'static str] {
        &["command"]
    }
    fn params(&self) -> Vec<ParamBuilder> {
        vec![
            ParamBuilder::new("command").type_of("string").description(
                "The exact shell command to execute (sh -c on Unix, cmd /C on Windows)",
            ),
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

        println!("> {}", command);
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
            std::thread::sleep(std::time::Duration::from_millis(50));
        };

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
                if !status.success() {
                    eprintln!("Note: Command exited with non-zero status: {}", status);
                }
                if !combined.is_empty() {
                    println!("{}", combined);
                }
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
