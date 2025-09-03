use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use llm::builder::{FunctionBuilder, LLMBuilder, ParamBuilder};
use llm::ToolCall;

mod dir;
mod fetch;
mod file;
mod shell;

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

    /// Format and print the result of this tool execution.
    /// Default implementation prints JSON, tools can override for custom formatting.
    fn print_result(&self, result: &Value) {
        use nu_ansi_term::{Color as NuColor, Style};
        let result_label = Style::new().fg(NuColor::LightMagenta).paint("result");
        let pretty = serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".into());
        println!("{}:\n{}", result_label, pretty);
    }
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
        reg.register(Box::new(file::ReadFileTool));
        reg.register(Box::new(file::WriteFileTool));
        reg.register(Box::new(file::PatchFileTool));
        reg.register(Box::new(dir::ListDirTool));
        reg.register(Box::new(dir::StatTool));
        reg.register(Box::new(dir::GlobTool));
        reg.register(Box::new(file::GrepTool));
        reg.register(Box::new(shell::ShellCommandTool));
        reg.register(Box::new(fetch::FetchUrlTool));
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
    pub fn handle_tool_call(&self, call: &ToolCall) -> Result<(Value, &dyn Tool)> {
        let name = &call.function.name;
        let args: Value = serde_json::from_str(&call.function.arguments)
            .with_context(|| format!("Failed parsing tool args for {}", name))?;
        let tool = self
            .find(name)
            .ok_or_else(|| anyhow!("Unknown tool: {}", name))?;
        let result = tool.execute_blocking(args)?;
        Ok((result, tool))
    }
}
