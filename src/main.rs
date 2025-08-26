use anyhow::{Context, Result};

use clap::{Parser, Subcommand};
use futures::future::{FutureExt, LocalBoxFuture};
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, ChatRole, MessageType},
    LLMProvider,
};
use serde::{Deserialize, Serialize};
use spinoff::{spinners, Color, Spinner};
use std::fs;
use std::path::PathBuf;

mod history;
use history::History;

mod tools;
use tools::ToolsRegistry;

#[derive(Parser)]
#[command(name = "tai")]
#[command(about = "Terminal AI Assistant")]
#[command(disable_help_subcommand = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Skip loading context files
    #[arg(long)]
    nocontext: bool,

    /// Load specific context (if not specified, loads .context.tai from current dir)
    #[arg(long)]
    context: Option<String>,

    /// Clear conversation history
    #[arg(long)]
    clear_history: bool,

    /// The message to send to the AI
    #[arg(trailing_var_arg = true)]
    message: Vec<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Configure tai settings
    Config {
        /// Configuration key to get/set
        key: Option<String>,
        /// Value to set (if not provided, will get the value)
        value: Option<String>,
        /// Set configuration globally instead of locally
        #[arg(long)]
        global: bool,
    },
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct Config {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    anthropic_api_key: Option<String>,
    #[serde(default)]
    global_contexts: Vec<String>,
}

pub struct Session<'a> {
    llm: &'a dyn LLMProvider,
    tools: ToolsRegistry,
    history: Vec<ChatMessage>,
    file_history: History,
    context_added: bool,
}

fn get_git_root() -> Option<PathBuf> {
    std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| PathBuf::from(s.trim()))
            } else {
                None
            }
        })
}

fn find_config_file() -> Option<PathBuf> {
    let current_dir = std::env::current_dir().ok()?;

    // Try current directory first
    let local_config = current_dir.join(".config.tai");
    if local_config.exists() {
        return Some(local_config);
    }

    // Try git root if we're in a git repository
    if let Some(git_root) = get_git_root() {
        let git_config = git_root.join(".config.tai");
        if git_config.exists() {
            return Some(git_config);
        }
    }

    None
}

fn get_global_config_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Failed to get home directory")?;
    let config_dir = home.join(".config").join("tai");
    fs::create_dir_all(&config_dir)?;
    Ok(config_dir)
}

fn load_config() -> Result<Config> {
    let mut config = Config::default();

    // Load global config
    let global_config_dir = get_global_config_dir()?;
    let global_config_path = global_config_dir.join("config.tai");
    if global_config_path.exists() {
        let global_content = fs::read_to_string(&global_config_path)?;
        config = toml::from_str(&global_content)?;
    }

    // Load local config (overrides global)
    if let Some(local_config_path) = find_config_file() {
        let local_content = fs::read_to_string(&local_config_path)?;
        let local_config: Config = toml::from_str(&local_content)?;

        // Merge configs (local overrides global)
        if local_config.model.is_some() {
            config.model = local_config.model;
        }
        if local_config.temperature.is_some() {
            config.temperature = local_config.temperature;
        }
        if local_config.max_tokens.is_some() {
            config.max_tokens = local_config.max_tokens;
        }
        if local_config.anthropic_api_key.is_some() {
            config.anthropic_api_key = local_config.anthropic_api_key;
        }
        if !local_config.global_contexts.is_empty() {
            config.global_contexts = local_config.global_contexts;
        }
    }

    Ok(config)
}

fn save_config(config: &Config, global: bool) -> Result<()> {
    let config_path = if global {
        let global_config_dir = get_global_config_dir()?;
        global_config_dir.join("config.tai")
    } else {
        let current_dir = std::env::current_dir()?;
        if let Some(git_root) = get_git_root() {
            git_root.join(".config.tai")
        } else {
            current_dir.join(".config.tai")
        }
    };

    let content = toml::to_string_pretty(config)?;
    fs::write(&config_path, content)?;

    Ok(())
}

fn find_context_files(context_name: Option<&str>) -> Result<Vec<(String, String)>> {
    let mut contexts = Vec::new();
    let current_dir = std::env::current_dir()?;

    if let Some(name) = context_name {
        // Load specific context from ~/.config/tai/context/
        let global_config_dir = get_global_config_dir()?;
        let context_dir = global_config_dir.join("context");
        let context_file = context_dir.join(format!("{}.context.tai", name));

        if context_file.exists() {
            let content = fs::read_to_string(&context_file)?;
            contexts.push((name.to_string(), content));
        } else {
            eprintln!("Warning: Context '{}' not found", name);
        }
    } else {
        // Auto-load .context.tai from current dir or git root
        let context_file = current_dir.join(".context.tai");
        if context_file.exists() {
            let content = fs::read_to_string(&context_file)?;
            contexts.push(("local".to_string(), content));
        } else if let Some(git_root) = get_git_root() {
            let git_context_file = git_root.join(".context.tai");
            if git_context_file.exists() {
                let content = fs::read_to_string(&git_context_file)?;
                contexts.push(("project".to_string(), content));
            }
        }
    }

    // Load global contexts from config
    let config = load_config().unwrap_or_default();
    let global_config_dir = get_global_config_dir()?;
    let context_dir = global_config_dir.join("context");

    for global_context in &config.global_contexts {
        let context_file = context_dir.join(format!("{}.context.tai", global_context));
        if context_file.exists() {
            let content = fs::read_to_string(&context_file)?;
            contexts.push((format!("global:{}", global_context), content));
        }
    }

    Ok(contexts)
}

fn setup(tools: &ToolsRegistry) -> Result<Box<dyn LLMProvider>> {
    let config = load_config().unwrap_or_default();

    let builder = LLMBuilder::new()
        .max_tokens(config.max_tokens.unwrap_or(1500))
        .temperature(config.temperature.unwrap_or(0.0))
        .stream(false);
    let builder = tools.apply_to_builder(builder);

    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        if !key.is_empty() {
            return builder
                .backend(LLMBackend::Anthropic)
                .api_key(key)
                .model(
                    config
                        .model
                        .as_deref()
                        .unwrap_or("claude-3-5-sonnet-latest"),
                )
                .build()
                .context("Failed to build Anthropic Client");
        }
    } else if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        if !key.is_empty() {
            let mut b = builder
                .backend(LLMBackend::OpenAI)
                .api_key(key)
                .model(config.model.as_deref().unwrap_or("gpt-4o-mini"));
            if let Ok(base) = std::env::var("OPENAI_BASE_URL") {
                if !base.is_empty() {
                    b = b.base_url(base);
                }
            }
            return b.build().context("Failed to build OpenAI Client");
        }
    }

    if let Ok(base) = std::env::var("OLLAMA_BASE_URL") {
        if !base.is_empty() {
            return builder
                .backend(LLMBackend::Ollama)
                .base_url(base)
                .model(config.model.as_deref().unwrap_or("deepseek-r1:8b"))
                .build()
                .context("Failed to build Ollama Client");
        }
    }

    // LM Studio support via OpenAI-compatible endpoint
    if let Ok(base) = std::env::var("LM_STUDIO_BASE_URL") {
        if !base.is_empty() {
            return builder
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "lm-studio".into()))
                .base_url(base)
                .model(config.model.as_deref().unwrap_or("gpt-4o-mini"))
                .build()
                .context("Failed to build LM Studio (OpenAI compat) Client");
        }
    }

    // fallback to local Ollama defaults
    builder
        .backend(LLMBackend::Ollama)
        .model(config.model.as_deref().unwrap_or("deepseek-r1:8b"))
        .build()
        .context("Failed to build Ollama Client")
}

fn handle_config_command(key: Option<String>, value: Option<String>, global: bool) -> Result<()> {
    let mut config = if global {
        // Load only global config
        let global_config_dir = get_global_config_dir()?;
        let global_config_path = global_config_dir.join("config.tai");
        if global_config_path.exists() {
            let content = fs::read_to_string(&global_config_path)?;
            toml::from_str(&content)?
        } else {
            Config::default()
        }
    } else {
        load_config()?
    };

    match (key, value) {
        (None, None) => {
            // List all config values with elegant formatting
            println!("Configuration:");
            println!("  model:");
            match &config.model {
                Some(m) => println!("    {}", m),
                None => println!("    <not set>"),
            }
            println!("  temperature:");
            match &config.temperature {
                Some(t) => println!("    {}", t),
                None => println!("    <not set>"),
            }
            println!("  max_tokens:");
            match &config.max_tokens {
                Some(mt) => println!("    {}", mt),
                None => println!("    <not set>"),
            }
            println!("  anthropic_api_key:");
            match &config.anthropic_api_key {
                Some(_) => println!("    ***"),
                None => println!("    <not set>"),
            }
            println!("  global_contexts:");
            if config.global_contexts.is_empty() {
                println!("    <none>");
            } else {
                for context in &config.global_contexts {
                    println!("    - {}", context);
                }
            }
        }
        (Some(key), None) => {
            // Get specific value
            match key.as_str() {
                "model" => match &config.model {
                    Some(m) => println!("{}", m),
                    None => println!("<not set>"),
                },
                "temperature" => match &config.temperature {
                    Some(t) => println!("{}", t),
                    None => println!("<not set>"),
                },
                "max_tokens" => match &config.max_tokens {
                    Some(mt) => println!("{}", mt),
                    None => println!("<not set>"),
                },
                "anthropic_api_key" => match &config.anthropic_api_key {
                    Some(_) => println!("***"),
                    None => println!("<not set>"),
                },
                "global_contexts" => {
                    if config.global_contexts.is_empty() {
                        println!("<none>");
                    } else {
                        for context in &config.global_contexts {
                            println!("{}", context);
                        }
                    }
                }
                _ => anyhow::bail!("Unknown config key: {}", key),
            }
        }
        (Some(key), Some(value)) => {
            // Set value
            match key.as_str() {
                "model" => config.model = Some(value),
                "temperature" => config.temperature = Some(value.parse()?),
                "max_tokens" => config.max_tokens = Some(value.parse()?),
                "anthropic_api_key" => config.anthropic_api_key = Some(value),
                "global_contexts" => {
                    let requested_contexts: Vec<String> =
                        value.split(',').map(|s| s.trim().to_string()).collect();
                    let global_config_dir = get_global_config_dir()?;
                    let context_dir = global_config_dir.join("context");

                    let mut valid_contexts = Vec::new();
                    let mut missing_contexts = Vec::new();

                    for context in requested_contexts {
                        let context_file = context_dir.join(format!("{}.context.tai", context));
                        if context_file.exists() {
                            valid_contexts.push(context);
                        } else {
                            missing_contexts.push(context);
                        }
                    }

                    if !missing_contexts.is_empty() {
                        eprintln!("Warning: The following context files do not exist:");
                        for missing in &missing_contexts {
                            eprintln!("  - {}.context.tai", missing);
                        }
                    }

                    config.global_contexts = valid_contexts;
                }
                _ => anyhow::bail!("Unknown config key: {}", key),
            }
            save_config(&config, global)?;
            println!("Configuration updated");
        }
        _ => unimplemented!(),
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle config subcommand
    if let Some(Commands::Config { key, value, global }) = cli.command {
        return handle_config_command(key, value, global);
    }

    // Handle clear history
    if cli.clear_history {
        History::clear()?;
        println!("History cleared");
        return Ok(());
    }

    // Handle empty input - read from stdin
    let user_input = if cli.message.is_empty() {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout()).context("Failed to flush stdout")?;

        let mut input = String::new();
        loop {
            let mut line = String::new();
            match std::io::stdin().read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    input.push_str(&line);
                    // Check if this looks like the end of input (empty line or just whitespace)
                    if line.trim().is_empty() && !input.trim().is_empty() {
                        break;
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }

        if input.trim().is_empty() {
            std::process::exit(0);
        }

        input.trim().to_string()
    } else {
        cli.message.join(" ")
    };

    let tools = ToolsRegistry::with_default();
    let llm = setup(&tools)?;
    let mut session = Session::new(llm.as_ref(), tools);

    // Load context files if not disabled
    let contexts = if cli.nocontext {
        Vec::new()
    } else {
        find_context_files(cli.context.as_deref()).unwrap_or_else(|e| {
            eprintln!("Warning: Failed to load context files: {}", e);
            Vec::new()
        })
    };

    if !contexts.is_empty() {
        let context_names: Vec<&str> = contexts.iter().map(|(name, _)| name.as_str()).collect();
        println!("Using context files: [{}]", context_names.join(", "));
    }

    // Execute first step
    session.step(&user_input, &contexts).await?;
    Ok(())
}

impl<'a> Session<'a> {
    pub fn new(llm: &'a dyn LLMProvider, tools: ToolsRegistry) -> Self {
        let file_history = History::load().unwrap_or_default();

        Self {
            llm,
            tools,
            history: Vec::new(),
            file_history,
            context_added: false,
        }
    }

    pub fn step<'b>(
        &'b mut self,
        input: &'b str,
        contexts: &'b [(String, String)],
    ) -> LocalBoxFuture<'b, Result<()>> {
        async move {
            let mut spinner = Spinner::new(spinners::Dots, "Thinking...", Color::Blue);

            if self.history.is_empty() {
                let system_prompt = self.build_system_prompt(contexts);
                self.history.push(ChatMessage {
                    role: ChatRole::Assistant,
                    message_type: MessageType::Text,
                    content: system_prompt,
                });
            }

            self.history.push(ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: input.to_string(),
            });

            loop {
                let response = self
                    .llm
                    .chat_with_tools(&self.history, self.llm.tools())
                    .await
                    .context("Chat failed")?;

                if let Some(calls) = response.tool_calls() {
                    if !calls.is_empty() {
                        // Stop/clear spinner before interactive tool handling
                        spinner.clear();

                        self.history.push(
                            ChatMessage::assistant()
                                .tool_use(calls.clone())
                                .content("")
                                .build(),
                        );

                        let mut tool_results = Vec::new();
                        for call in &calls {
                            match self.tools.handle_tool_call(call) {
                                Ok(result) => {
                                    tool_results.push(llm::ToolCall {
                                        id: call.id.clone(),
                                        call_type: "function".to_string(),
                                        function: llm::FunctionCall {
                                            name: call.function.name.clone(),
                                            arguments: serde_json::to_string(&result)
                                                .unwrap_or("{}".into()),
                                        },
                                    });
                                }
                                Err(e) => {
                                    tool_results.push(llm::ToolCall {
                                        id: call.id.clone(),
                                        call_type: "function".to_string(),
                                        function: llm::FunctionCall {
                                            name: call.function.name.clone(),
                                            arguments: serde_json::to_string(
                                                &serde_json::json!({"error": e.to_string()}),
                                            )
                                            .unwrap_or("{}".into()),
                                        },
                                    });
                                }
                            }
                        }

                        self.history.push(
                            ChatMessage::user()
                                .tool_result(tool_results)
                                .content("")
                                .build(),
                        );

                        self.history.push(ChatMessage {
                            role: ChatRole::Assistant,
                            message_type: MessageType::Text,
                            content: "The user already saw the exact command output in their terminal. Do not summarize it. If another action is needed, call a tool. If nothing else is needed, reply only with: OK".to_string(),
                        });

                        // Restart spinner for the next LLM step
                        spinner = Spinner::new(spinners::Dots, "Thinking...", Color::Blue);
                        continue;
                    }
                }

                spinner.clear();

                let text = response.text().unwrap_or_else(|| response.to_string());
                self.file_history
                    .add_entry(input.to_string(), text.clone())?;
                println!("{}", text);
                break;
            }
            Ok(())
        }
        .boxed_local()
    }

    fn build_system_prompt(&mut self, contexts: &[(String, String)]) -> String {
        let relevant_entries = self.file_history.get_relevant_entries();

        let mut history_context = String::new();
        if !relevant_entries.is_empty() {
            history_context.push_str("\nHere are some of your previous interactions (these may not be related to the current query and are just for reference):\n\n");

            for (idx, (entry, age)) in relevant_entries.iter().enumerate() {
                let minutes = age.num_minutes();
                history_context.push_str(&format!(
                    "Interaction {} (from {} minutes ago):\n",
                    idx + 1,
                    minutes
                ));
                history_context.push_str(&format!("User: {}\n", entry.user_input));
                history_context.push_str(&format!("Assistant: {}\n\n", entry.llm_response));
            }
        }

        let mut context_section = String::new();
        if !contexts.is_empty() && !self.context_added {
            context_section.push_str("\n## Additional Context\n\n");
            for (name, content) in contexts {
                context_section.push_str(&format!("### Context from {}\n\n{}\n\n", name, content));
            }
            self.context_added = true;
        }

        format!(
            r#"You are an AI assistant running in a terminal that can call tools to operate on the user's machine.
Your goal is to help the user achieve their task efficiently and safely.

System rules:
- If the user asks you to perform a terminal task, call the run_shell tool with the exact command to execute. Prefer pipes over multiple sequential commands when possible.
- Keep commands non-interactive, idempotent, and safe by default. Avoid destructive operations unless the user explicitly requests them.
- When executing a terminal command the user can already see the output of the command. Do NOT summarize or restate the command's output.
- If the user is asking about a command (explanatory), answer concisely and include a one-line example, then a brief explanation of key flags.
- After running a command via the tool, use its output to decide next steps. You may call tools multiple times until the task is complete.
- Do not invent file paths or secrets. Never print sensitive values.

{context_section}{history_context}"#
        )
    }
}

// Extension for History to add clear functionality
impl History {
    pub fn clear() -> Result<()> {
        let history_file = dirs::home_dir()
            .context("Failed to get home directory")?
            .join(".tai_history.json");

        if history_file.exists() {
            fs::remove_file(&history_file)?;
        }

        Ok(())
    }
}
