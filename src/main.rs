use anyhow::{Context, Result};
use arboard::Clipboard;
use clap::{Parser, Subcommand};
use futures::future::{BoxFuture, FutureExt};
use rllm::{
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

#[derive(Debug, Deserialize, Serialize)]
struct Response {
    pub r#type: String,
    pub result: String,
    pub r#continue: bool,
}

pub struct Session<'a> {
    llm: &'a Box<dyn LLMProvider>,
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

fn setup() -> Result<Box<dyn LLMProvider>> {
    let config = load_config().unwrap_or_default();

    let builder = LLMBuilder::new()
        .max_tokens(config.max_tokens.unwrap_or(1500))
        .temperature(config.temperature.unwrap_or(0.0))
        .stream(false);

    // Try environment variable first, then config
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .filter(|key| !key.is_empty())
        .or(config.anthropic_api_key);

    if let Some(key) = api_key {
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

    // fallback to ollama using deepseek-r1
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

    let llm = setup()?;
    let mut session = Session::new(&llm);

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
    pub fn new(llm: &'a Box<dyn LLMProvider>) -> Self {
        let file_history = History::load().unwrap_or_default();

        Self {
            llm,
            history: Vec::new(),
            file_history,
            context_added: false,
        }
    }

    pub fn step<'b>(
        &'b mut self,
        input: &'b str,
        contexts: &'b [(String, String)],
    ) -> BoxFuture<'b, Result<()>> {
        async move {
            let mut spinner = Spinner::new(spinners::Dots, "Thinking...", Color::Blue);

            // Build the prompt with history context
            let prompt = self.build_prompt(input, contexts);

            self.history.push(ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: prompt,
            });

            let text = self
                .llm
                .chat(&self.history)
                .await
                .context("Chat failed")?
                .to_string();
            self.history.push(ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: text.clone(),
            });

            // Save to history file
            self.file_history
                .add_entry(input.to_string(), text.clone())?;

            let result: Response =
                serde_json::from_str(&text).context("Failed to parse response")?;

            spinner.clear();

            match result.r#type.as_str() {
                "command" => self.handle_command(&result).await,
                "answer" => self.handle_answer(&result),
                _ => anyhow::bail!("Unknown response type: {}", result.r#type),
            }
        }
        .boxed()
    }

    fn build_prompt(&mut self, input: &str, contexts: &[(String, String)]) -> String {
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

        // Add context files only to the first message
        let mut context_section = String::new();
        if !contexts.is_empty() && !self.context_added {
            context_section.push_str("\n## Additional Context\n\n");
            for (name, content) in contexts {
                context_section.push_str(&format!("### Context from {}\n\n{}\n\n", name, content));
            }
            self.context_added = true;
        }

        format!(
            r#"You are an ai assistant that is running on the terminal.
Your goal is to assist the user in reaching his goal.

These are the rules that you have to obey:
- If the user is asking a question about a specific terminal command, give a very brief example of the command and explain each parameter.
- If the user is asking you (imperatively) to do something, just write down the exact command and nothing else. Only output the exact command to execute the given task without any explanation whatsoever.

The response must be a valid json and only a json. The format of the json has to be:
{{
    "type": "command",
    "result": "curl -I https://google.com",
    "continue": false,
}}

The "type" can either be "command" (to execute a command) or an "answer".
The "result" is the actual command or answer.

- If the "result" is a "command" and you need it's output to continue to solve the given task respond with "continue" set to true.
- If this is the last command of the chain respond with "continue" set to false.
- If the result can be achieved by piping don't use "continue" but provide the command with pipes directly - be extra cautious when using case-sensitive grep.
- If you need to provide an answer that contains an example command without actual proper parameters still use the "type" "answer".
{context_section}{history_context}
The user asks:
{input}"#
        )
    }

    fn handle_command<'b>(&'b mut self, response: &'b Response) -> BoxFuture<'b, Result<()>> {
        async move {
            println!("> {}", response.result);
            print!("Do you want to execute this command? [Y/n/c] ");
            std::io::Write::flush(&mut std::io::stdout()).context("Failed to flush stdout")?;

            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .context("Failed to read user input")?;

            let choice = input.trim().to_lowercase();

            if choice == "c" {
                // Copy to clipboard
                match Clipboard::new() {
                    Ok(mut clipboard) => {
                        if let Err(e) = clipboard.set_text(&response.result) {
                            eprintln!("Failed to copy to clipboard: {}", e);
                        } else {
                            println!("Command copied to clipboard");
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to access clipboard: {}", e);
                    }
                }
                return Ok(());
            }

            if choice == "n" {
                println!("Command execution cancelled");
                return Ok(());
            }

            print!("\x1B[1A\x1B[2K\r"); // Move up 1 line and clear it
            print!("\x1B[2K\r"); // Clear current line

            let mut spinner = Spinner::new(spinners::Dots, "Executing...", Color::Blue);
            let mut spinning = true;

            // Check if command contains sudo and clear spinner immediately
            // This ensures password input will work
            let contains_sudo = response.result.contains("sudo ");
            if contains_sudo && !cfg!(target_os = "windows") {
                spinner.clear();
                spinning = false;
            }

            let output = if cfg!(target_os = "windows") {
                std::process::Command::new("cmd")
                    .args(["/C", &response.result])
                    .output()
            } else {
                std::process::Command::new("sh")
                    .args(["-c", &response.result])
                    .output()
            }
            .context("Failed to execute command")?;

            // Combine stdout and stderr
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            // Create combined output
            let combined_output = if stderr.is_empty() {
                stdout.to_string()
            } else if stdout.is_empty() {
                stderr.to_string()
            } else {
                format!("{}\n{}", stdout, stderr)
            };

            if spinning {
                spinner.clear();
            }

            // Print combined output regardless of success
            println!("{}", combined_output.trim_end());

            // If command failed, show a warning but continue
            if !output.status.success() {
                eprintln!(
                    "Note: Command exited with non-zero status: {}",
                    output.status
                );
            }

            if response.r#continue {
                return self
                    .step(
                        &format!(
                            r#"The output of {} is:

{}"#,
                            response.result, stdout
                        ),
                        &[],
                    )
                    .await;
            }

            Ok(())
        }
        .boxed()
    }

    fn handle_answer(&mut self, response: &Response) -> Result<()> {
        println!("{}", response.result);
        Ok(())
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
