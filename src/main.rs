use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};

mod history;
use history::History;

mod tools;

mod config;
use config::{
    handle_config_command, handle_config_provider_auto, handle_config_provider_list,
    handle_config_provider_set, handle_config_provider_show, handle_config_provider_update,
};

mod chat;
mod chat_render;

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
    Config(ConfigCommand),
}

#[derive(Args)]
struct ConfigCommand {
    #[command(subcommand)]
    command: Option<ConfigSubcommand>,

    /// Legacy get/set still supported for global_contexts only
    key: Option<String>,
    value: Option<String>,
    #[arg(long)]
    global: bool,
}

#[derive(Subcommand)]
enum ConfigSubcommand {
    /// Provider management
    Provider(ProviderCmd),
    /// Show or set legacy values (global_contexts only)
    Legacy,
    /// Provider-specific settings
    #[command(name = "anthropic")]
    Anthropic(ProviderSettingsArgs),
    #[command(name = "openai")]
    OpenAI(OpenAISettingsArgs),
    #[command(name = "ollama")]
    Ollama(OllamaSettingsArgs),
    #[command(name = "lmstudio")]
    LMStudio(LMStudioSettingsArgs),
}

#[derive(Args)]
struct ProviderCmd {
    #[command(subcommand)]
    cmd: ProviderSub,
}

#[derive(Subcommand)]
enum ProviderSub {
    /// List providers and availability
    List,
    /// Set active provider
    Set { provider: ProviderChoice },
    /// Auto mode (clear active)
    Auto,
    /// Show effective settings for provider
    Show { provider: ProviderChoice },
}

#[derive(Clone, ValueEnum)]
enum ProviderChoice {
    Anthropic,
    Openai,
    Ollama,
    Lmstudio,
}

impl ProviderChoice {
    fn as_str(&self) -> &'static str {
        match self {
            ProviderChoice::Anthropic => "anthropic",
            ProviderChoice::Openai => "openai",
            ProviderChoice::Ollama => "ollama",
            ProviderChoice::Lmstudio => "lmstudio",
        }
    }
}

#[derive(Args)]
struct ProviderSettingsArgs {
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    max_tokens: Option<u32>,
}

#[derive(Args)]
struct OpenAISettingsArgs {
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    max_tokens: Option<u32>,
}

#[derive(Args)]
struct OllamaSettingsArgs {
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    host: Option<String>,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    max_tokens: Option<u32>,
}

#[derive(Args)]
struct LMStudioSettingsArgs {
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    max_tokens: Option<u32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(Commands::Config(cfg)) = &cli.command {
        if let Some(sub) = &cfg.command {
            match sub {
                ConfigSubcommand::Provider(p) => match &p.cmd {
                    ProviderSub::List => return handle_config_provider_list(),
                    ProviderSub::Set { provider } => {
                        return handle_config_provider_set(provider.as_str())
                    }
                    ProviderSub::Auto => return handle_config_provider_auto(),
                    ProviderSub::Show { provider } => {
                        return handle_config_provider_show(provider.as_str())
                    }
                },
                ConfigSubcommand::Anthropic(args) => {
                    return handle_config_provider_update(
                        "anthropic",
                        args.model.clone(),
                        None,
                        None,
                        args.temperature,
                        args.max_tokens,
                    );
                }
                ConfigSubcommand::OpenAI(args) => {
                    return handle_config_provider_update(
                        "openai",
                        args.model.clone(),
                        args.base_url.clone(),
                        None,
                        args.temperature,
                        args.max_tokens,
                    );
                }
                ConfigSubcommand::Ollama(args) => {
                    return handle_config_provider_update(
                        "ollama",
                        args.model.clone(),
                        None,
                        args.host.clone(),
                        args.temperature,
                        args.max_tokens,
                    );
                }
                ConfigSubcommand::LMStudio(args) => {
                    return handle_config_provider_update(
                        "lmstudio",
                        args.model.clone(),
                        args.base_url.clone(),
                        None,
                        args.temperature,
                        args.max_tokens,
                    );
                }
                ConfigSubcommand::Legacy => {}
            }
        }
        return handle_config_command(cfg.key.clone(), cfg.value.clone(), cfg.global);
    }

    if cli.clear_history {
        History::clear()?;
        println!("History cleared");
        return Ok(());
    }

    let user_input = if cli.message.is_empty() {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout())?;
        let mut input = String::new();
        loop {
            let mut line = String::new();
            match std::io::stdin().read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    input.push_str(&line);
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

    chat::run_chat(cli.nocontext, cli.context, user_input).await
}
