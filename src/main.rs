use anyhow::Result;
use clap::{Parser, Subcommand};

mod history;
use history::History;

mod tools;

mod config;
use config::handle_config_command;

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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(Commands::Config { key, value, global }) = cli.command {
        return handle_config_command(key, value, global);
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
