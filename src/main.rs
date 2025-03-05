use anyhow::{Context, Result};
use futures::future::{BoxFuture, FutureExt};
use rllm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, ChatRole},
    LLMProvider,
};
use serde::{Deserialize, Serialize};
use spinoff::{spinners, Color, Spinner};

fn setup() -> Result<Box<dyn LLMProvider>> {
    let builder = LLMBuilder::new()
        .max_tokens(1500)
        .temperature(0.0)
        .stream(false);

    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        if !key.is_empty() {
            return builder
                .backend(LLMBackend::Anthropic)
                .api_key(
                    std::env::var("ANTHROPIC_API_KEY")
                        .context("Failed to get ANTHROPIC_API_KEY")?,
                )
                .model("claude-3-7-sonnet-latest")
                .build()
                .context("Failed to build Anthropic Client");
        }
    }

    // fallback to ollama using deepseek-r1
    builder
        .backend(LLMBackend::Ollama)
        .model("deepseek-r1:8b")
        .build()
        .context("Failed to build Ollama Client")
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <message>", args[0]);
        std::process::exit(1);
    }
    let user_input = &args[1..].join(" ");

    let llm = setup()?;
    let mut session = Session::new(&llm);

    // execute first step
    session.step(&format!(
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

The user asks:
{user_input}"#
            )).await?;
    Ok(())
}

pub struct Session<'a> {
    llm: &'a Box<dyn LLMProvider>,
    history: Vec<ChatMessage>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Response {
    pub r#type: String,
    pub result: String,
    pub r#continue: bool,
}

impl<'a> Session<'a> {
    pub fn new(llm: &'a Box<dyn LLMProvider>) -> Self {
        Self {
            llm,
            history: Vec::new(),
        }
    }

    pub fn step<'b>(&'b mut self, input: &'b str) -> BoxFuture<'b, Result<()>> {
        async move {
            let mut spinner = Spinner::new(spinners::Dots, "Thinking...", Color::Blue);

            self.history.push(ChatMessage {
                role: ChatRole::User,
                content: input.to_string(),
            });

            let text = self.llm.chat(&self.history).await.context("Chat failed")?;
            self.history.push(ChatMessage {
                role: ChatRole::Assistant,
                content: text.clone(),
            });

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

    fn handle_command<'b>(&'b mut self, response: &'b Response) -> BoxFuture<'b, Result<()>> {
        async move {
            println!("> {}", response.result);
            print!("Do you want to execute this command? [Y/n] ");
            std::io::Write::flush(&mut std::io::stdout()).context("Failed to flush stdout")?;

            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .context("Failed to read user input")?;

            if input.trim().to_lowercase() == "n" {
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
                    .step(&format!(
                        r#"The output of {} is:

{}"#,
                        response.result, stdout
                    ))
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
