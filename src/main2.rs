use anyhow::{Context, Result};
use indicatif::ProgressBar;
use rllm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, ChatRole},
    LLMProvider,
};
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <message>", args[0]);
        std::process::exit(1);
    }
    let input = &args[1..].join(" ");

    let llm = LLMBuilder::new()
        .backend(LLMBackend::Anthropic) // or LLMBackend::Anthropic, LLMBackend::Ollama, LLMBackend::DeepSeek, LLMBackend::XAI, LLMBackend::Phind ...
        .api_key(std::env::var("ANTHROPIC_API_KEY").context("Failed to get ANTHROPIC_API_KEY")?)
        //.model("claude-3-5-sonnet-20240620") // or model("claude-3-5-sonnet-20240620") or model("grok-2-latest") or model("deepseek-chat") or model("llama3.1") or model("Phind-70B") ...
        .max_tokens(1500)
        .temperature(0.0)
        //.system("You are a helpful assistant.")
        .stream(false)
        .build()
        .context("Failed to build LLM")?;

    let messages = vec![ChatMessage {
        role: ChatRole::User,
        content: format!(
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
{input}"#
        ),
    }];

    // TODO: stremaing output
    // TODO: input from stdin apart from args
    //
    // instructions: if it is a question do this, if it imperative do that

    let bar = ProgressBar::new_spinner();
    let text = llm.chat(&messages).context("Chat failed")?;
    bar.finish();

    let result: Response = serde_json::from_str(&text).context("Failed to parse response")?;
    match result.r#type.as_str() {
        "command" => handle_command(llm, messages.clone(), &result)?,
        "answer" => handle_answer(&result)?,
        _ => anyhow::bail!("Unknown response type: {}", result.r#type),
    }
    Ok(())
}

#[derive(Debug, Deserialize, Serialize)]
struct Response {
    pub r#type: String,
    pub result: String,
    pub r#continue: bool,
}

fn handle_command(
    llm: Box<dyn LLMProvider>,
    mut history: Vec<ChatMessage>,
    response: &Response,
) -> Result<()> {
    println!("$ {}", response.result);
    print!("Do you want to execute this command? [y/N] ");
    std::io::Write::flush(&mut std::io::stdout()).context("Failed to flush stdout")?;

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("Failed to read user input")?;

    if input.trim().to_lowercase() != "y" {
        println!("Command execution cancelled");
        return Ok(());
    }

    println!("Executing command...");
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

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Command failed: {}", stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);

    if response.r#continue {
        println!("NEXT LOOP");

        // append message
        history.push(ChatMessage {
            role: ChatRole::Assistant,
            content: serde_json::to_string_pretty(&response).unwrap(),
        });
        history.push(ChatMessage {
            role: ChatRole::User,
            content: format!(
                r#"The output of {} is:

        {}"#,
                response.result, stdout
            ),
        });

        let bar = ProgressBar::new_spinner();
        let text = llm.chat(&history).context("Chat failed")?;
        bar.finish();

        let result: Response = serde_json::from_str(&text).context("Failed to parse response")?;
        match result.r#type.as_str() {
            "command" => handle_command(llm, history.clone(), &result)?,
            "answer" => handle_answer(&result)?,
            _ => anyhow::bail!("Unknown response type: {}", result.r#type),
        }
    }

    Ok(())
}

fn handle_answer(response: &Response) -> Result<()> {
    // TODO: Implement answer handling
    println!("{}", response.result);
    Ok(())
}
