use anyhow::{Context, Result};
use futures::future::{FutureExt, LocalBoxFuture};
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, ChatRole, MessageType},
    LLMProvider,
};
use nu_ansi_term::{Color as NuColor, Style};
use serde_json::Value as JsonValue;
use spinoff::{spinners, Color, Spinner};

use crate::chat_render::render_markdown_to_terminal;
use crate::config::{find_context_files, load_config, select_effective_provider};
use crate::history::History;
use crate::tools::ToolsRegistry;

fn is_sensitive_key(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    let hints = [
        "key",
        "token",
        "secret",
        "password",
        "passwd",
        "auth",
        "authorization",
        "cookie",
        "api_key",
        "apikey",
        "access_key",
        "session",
        "bearer",
    ];
    hints.iter().any(|h| k.contains(h))
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut out = s.chars().take(max).collect::<String>();
        out.push('…');
        out
    }
}

fn render_value_for_kv(key: &str, v: &JsonValue) -> String {
    if is_sensitive_key(key) {
        return "***".to_string();
    }
    match v {
        JsonValue::String(s) => truncate_str(s, 160),
        JsonValue::Number(n) => n.to_string(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Null => "null".to_string(),
        JsonValue::Array(arr) => {
            if arr.is_empty() {
                "[]".to_string()
            } else if arr.len() <= 5
                && arr
                    .iter()
                    .all(|it| it.is_string() || it.is_number() || it.is_boolean() || it.is_null())
            {
                let parts: Vec<String> = arr
                    .iter()
                    .map(|it| match it {
                        JsonValue::String(s) => format!("\"{}\"", truncate_str(s, 60)),
                        JsonValue::Number(n) => n.to_string(),
                        JsonValue::Bool(b) => b.to_string(),
                        JsonValue::Null => "null".to_string(),
                        _ => "…".to_string(),
                    })
                    .collect();
                format!("[{}]", parts.join(", "))
            } else {
                format!("[{} items]", arr.len())
            }
        }
        JsonValue::Object(obj) => {
            format!("{{{} keys}}", obj.len())
        }
    }
}

fn format_tool_params(args_raw: &str) -> String {
    let parsed = serde_json::from_str::<JsonValue>(args_raw);
    match parsed {
        Ok(JsonValue::Object(map)) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let key_style = Style::new().bold().fg(NuColor::LightGreen);
            let mut out = String::new();
            for k in keys {
                let v = &map[k];
                match v {
                    JsonValue::Object(nested) => {
                        let header = key_style.paint(k.as_str()).to_string();
                        out.push_str(&format!("  {}:\n", header));
                        let mut sub_keys: Vec<&String> = nested.keys().collect();
                        sub_keys.sort();
                        for sk in sub_keys {
                            let sv = &nested[sk];
                            let sks = key_style.paint(sk.as_str()).to_string();
                            let val = render_value_for_kv(sk, sv);
                            out.push_str(&format!("    {}: {}\n", sks, val));
                        }
                    }
                    _ => {
                        let ks = key_style.paint(k.as_str()).to_string();
                        let val = render_value_for_kv(k, v);
                        out.push_str(&format!("  {}: {}\n", ks, val));
                    }
                }
            }
            out
        }
        Ok(other) => serde_json::to_string_pretty(&other).unwrap_or_else(|_| args_raw.to_string()),
        Err(_) => args_raw.to_string(),
    }
}

pub struct Session<'a> {
    llm: &'a dyn LLMProvider,
    tools: ToolsRegistry,
    history: Vec<ChatMessage>,
    file_history: History,
    context_added: bool,
}

pub fn setup(tools: &ToolsRegistry) -> Result<Box<dyn LLMProvider>> {
    let cfg = load_config().unwrap_or_default();
    let eff = select_effective_provider(&cfg);

    let mut builder = LLMBuilder::new()
        .stream(false);
    let is_openai_gpt5 = eff.name == "openai" && (eff.model.starts_with("gpt-5") || eff.model.starts_with("gpt-5-"));
    if !is_openai_gpt5 {
        builder = builder.temperature(eff.temperature);
        builder = builder.max_tokens(eff.max_tokens);
    }
    let builder = tools.apply_to_builder(builder);

    match eff.name.as_str() {
        "anthropic" => {
            let key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
            builder
                .backend(LLMBackend::Anthropic)
                .api_key(key)
                .model(&eff.model)
                .build()
                .context("Failed to build Anthropic Client")
        }
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
            let mut b = builder
                .backend(LLMBackend::OpenAI)
                .api_key(key)
                .model(&eff.model);
            if let Some(base) = eff.base_url_or_host.clone() {
                b = b.base_url(base);
            }
            b.build().context("Failed to build OpenAI Client")
        }
        "ollama" => {
            let mut b = builder.backend(LLMBackend::Ollama).model(&eff.model);
            if let Some(host) = eff.base_url_or_host.clone() {
                b = b.base_url(host);
            }
            b.build().context("Failed to build Ollama Client")
        }
        "lmstudio" => {
            let key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "lm-studio".into());
            let mut b = builder
                .backend(LLMBackend::OpenAI)
                .api_key(key)
                .model(&eff.model);
            if let Some(base) = eff.base_url_or_host.clone() {
                b = b.base_url(base);
            }
            b.build()
                .context("Failed to build LM Studio (OpenAI compat) Client")
        }
        _ => builder
            .backend(LLMBackend::Ollama)
            .model(&eff.model)
            .build()
            .context("Failed to build Ollama Client"),
    }
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
                        spinner.clear();

                        self.history.push(
                            ChatMessage::assistant()
                                .tool_use(calls.clone())
                                .content("")
                                .build(),
                        );

                        let mut tool_results = Vec::new();
                        for call in &calls {
                            let name = &call.function.name;
                            let args_raw = &call.function.arguments;
                            let formatted = format_tool_params(args_raw);
                            let header = Style::new()
                                .bold()
                                .fg(NuColor::LightCyan)
                                .paint("Tool call");
                            let name_col = Style::new().bold().fg(NuColor::Yellow).paint(name);
                            println!("{}: {}", header, name_col);
                            let args_label = Style::new().fg(NuColor::Green).paint("params");
                            println!("{}:\n{}", args_label, formatted);

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

                        let has_shell = calls.iter().any(|c| c.function.name == "run_shell");
                        if has_shell {
                            self.history.push(ChatMessage {
                                role: ChatRole::Assistant,
                                message_type: MessageType::Text,
                                content: "Summarize the results of the terminal command succinctly and proceed with any next steps to complete the user's request. If the command output already satisfies the request, provide the final answer concisely.".to_string(),
                            });
                        } else {
                            self.history.push(ChatMessage {
                                role: ChatRole::Assistant,
                                message_type: MessageType::Text,
                                content: "Use the tool outputs above to answer the user directly. Provide a concise summary or the requested information. If more actions are needed, call a tool.".to_string(),
                            });
                        }

                        spinner = Spinner::new(spinners::Dots, "Thinking...", Color::Blue);
                        continue;
                    }
                }

                spinner.clear();

                let text = response.text().unwrap_or_else(|| response.to_string());
                let rendered = render_markdown_to_terminal(&text);
                self.file_history
                    .add_entry(input.to_string(), text.clone())?;
                println!("{}", rendered);
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
- When you include code, always use fenced code blocks with a language identifier like ```rust, ```bash, ```python, etc. Avoid plain triple backticks without a language.
- Always respond using Markdown syntax.

{context_section}{history_context}"#
        )
    }
}

pub async fn run_chat(nocontext: bool, context: Option<String>, user_input: String) -> Result<()> {
    let tools = ToolsRegistry::with_default();
    let cfg = load_config().unwrap_or_default();
    let eff = select_effective_provider(&cfg);
    let llm = setup(&tools)?;
    println!(
        "Using provider {} (model: {}{})",
        eff.name,
        eff.model,
        eff.base_url_or_host
            .as_ref()
            .map(|u| format!("; base: {}", u))
            .unwrap_or_default()
    );
    let mut session = Session::new(llm.as_ref(), tools);

    let contexts = if nocontext {
        Vec::new()
    } else {
        find_context_files(context.as_deref()).unwrap_or_else(|e| {
            eprintln!("Warning: Failed to load context files: {}", e);
            Vec::new()
        })
    };

    if !contexts.is_empty() {
        let context_names: Vec<&str> = contexts.iter().map(|(name, _)| name.as_str()).collect();
        println!("Using context files: [{}]", context_names.join(", "));
    }

    session.step(&user_input, &contexts).await
}
