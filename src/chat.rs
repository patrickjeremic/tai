use anyhow::{Context, Result};
use futures::future::{FutureExt, LocalBoxFuture};
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, ChatRole, MessageType},
    LLMProvider,
};
use spinoff::{spinners, Color, Spinner};

use crate::chat_render::render_markdown_to_terminal;
use crate::config::{find_context_files, load_config};
use crate::history::History;
use crate::tools::ToolsRegistry;

pub struct Session<'a> {
    llm: &'a dyn LLMProvider,
    tools: ToolsRegistry,
    history: Vec<ChatMessage>,
    file_history: History,
    context_added: bool,
}

pub fn setup(tools: &ToolsRegistry) -> Result<Box<dyn LLMProvider>> {
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

    builder
        .backend(LLMBackend::Ollama)
        .model(config.model.as_deref().unwrap_or("deepseek-r1:8b"))
        .build()
        .context("Failed to build Ollama Client")
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

{context_section}{history_context}"#
        )
    }
}

pub async fn run_chat(nocontext: bool, context: Option<String>, user_input: String) -> Result<()> {
    let tools = ToolsRegistry::with_default();
    let llm = setup(&tools)?;
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
