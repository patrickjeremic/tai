use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct Config {
    #[serde(default)]
    pub core: CoreConfig,
    #[serde(default)]
    pub providers: ProvidersConfig,
    #[serde(default)]
    pub global_contexts: Vec<String>,

    #[serde(default, skip_serializing)]
    pub model: Option<String>,
    #[serde(default, skip_serializing)]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing)]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing)]
    pub anthropic_api_key: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct CoreConfig {
    #[serde(default)]
    pub active_provider: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct ProvidersConfig {
    #[serde(default)]
    pub anthropic: AnthropicConfig,
    #[serde(default)]
    pub openai: OpenAIConfig,
    #[serde(default)]
    pub ollama: OllamaConfig,
    #[serde(default)]
    pub lmstudio: LMStudioConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ProviderCommon {
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}


#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct AnthropicConfig {
    #[serde(flatten)]
    pub common: ProviderCommon,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct OpenAIConfig {
    #[serde(flatten)]
    pub common: ProviderCommon,
    #[serde(default)]
    pub base_url: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct OllamaConfig {
    #[serde(flatten)]
    pub common: ProviderCommon,
    #[serde(default)]
    pub host: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct LMStudioConfig {
    #[serde(flatten)]
    pub common: ProviderCommon,
    #[serde(default)]
    pub base_url: Option<String>,
}

pub fn get_git_root() -> Option<PathBuf> {
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

pub fn find_config_file() -> Option<PathBuf> {
    let current_dir = std::env::current_dir().ok()?;
    let local_config = current_dir.join(".config.tai");
    if local_config.exists() {
        return Some(local_config);
    }
    if let Some(git_root) = get_git_root() {
        let git_config = git_root.join(".config.tai");
        if git_config.exists() {
            return Some(git_config);
        }
    }
    None
}

pub fn get_global_config_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Failed to get home directory")?;
    let config_dir = home.join(".config").join("tai");
    fs::create_dir_all(&config_dir)?;
    Ok(config_dir)
}

fn global_config_path() -> Result<PathBuf> {
    Ok(get_global_config_dir()?.join("config.tai"))
}

pub fn load_config() -> Result<Config> {
    let mut config = Config::default();
    let global_config_path = global_config_path()?;
    if global_config_path.exists() {
        let global_content = fs::read_to_string(&global_config_path)?;
        config = toml::from_str(&global_content)?;
    }
    if let Some(local_config_path) = find_config_file() {
        let local_content = fs::read_to_string(&local_config_path)?;
        let local_config: Config = toml::from_str(&local_content)?;
        merge_config(&mut config, &local_config);
    }
    migrate_legacy_keys(&mut config)?;
    Ok(config)
}

fn merge_config(base: &mut Config, over: &Config) {
    if over.core.active_provider.is_some() {
        base.core.active_provider = over.core.active_provider.clone();
    }
    merge_provider_common(
        &mut base.providers.anthropic.common,
        &over.providers.anthropic.common,
    );
    merge_provider_common(
        &mut base.providers.openai.common,
        &over.providers.openai.common,
    );
    if over.providers.openai.base_url.is_some() {
        base.providers.openai.base_url = over.providers.openai.base_url.clone();
    }
    merge_provider_common(
        &mut base.providers.ollama.common,
        &over.providers.ollama.common,
    );
    if over.providers.ollama.host.is_some() {
        base.providers.ollama.host = over.providers.ollama.host.clone();
    }
    merge_provider_common(
        &mut base.providers.lmstudio.common,
        &over.providers.lmstudio.common,
    );
    if over.providers.lmstudio.base_url.is_some() {
        base.providers.lmstudio.base_url = over.providers.lmstudio.base_url.clone();
    }
    if !over.global_contexts.is_empty() {
        base.global_contexts = over.global_contexts.clone();
    }
}

fn merge_provider_common(base: &mut ProviderCommon, over: &ProviderCommon) {
    if over.default_model.is_some() {
        base.default_model = over.default_model.clone();
    }
    if over.temperature.is_some() {
        base.temperature = over.temperature;
    }
    if over.max_tokens.is_some() {
        base.max_tokens = over.max_tokens;
    }
}

fn migrate_legacy_keys(cfg: &mut Config) -> Result<()> {
    let mut changed = false;
    if cfg.model.is_some() || cfg.temperature.is_some() || cfg.max_tokens.is_some() {
        let target = detect_preferred_provider_env().unwrap_or_else(|| "anthropic".to_string());
        let (model, temp, tokens) = (
            cfg.model.take(),
            cfg.temperature.take(),
            cfg.max_tokens.take(),
        );
        match target.as_str() {
            "anthropic" => {
                if model.is_some() {
                    cfg.providers.anthropic.common.default_model = model;
                }
                if temp.is_some() {
                    cfg.providers.anthropic.common.temperature = temp;
                }
                if tokens.is_some() {
                    cfg.providers.anthropic.common.max_tokens = tokens;
                }
            }
            "openai" => {
                if model.is_some() {
                    cfg.providers.openai.common.default_model = model;
                }
                if temp.is_some() {
                    cfg.providers.openai.common.temperature = temp;
                }
                if tokens.is_some() {
                    cfg.providers.openai.common.max_tokens = tokens;
                }
            }
            "ollama" => {
                if model.is_some() {
                    cfg.providers.ollama.common.default_model = model;
                }
                if temp.is_some() {
                    cfg.providers.ollama.common.temperature = temp;
                }
                if tokens.is_some() {
                    cfg.providers.ollama.common.max_tokens = tokens;
                }
            }
            "lmstudio" => {
                if model.is_some() {
                    cfg.providers.lmstudio.common.default_model = model;
                }
                if temp.is_some() {
                    cfg.providers.lmstudio.common.temperature = temp;
                }
                if tokens.is_some() {
                    cfg.providers.lmstudio.common.max_tokens = tokens;
                }
            }
            _ => {}
        }
        changed = true;
    }
    if cfg.anthropic_api_key.take().is_some() {
        changed = true;
    }
    if changed {
        let _ = save_config(cfg, true);
    }
    Ok(())
}

pub fn save_config(config: &Config, global: bool) -> Result<()> {
    let config_path = if global {
        global_config_path()?
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

pub fn find_context_files(context_name: Option<&str>) -> Result<Vec<(String, String)>> {
    let mut contexts = Vec::new();
    let current_dir = std::env::current_dir()?;
    if let Some(name) = context_name {
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

pub struct ProviderStatus {
    pub name: String,
    pub available: bool,
    pub reason: String,
    pub active: bool,
    pub model: Option<String>,
}

pub fn list_providers(cfg: &Config) -> Vec<ProviderStatus> {
    let active = cfg.core.active_provider.clone();
    let mut out = Vec::new();
    let anth = is_anthropic_available();
    out.push(ProviderStatus {
        name: "anthropic".into(),
        available: anth,
        reason: if anth {
            "key present".into()
        } else {
            "no ANTHROPIC_API_KEY".into()
        },
        active: active.as_deref() == Some("anthropic"),
        model: cfg.providers.anthropic.common.default_model.clone(),
    });
    let (ok, why) = is_openai_available();
    out.push(ProviderStatus {
        name: "openai".into(),
        available: ok,
        reason: why,
        active: active.as_deref() == Some("openai"),
        model: cfg.providers.openai.common.default_model.clone(),
    });
    let (ok, why) = is_ollama_available(cfg);
    out.push(ProviderStatus {
        name: "ollama".into(),
        available: ok,
        reason: why,
        active: active.as_deref() == Some("ollama"),
        model: cfg.providers.ollama.common.default_model.clone(),
    });
    let (ok, why) = is_lmstudio_available(cfg);
    out.push(ProviderStatus {
        name: "lmstudio".into(),
        available: ok,
        reason: why,
        active: active.as_deref() == Some("lmstudio"),
        model: cfg.providers.lmstudio.common.default_model.clone(),
    });
    out
}

pub fn set_active_provider_global(name: &str) -> Result<()> {
    let mut cfg = load_config()?;
    match name {
        "anthropic" | "openai" | "ollama" | "lmstudio" => {
            cfg.core.active_provider = Some(name.to_string());
            save_config(&cfg, true)
        }
        _ => Err(anyhow!("Unsupported provider: {}", name)),
    }
}

pub fn clear_active_provider_global() -> Result<()> {
    let mut cfg = load_config()?;
    cfg.core.active_provider = None;
    save_config(&cfg, true)
}

fn is_anthropic_available() -> bool {
    std::env::var("ANTHROPIC_API_KEY")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

fn is_openai_available() -> (bool, String) {
    let key_ok = std::env::var("OPENAI_API_KEY")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    if key_ok {
        return (true, "key present".into());
    }
    if std::env::var("OPENAI_BASE_URL")
        .ok()
        .filter(|v| !v.is_empty())
        .is_some()
    {
        return (true, "base_url set (OPENAI-compatible)".into());
    }
    (false, "no OPENAI_API_KEY or base_url".into())
}

fn is_ollama_available(cfg: &Config) -> (bool, String) {
    let host = cfg
        .providers
        .ollama
        .host
        .clone()
        .or_else(|| std::env::var("OLLAMA_BASE_URL").ok())
        .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());
    let url = format!("{}/api/tags", host.trim_end_matches('/'));
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(500))
        .build();
    if let Ok(c) = client {
        if let Ok(resp) = c.get(&url).send() {
            if resp.status().is_success() {
                return (true, format!("server up at {}", host));
            }
        }
    }
    (false, format!("server down at {}", host))
}

fn is_lmstudio_available(cfg: &Config) -> (bool, String) {
    let base = cfg
        .providers
        .lmstudio
        .base_url
        .clone()
        .or_else(|| std::env::var("LM_STUDIO_BASE_URL").ok())
        .unwrap_or_else(|| "http://127.0.0.1:1234/v1".to_string());
    let url = format!("{}/models", base.trim_end_matches('/'));
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(500))
        .build();
    if let Ok(c) = client {
        if let Ok(resp) = c.get(&url).send() {
            if resp.status().is_success() {
                return (true, format!("server up at {}", base));
            }
        }
    }
    (false, format!("server down at {}", base))
}

pub struct EffectiveProvider {
    pub name: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub base_url_or_host: Option<String>,
}

pub fn detect_preferred_provider_env() -> Option<String> {
    if is_anthropic_available() {
        return Some("anthropic".into());
    }
    let (ok, _) = is_openai_available();
    if ok {
        return Some("openai".into());
    }
    None
}

pub fn select_effective_provider(cfg: &Config) -> EffectiveProvider {
    if let Some(name) = cfg.core.active_provider.clone() {
        return build_effective(&name, cfg).unwrap_or_else(|| auto_select(cfg));
    }
    auto_select(cfg)
}

fn auto_select(cfg: &Config) -> EffectiveProvider {
    if is_anthropic_available() {
        if let Some(eff) = build_effective("anthropic", cfg) {
            return eff;
        }
    }
    let (ok, _) = is_openai_available();
    if ok {
        if let Some(eff) = build_effective("openai", cfg) {
            return eff;
        }
    }
    let (ok, _) = is_ollama_available(cfg);
    if ok {
        if let Some(eff) = build_effective("ollama", cfg) {
            return eff;
        }
    }
    let (ok, _) = is_lmstudio_available(cfg);
    if ok {
        if let Some(eff) = build_effective("lmstudio", cfg) {
            return eff;
        }
    }
    build_effective("ollama", cfg).unwrap()
}

fn build_effective(name: &str, cfg: &Config) -> Option<EffectiveProvider> {
    match name {
        "anthropic" => Some(EffectiveProvider {
            name: "anthropic".into(),
            model: cfg
                .providers
                .anthropic
                .common
                .default_model
                .clone()
                .unwrap_or_else(|| "claude-3-5-sonnet-latest".into()),
            temperature: cfg.providers.anthropic.common.temperature.unwrap_or(0.0),
            max_tokens: cfg.providers.anthropic.common.max_tokens.unwrap_or(1500),
            base_url_or_host: None,
        }),
        "openai" => Some(EffectiveProvider {
            name: "openai".into(),
            model: cfg
                .providers
                .openai
                .common
                .default_model
                .clone()
                .unwrap_or_else(|| "gpt-4o-mini".into()),
            temperature: cfg.providers.openai.common.temperature.unwrap_or(0.0),
            max_tokens: cfg.providers.openai.common.max_tokens.unwrap_or(1500),
            base_url_or_host: cfg
                .providers
                .openai
                .base_url
                .clone()
                .or_else(|| std::env::var("OPENAI_BASE_URL").ok()),
        }),
        "ollama" => Some(EffectiveProvider {
            name: "ollama".into(),
            model: cfg
                .providers
                .ollama
                .common
                .default_model
                .clone()
                .unwrap_or_else(|| "deepseek-r1:8b".into()),
            temperature: cfg.providers.ollama.common.temperature.unwrap_or(0.0),
            max_tokens: cfg.providers.ollama.common.max_tokens.unwrap_or(1500),
            base_url_or_host: cfg
                .providers
                .ollama
                .host
                .clone()
                .or_else(|| std::env::var("OLLAMA_BASE_URL").ok())
                .or_else(|| Some("http://127.0.0.1:11434".into())),
        }),
        "lmstudio" => Some(EffectiveProvider {
            name: "lmstudio".into(),
            model: cfg
                .providers
                .lmstudio
                .common
                .default_model
                .clone()
                .unwrap_or_else(|| "gpt-4o-mini".into()),
            temperature: cfg.providers.lmstudio.common.temperature.unwrap_or(0.0),
            max_tokens: cfg.providers.lmstudio.common.max_tokens.unwrap_or(1500),
            base_url_or_host: cfg
                .providers
                .lmstudio
                .base_url
                .clone()
                .or_else(|| std::env::var("LM_STUDIO_BASE_URL").ok())
                .or_else(|| Some("http://127.0.0.1:1234/v1".into())),
        }),
        _ => None,
    }
}

pub fn format_provider_statuses(statuses: &[ProviderStatus]) -> String {
    let mut out = String::new();
    for s in statuses {
        let mark = if s.active { "[active]" } else { "" };
        let icon = if s.available { "✓" } else { "○" };
        let model = s.model.clone().unwrap_or_else(|| "-".into());
        out.push_str(&format!(
            "{} {} ({}; model: {}) {}\n",
            icon, s.name, s.reason, model, mark
        ));
    }
    out
}

pub fn update_provider_settings(
    name: &str,
    model: Option<String>,
    base_url: Option<String>,
    host: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> Result<()> {
    let mut cfg = load_config()?;
    match name {
        "anthropic" => {
            if let Some(m) = model {
                cfg.providers.anthropic.common.default_model = Some(m);
            }
            if let Some(t) = temperature {
                cfg.providers.anthropic.common.temperature = Some(t);
            }
            if let Some(mt) = max_tokens {
                cfg.providers.anthropic.common.max_tokens = Some(mt);
            }
        }
        "openai" => {
            if let Some(m) = model {
                cfg.providers.openai.common.default_model = Some(m);
            }
            if let Some(b) = base_url {
                cfg.providers.openai.base_url = Some(b);
            }
            if let Some(t) = temperature {
                cfg.providers.openai.common.temperature = Some(t);
            }
            if let Some(mt) = max_tokens {
                cfg.providers.openai.common.max_tokens = Some(mt);
            }
        }
        "ollama" => {
            if let Some(m) = model {
                cfg.providers.ollama.common.default_model = Some(m);
            }
            if let Some(h) = host {
                cfg.providers.ollama.host = Some(h);
            }
            if let Some(t) = temperature {
                cfg.providers.ollama.common.temperature = Some(t);
            }
            if let Some(mt) = max_tokens {
                cfg.providers.ollama.common.max_tokens = Some(mt);
            }
        }
        "lmstudio" => {
            if let Some(m) = model {
                cfg.providers.lmstudio.common.default_model = Some(m);
            }
            if let Some(b) = base_url {
                cfg.providers.lmstudio.base_url = Some(b);
            }
            if let Some(t) = temperature {
                cfg.providers.lmstudio.common.temperature = Some(t);
            }
            if let Some(mt) = max_tokens {
                cfg.providers.lmstudio.common.max_tokens = Some(mt);
            }
        }
        _ => return Err(anyhow!("Unsupported provider: {}", name)),
    }
    save_config(&cfg, true)
}

pub fn handle_config_command(
    key: Option<String>,
    value: Option<String>,
    global: bool,
) -> Result<()> {
    let mut config = if global {
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
            println!("Configuration:");
            println!("  global_contexts:");
            if config.global_contexts.is_empty() {
                println!("    <none>");
            } else {
                for context in &config.global_contexts {
                    println!("    - {}", context);
                }
            }
            println!("  note: provider settings (model, temperature, max_tokens) are now per-provider. Use 'tai config provider ...' or 'tai config <provider> ...'.");
        }
        (Some(key), None) => match key.as_str() {
            "global_contexts" => {
                if config.global_contexts.is_empty() {
                    println!("<none>");
                } else {
                    for context in &config.global_contexts {
                        println!("{}", context);
                    }
                }
            }
            _ => {
                return Err(anyhow!(
                    "Unknown or moved config key: {} (use provider-specific commands)",
                    key
                ))
            }
        },
        (Some(key), Some(value)) => {
            match key.as_str() {
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
                _ => {
                    return Err(anyhow!(
                        "Unknown or moved config key: {} (use provider-specific commands)",
                        key
                    ))
                }
            }
            save_config(&config, global)?;
            println!("Configuration updated");
        }
        _ => unreachable!(),
    }
    Ok(())
}


pub fn handle_config_provider_list() -> Result<()> {
    let cfg = load_config()?;
    let statuses = list_providers(&cfg);
    print!("{}", format_provider_statuses(&statuses));
    Ok(())
}

pub fn handle_config_provider_set(name: &str) -> Result<()> {
    set_active_provider_global(name)?;
    println!("Active provider set to {}", name);
    Ok(())
}

pub fn handle_config_provider_auto() -> Result<()> {
    clear_active_provider_global()?;
    let cfg = load_config()?;
    let eff = select_effective_provider(&cfg);
    println!("Auto provider would be {} (model: {})", eff.name, eff.model);
    Ok(())
}

pub fn handle_config_provider_show(name: &str) -> Result<()> {
    let cfg = load_config()?;
    let eff = build_effective(name, &cfg).ok_or_else(|| anyhow!("Unknown provider"))?;
    println!(
        "provider: {}\nmodel: {}\ntemperature: {}\nmax_tokens: {}\nendpoint: {}",
        eff.name,
        eff.model,
        eff.temperature,
        eff.max_tokens,
        eff.base_url_or_host.unwrap_or_else(|| "-".into())
    );
    Ok(())
}

pub fn handle_config_provider_update(
    name: &str,
    model: Option<String>,
    base_url: Option<String>,
    host: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> Result<()> {
    update_provider_settings(name, model, base_url, host, temperature, max_tokens)?;
    println!("Provider {} updated", name);
    Ok(())
}
