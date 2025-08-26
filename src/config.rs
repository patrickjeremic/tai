use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct Config {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub anthropic_api_key: Option<String>,
    #[serde(default)]
    pub global_contexts: Vec<String>,
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

pub fn load_config() -> Result<Config> {
    let mut config = Config::default();

    let global_config_dir = get_global_config_dir()?;
    let global_config_path = global_config_dir.join("config.tai");
    if global_config_path.exists() {
        let global_content = fs::read_to_string(&global_config_path)?;
        config = toml::from_str(&global_content)?;
    }

    if let Some(local_config_path) = find_config_file() {
        let local_content = fs::read_to_string(&local_config_path)?;
        let local_config: Config = toml::from_str(&local_content)?;

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

pub fn save_config(config: &Config, global: bool) -> Result<()> {
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
        (Some(key), None) => match key.as_str() {
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
        },
        (Some(key), Some(value)) => {
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
        _ => unreachable!(),
    }

    Ok(())
}
