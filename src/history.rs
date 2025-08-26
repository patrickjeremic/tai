use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use dirs::home_dir;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub user_input: String,
    pub llm_response: String,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct History {
    pub entries: Vec<HistoryEntry>,
}

impl History {
    pub fn load() -> Result<Self> {
        let history_path = Self::history_path()?;

        if !history_path.exists() {
            return Ok(Self::default());
        }

        let mut file = File::open(&history_path)
            .context(format!("Failed to open history file at {:?}", history_path))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .context("Failed to read history file")?;

        if contents.is_empty() {
            return Ok(Self::default());
        }

        serde_json::from_str(&contents).context("Failed to parse history file")
    }

    pub fn save(&self) -> Result<()> {
        let history_path = Self::history_path()?;

        if let Some(parent) = history_path.parent() {
            fs::create_dir_all(parent)
                .context(format!("Failed to create directory at {:?}", parent))?;
        }

        let json = serde_json::to_string_pretty(self).context("Failed to serialize history")?;

        let mut file = File::create(&history_path).context(format!(
            "Failed to create history file at {:?}",
            history_path
        ))?;

        file.write_all(json.as_bytes())
            .context("Failed to write history file")?;

        Ok(())
    }

    pub fn add_entry(&mut self, user_input: String, llm_response: String) -> Result<()> {
        let entry = HistoryEntry {
            timestamp: Utc::now(),
            user_input,
            llm_response,
        };

        self.entries.push(entry);

        if self.entries.len() > 10 {
            self.entries = self.entries.split_off(self.entries.len() - 10);
        }

        self.save()
    }

    pub fn get_relevant_entries(&self) -> Vec<(HistoryEntry, Duration)> {
        let now = Utc::now();
        let one_hour = Duration::hours(1);

        self.entries
            .iter()
            .filter_map(|entry| {
                let age = now - entry.timestamp;
                if age < one_hour {
                    Some((entry.clone(), age))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn clear() -> Result<()> {
        let history_file = dirs::home_dir()
            .context("Failed to get home directory")?
            .join(".tai_history.json");

        if history_file.exists() {
            fs::remove_file(&history_file)?;
        }

        Ok(())
    }

    fn history_path() -> Result<PathBuf> {
        let mut path = home_dir().context("Failed to get home directory")?;
        path.push(".tai.history");
        Ok(path)
    }
}
