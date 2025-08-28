use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use globset::{Glob, GlobSet, GlobSetBuilder};
use llm::builder::ParamBuilder;
use llm::chat::ParameterProperty;
use walkdir::WalkDir;

use super::Tool;

pub(super) fn resolve_path(p: &str, allow_nonexistent: bool) -> Result<PathBuf> {
    let root = workspace_root()?;
    let candidate = Path::new(p);
    let abs = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root.join(candidate)
    };
    let canonical = if allow_nonexistent {
        if let Some(parent) = abs.parent() {
            let can_parent = parent
                .canonicalize()
                .with_context(|| format!("Failed to canonicalize parent of {}", abs.display()))?;
            can_parent.join(abs.file_name().unwrap_or_default())
        } else {
            abs.canonicalize()
                .with_context(|| format!("Failed to canonicalize {}", abs.display()))?
        }
    } else {
        abs.canonicalize()
            .with_context(|| format!("Failed to canonicalize {}", abs.display()))?
    };
    if !is_within(&root.canonicalize()?, &canonical) {
        return Err(anyhow!("Path escapes workspace root"));
    }
    Ok(canonical)
}

fn workspace_root() -> Result<PathBuf> {
    std::env::current_dir().context("Failed to determine current directory")
}

fn is_within(root: &Path, path: &Path) -> bool {
    let Ok(root_c) = root.canonicalize() else {
        return false;
    };
    let Ok(path_c) = path.canonicalize() else {
        return false;
    };
    path_c.starts_with(root_c)
}

pub struct ListDirTool;
impl Tool for ListDirTool {
    fn name(&self) -> &'static str {
        "list_dir"
    }
    fn description(&self) -> &'static str {
        "List files in a directory with optional recursion and glob filters."
    }
    fn params(&self) -> Vec<ParamBuilder> {
        vec![
            ParamBuilder::new("path")
                .type_of("string")
                .description("Directory path (default '.')"),
            ParamBuilder::new("recursive")
                .type_of("boolean")
                .description("Recurse into subdirectories (default false)"),
            ParamBuilder::new("include_globs")
                .type_of("array")
                .items(ParameterProperty {
                    property_type: "string".into(),
                    description: "glob".into(),
                    items: None,
                    enum_list: None,
                })
                .description("Include glob patterns"),
            ParamBuilder::new("exclude_globs")
                .type_of("array")
                .items(ParameterProperty {
                    property_type: "string".into(),
                    description: "glob".into(),
                    items: None,
                    enum_list: None,
                })
                .description("Exclude glob patterns"),
            ParamBuilder::new("limit")
                .type_of("integer")
                .description("Limit number of entries (default 1000)"),
            ParamBuilder::new("include_hidden")
                .type_of("boolean")
                .description("Include dotfiles (default false)"),
        ]
    }
    fn execute_blocking(&self, args: Value) -> Result<Value> {
        let path_s = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let recursive = args
            .get("recursive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let include_hidden = args
            .get("include_hidden")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(1000) as usize;
        let includes = args.get("include_globs").and_then(|v| v.as_array());
        let excludes = args.get("exclude_globs").and_then(|v| v.as_array());
        let path = resolve_path(path_s, false)?;

        let mut gb = GlobSetBuilder::new();
        let mut inc_any = false;
        if let Some(arr) = includes {
            for g in arr {
                if let Some(s) = g.as_str() {
                    gb.add(Glob::new(s).with_context(|| format!("bad include glob {}", s))?);
                    inc_any = true;
                }
            }
        }
        let include_set: Option<GlobSet> = if inc_any { Some(gb.build()?) } else { None };
        let mut gb2 = GlobSetBuilder::new();
        let mut exc_any = false;
        if let Some(arr) = excludes {
            for g in arr {
                if let Some(s) = g.as_str() {
                    gb2.add(Glob::new(s).with_context(|| format!("bad exclude glob {}", s))?);
                    exc_any = true;
                }
            }
        }
        let exclude_set: Option<GlobSet> = if exc_any { Some(gb2.build()?) } else { None };

        let mut items = Vec::new();
        if recursive {
            for entry in WalkDir::new(&path).into_iter().filter_map(|e| e.ok()) {
                let p = entry.path().to_path_buf();
                if p == path {
                    continue;
                }
                if !include_hidden {
                    if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                        if name.starts_with('.') {
                            continue;
                        }
                    }
                }
                if let Some(ref ex) = exclude_set {
                    if ex.is_match(&p) {
                        continue;
                    }
                }
                if let Some(ref inc) = include_set {
                    if !inc.is_match(&p) {
                        continue;
                    }
                }
                items.push(path_info(&p)?);
                if items.len() >= limit {
                    break;
                }
            }
        } else {
            for entry in
                fs::read_dir(&path).with_context(|| format!("Failed to read {}", path.display()))?
            {
                let entry = entry?;
                let p = entry.path();
                if !include_hidden {
                    if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                        if name.starts_with('.') {
                            continue;
                        }
                    }
                }
                if let Some(ref ex) = exclude_set {
                    if ex.is_match(&p) {
                        continue;
                    }
                }
                if let Some(ref inc) = include_set {
                    if !inc.is_match(&p) {
                        continue;
                    }
                }
                items.push(path_info(&p)?);
                if items.len() >= limit {
                    break;
                }
            }
        }
        Ok(json!({ "path": path.display().to_string(), "count": items.len(), "items": items }))
    }
}

fn fmt_time(t: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn path_info(p: &Path) -> Result<Value> {
    let md = fs::symlink_metadata(p).with_context(|| format!("stat failed for {}", p.display()))?;
    let file_type = if md.is_dir() {
        "dir"
    } else if md.is_file() {
        "file"
    } else if md.file_type().is_symlink() {
        "symlink"
    } else {
        "other"
    };
    let size = md.len();
    let modified = md.modified().ok();
    let created = md.created().ok();
    let mode = if cfg!(unix) {
        use std::os::unix::fs::PermissionsExt;
        format!("{:o}", md.permissions().mode())
    } else {
        String::from("")
    };
    Ok(json!({
        "path": p.display().to_string(),
        "type": file_type,
        "size": size,
        "modified": modified.map(fmt_time),
        "created": created.map(fmt_time),
        "mode": mode,
    }))
}

pub struct StatTool;
impl Tool for StatTool {
    fn name(&self) -> &'static str {
        "stat"
    }
    fn description(&self) -> &'static str {
        "Get file metadata (type, size, mtime, ctime, mode)."
    }
    fn required_params(&self) -> &'static [&'static str] {
        &["path"]
    }
    fn params(&self) -> Vec<ParamBuilder> {
        vec![ParamBuilder::new("path")
            .type_of("string")
            .description("Path to stat")]
    }
    fn execute_blocking(&self, args: Value) -> Result<Value> {
        let path_s = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'path'"))?;
        let path = resolve_path(path_s, false)?;
        path_info(&path)
    }
}

pub struct GlobTool;
impl Tool for GlobTool {
    fn name(&self) -> &'static str {
        "glob"
    }
    fn description(&self) -> &'static str {
        "Find files matching a glob pattern under a root (recursive)."
    }
    fn required_params(&self) -> &'static [&'static str] {
        &["pattern"]
    }
    fn params(&self) -> Vec<ParamBuilder> {
        vec![
            ParamBuilder::new("pattern")
                .type_of("string")
                .description("Glob pattern (e.g., src/**/*.rs)"),
            ParamBuilder::new("root")
                .type_of("string")
                .description("Root directory to search (default '.')"),
            ParamBuilder::new("limit")
                .type_of("integer")
                .description("Max results (default 200)"),
        ]
    }
    fn execute_blocking(&self, args: Value) -> Result<Value> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'pattern'"))?;
        let root_s = args.get("root").and_then(|v| v.as_str()).unwrap_or(".");
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
        let root = resolve_path(root_s, false)?;
        let glob = Glob::new(pattern)
            .with_context(|| format!("bad glob {}", pattern))?
            .compile_matcher();
        let mut results = Vec::new();
        for entry in WalkDir::new(&root).into_iter().filter_map(|e| e.ok()) {
            let p = entry.path();
            if p.is_file() && glob.is_match(p) {
                results.push(p.display().to_string());
                if results.len() >= limit {
                    break;
                }
            }
        }
        Ok(
            json!({ "root": root.display().to_string(), "pattern": pattern, "count": results.len(), "paths": results }),
        )
    }
}
