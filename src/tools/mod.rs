use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime};

use globset::{Glob, GlobSet, GlobSetBuilder};
use ignore::WalkBuilder;
use llm::builder::{FunctionBuilder, LLMBuilder, ParamBuilder};
use llm::chat::ParameterProperty;
use llm::ToolCall;
use regex::RegexBuilder;
use reqwest::blocking::{Client, Response};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::str::FromStr;
use walkdir::WalkDir;

pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn required_params(&self) -> &'static [&'static str] {
        &[]
    }
    fn params(&self) -> Vec<ParamBuilder>;
    fn register_on(&self, builder: LLMBuilder) -> LLMBuilder {
        let mut fb = FunctionBuilder::new(self.name()).description(self.description());
        for p in self.params() {
            fb = fb.param(p);
        }
        let required = self
            .required_params()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let fb = if required.is_empty() {
            fb
        } else {
            fb.required(required)
        };
        builder.function(fb)
    }
    fn execute_blocking(&self, args: Value) -> Result<Value>;
}

pub struct ToolsRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl Default for ToolsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolsRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }
    pub fn with_default() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(ReadFileTool));
        reg.register(Box::new(WriteFileTool));
        reg.register(Box::new(PatchFileTool));
        reg.register(Box::new(ListDirTool));
        reg.register(Box::new(StatTool));
        reg.register(Box::new(GlobTool));
        reg.register(Box::new(GrepTool));
        reg.register(Box::new(ShellCommandTool));
        reg.register(Box::new(FetchUrlTool));
        reg
    }
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }
    pub fn apply_to_builder(&self, mut builder: LLMBuilder) -> LLMBuilder {
        for t in &self.tools {
            builder = t.register_on(builder);
        }
        builder
    }
    pub fn find(&self, name: &str) -> Option<&dyn Tool> {
        for t in &self.tools {
            if t.name() == name {
                return Some(t.as_ref());
            }
        }
        None
    }
    pub fn handle_tool_call(&self, call: &ToolCall) -> Result<Value> {
        let name = &call.function.name;
        let args: Value = serde_json::from_str(&call.function.arguments)
            .with_context(|| format!("Failed parsing tool args for {}", name))?;
        let tool = self
            .find(name)
            .ok_or_else(|| anyhow!("Unknown tool: {}", name))?;
        tool.execute_blocking(args)
    }
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

fn resolve_path(p: &str, allow_nonexistent: bool) -> Result<PathBuf> {
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

pub struct ReadFileTool;
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }
    fn description(&self) -> &'static str {
        "Read a text file with optional line offset and limit. Returns content and metadata."
    }
    fn required_params(&self) -> &'static [&'static str] {
        &["path"]
    }
    fn params(&self) -> Vec<ParamBuilder> {
        vec![
            ParamBuilder::new("path")
                .type_of("string")
                .description("File path to read (relative to workspace)"),
            ParamBuilder::new("offset")
                .type_of("integer")
                .description("Optional starting line (0-based)"),
            ParamBuilder::new("limit")
                .type_of("integer")
                .description("Optional number of lines to return"),
        ]
    }
    fn execute_blocking(&self, args: Value) -> Result<Value> {
        let path_s = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'path'"))?;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let path = resolve_path(path_s, false)?;
        let mut s = String::new();
        fs::File::open(&path)
            .and_then(|mut f| f.read_to_string(&mut s))
            .with_context(|| format!("Failed reading {}", path.display()))?;
        let lines: Vec<&str> = s.lines().collect();
        let total_lines = lines.len();
        let start = offset.min(total_lines);
        let end = match limit {
            Some(l) => (start + l).min(total_lines),
            None => total_lines,
        };
        let slice = lines[start..end].join("\n");
        Ok(json!({
            "path": path.display().to_string(),
            "start": start,
            "end": end,
            "total_lines": total_lines,
            "content": slice,
        }))
    }
}

pub struct WriteFileTool;
impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }
    fn description(&self) -> &'static str {
        "Write content to a file atomically. Creates parent directories if needed."
    }
    fn required_params(&self) -> &'static [&'static str] {
        &["path", "content"]
    }
    fn params(&self) -> Vec<ParamBuilder> {
        vec![
            ParamBuilder::new("path")
                .type_of("string")
                .description("File path to write (relative to workspace)"),
            ParamBuilder::new("content")
                .type_of("string")
                .description("Full file content to write"),
            ParamBuilder::new("atomic")
                .type_of("boolean")
                .description("Write atomically (default true)"),
            ParamBuilder::new("create_parents")
                .type_of("boolean")
                .description("Create parent directories if needed (default true)"),
        ]
    }
    fn execute_blocking(&self, args: Value) -> Result<Value> {
        let path_s = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'path'"))?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'content'"))?;
        let atomic = args.get("atomic").and_then(|v| v.as_bool()).unwrap_or(true);
        let create_parents = args
            .get("create_parents")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let path = resolve_path(path_s, true)?;
        if let Some(parent) = path.parent() {
            if create_parents {
                fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create {}", parent.display()))?;
            }
        }
        if atomic {
            let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("tmp");
            let mut suffix = 0u32;
            let tmp = loop {
                let candidate = parent_join(&path, &format!(".{}.tmp.{}", file_name, suffix));
                if !candidate.exists() {
                    break candidate;
                }
                suffix += 1;
            };
            fs::write(&tmp, content)
                .with_context(|| format!("Failed to write temp file {}", tmp.display()))?;
            fs::rename(&tmp, &path)
                .with_context(|| format!("Failed to replace {}", path.display()))?;
        } else {
            fs::write(&path, content)
                .with_context(|| format!("Failed to write {}", path.display()))?;
        }
        Ok(json!({ "path": path.display().to_string(), "bytes": content.len() }))
    }
}

fn parent_join(path: &Path, file: &str) -> PathBuf {
    path.parent().unwrap_or_else(|| Path::new("")).join(file)
}

pub struct PatchFileTool;
impl Tool for PatchFileTool {
    fn name(&self) -> &'static str {
        "patch_file"
    }
    fn description(&self) -> &'static str {
        "Apply multiple string replacements to a file (transactional). Each replacement may be replace_all or single occurrence."
    }
    fn required_params(&self) -> &'static [&'static str] {
        &["path", "replacements"]
    }
    fn params(&self) -> Vec<ParamBuilder> {
        vec![
            ParamBuilder::new("path")
                .type_of("string")
                .description("File path to patch"),
            ParamBuilder::new("replacements")
                .type_of("array")
                .items(ParameterProperty {
                    property_type: "object".into(),
                    description: "Replacement operation".into(),
                    items: None,
                    enum_list: None,
                })
                .description("Array of {old_string,new_string,replace_all?}"),
            ParamBuilder::new("atomic")
                .type_of("boolean")
                .description("Apply atomically (default true)"),
        ]
    }
    fn execute_blocking(&self, args: Value) -> Result<Value> {
        let path_s = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'path'"))?;
        let path = resolve_path(path_s, false)?;
        let replacements = args
            .get("replacements")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("Missing 'replacements'"))?;
        let atomic = args.get("atomic").and_then(|v| v.as_bool()).unwrap_or(true);
        let mut content = String::new();
        fs::File::open(&path)
            .and_then(|mut f| f.read_to_string(&mut content))
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let mut counts: Vec<usize> = Vec::new();
        let mut updated = content.clone();
        for rep in replacements {
            let old_s = rep
                .get("old_string")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("replacement missing 'old_string'"))?;
            let new_s = rep
                .get("new_string")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("replacement missing 'new_string'"))?;
            let replace_all = rep
                .get("replace_all")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if old_s.is_empty() {
                return Err(anyhow!("old_string cannot be empty"));
            }
            if replace_all {
                let c = updated.matches(old_s).count();
                updated = updated.replace(old_s, new_s);
                counts.push(c);
            } else if let Some(idx) = updated.find(old_s) {
                updated.replace_range(idx..idx + old_s.len(), new_s);
                counts.push(1);
            } else {
                counts.push(0);
            }
        }
        if updated == content {
            return Ok(json!({
                "path": path.display().to_string(),
                "changed": false,
                "replacements": counts,
                "total_replacements": counts.iter().sum::<usize>(),
            }));
        }
        if atomic {
            let tmp = parent_join(
                &path,
                &format!(
                    ".{}.patch.tmp",
                    path.file_name().and_then(|s| s.to_str()).unwrap_or("file")
                ),
            );
            fs::write(&tmp, updated.as_bytes())
                .with_context(|| format!("Failed to write temp {}", tmp.display()))?;
            fs::rename(&tmp, &path)
                .with_context(|| format!("Failed to replace {}", path.display()))?;
        } else {
            fs::write(&path, updated.as_bytes())
                .with_context(|| format!("Failed to write {}", path.display()))?;
        }
        Ok(json!({
            "path": path.display().to_string(),
            "changed": true,
            "replacements": counts,
            "total_replacements": counts.iter().sum::<usize>(),
        }))
    }
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

pub struct GrepTool;
impl Tool for GrepTool {
    fn name(&self) -> &'static str {
        "grep"
    }
    fn description(&self) -> &'static str {
        "Search files for a pattern. Respects .gitignore. Returns file, line, and match snippet."
    }
    fn required_params(&self) -> &'static [&'static str] {
        &["pattern"]
    }
    fn params(&self) -> Vec<ParamBuilder> {
        vec![
            ParamBuilder::new("pattern")
                .type_of("string")
                .description("Regex or literal text to search for"),
            ParamBuilder::new("root")
                .type_of("string")
                .description("Root directory to search (default '.')"),
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
            ParamBuilder::new("literal")
                .type_of("boolean")
                .description("Treat pattern as literal (default false)"),
            ParamBuilder::new("case_sensitive")
                .type_of("boolean")
                .description("Case sensitive (default true)"),
            ParamBuilder::new("max_results")
                .type_of("integer")
                .description("Maximum results to return (default 100)"),
        ]
    }
    fn execute_blocking(&self, args: Value) -> Result<Value> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'pattern'"))?;
        let root_s = args.get("root").and_then(|v| v.as_str()).unwrap_or(".");
        let literal = args
            .get("literal")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let case_sensitive = args
            .get("case_sensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;
        let includes = args.get("include_globs").and_then(|v| v.as_array());
        let excludes = args.get("exclude_globs").and_then(|v| v.as_array());
        let root = resolve_path(root_s, false)?;

        let mut gb_inc = GlobSetBuilder::new();
        let mut inc_any = false;
        if let Some(arr) = includes {
            for g in arr {
                if let Some(s) = g.as_str() {
                    gb_inc.add(Glob::new(s).with_context(|| format!("bad include glob {}", s))?);
                    inc_any = true;
                }
            }
        }
        let inc = if inc_any { Some(gb_inc.build()?) } else { None };
        let mut gb_exc = GlobSetBuilder::new();
        let mut exc_any = false;
        if let Some(arr) = excludes {
            for g in arr {
                if let Some(s) = g.as_str() {
                    gb_exc.add(Glob::new(s).with_context(|| format!("bad exclude glob {}", s))?);
                    exc_any = true;
                }
            }
        }
        let exc = if exc_any { Some(gb_exc.build()?) } else { None };

        let pattern_str = if literal {
            regex::escape(pattern)
        } else {
            pattern.to_string()
        };
        let re = RegexBuilder::new(&pattern_str)
            .case_insensitive(!case_sensitive)
            .build()
            .with_context(|| "Invalid regex pattern")?;

        let mut results = Vec::new();
        let walker = WalkBuilder::new(&root)
            .hidden(false)
            .ignore(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .build();
        for dent in walker {
            if results.len() >= max_results {
                break;
            }
            let dent = match dent {
                Ok(d) => d,
                Err(_) => continue,
            };
            let p = dent.path();
            if !p.is_file() {
                continue;
            }
            if let Some(ref ex) = exc {
                if ex.is_match(p) {
                    continue;
                }
            }
            if let Some(ref ic) = inc {
                if !ic.is_match(p) {
                    continue;
                }
            }
            let mut buf = Vec::new();
            if fs::File::open(p)
                .and_then(|mut f| f.read_to_end(&mut buf))
                .is_err()
            {
                continue;
            }
            if is_binary(&buf) {
                continue;
            }
            let text = match String::from_utf8(buf) {
                Ok(s) => s,
                Err(_) => continue,
            };
            for (lineno, line) in text.lines().enumerate() {
                if re.is_match(line) {
                    results.push(json!({
                        "file": p.strip_prefix(&root).unwrap_or(p).display().to_string(),
                        "abs_path": p.display().to_string(),
                        "line": lineno + 1,
                        "match": line,
                    }));
                    if results.len() >= max_results {
                        break;
                    }
                }
            }
        }
        Ok(
            json!({ "root": root.display().to_string(), "pattern": pattern, "count": results.len(), "results": results }),
        )
    }
}

fn is_binary(buf: &[u8]) -> bool {
    const SAMPLE: usize = 8000;
    let n = buf.len().min(SAMPLE);
    for &b in &buf[..n] {
        if b == 0 {
            return true;
        }
    }
    false
}

pub struct ShellCommandTool;

impl Tool for ShellCommandTool {
    fn name(&self) -> &'static str {
        "run_shell"
    }
    fn description(&self) -> &'static str {
        "Execute a shell command on the user's machine. The user can see the command output! Use for tasks that require terminal operations. Always prefer safe, idempotent commands and avoid destructive operations."
    }
    fn required_params(&self) -> &'static [&'static str] {
        &["command"]
    }
    fn params(&self) -> Vec<ParamBuilder> {
        vec![
            ParamBuilder::new("command").type_of("string").description(
                "The exact shell command to execute (sh -c on Unix, cmd /C on Windows)",
            ),
            ParamBuilder::new("timeout_sec")
                .type_of("integer")
                .description("Optional timeout in seconds (defaults to 120)"),
        ]
    }
    fn execute_blocking(&self, args: Value) -> Result<Value> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'command'"))?
            .to_string();
        let timeout = args
            .get("timeout_sec")
            .and_then(|v| v.as_u64())
            .unwrap_or(120);

        println!("> {}", command);
        print!("Do you want to execute this command? [Y/n/c] ");
        std::io::Write::flush(&mut std::io::stdout()).context("Failed to flush stdout")?;
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .context("Failed to read user input")?;
        let choice = input.trim().to_lowercase();
        if choice == "c" {
            if let Ok(mut cb) = arboard::Clipboard::new() {
                if let Err(e) = cb.set_text(&command) {
                    eprintln!("Failed to copy to clipboard: {}", e);
                } else {
                    println!("Command copied to clipboard");
                }
            } else {
                eprintln!("Failed to access clipboard");
            }
            return Ok(json!({
                "command": command,
                "executed": false,
                "copied": true
            }));
        }
        if choice == "n" {
            println!("Command execution cancelled");
            return Ok(json!({
                "command": command,
                "executed": false
            }));
        }

        print!("\x1B[1A\x1B[2K\r");
        print!("\x1B[2K\r");

        let mut child = if cfg!(target_os = "windows") {
            std::process::Command::new("cmd")
                .args(["/C", &command])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
        } else {
            std::process::Command::new("sh")
                .args(["-c", &command])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
        }
        .context("Failed to execute command")?;

        let start = std::time::Instant::now();
        let status_output = loop {
            if let Some(status) = child.try_wait().context("wait failed")? {
                let output = child.wait_with_output().context("output failed")?;
                break Ok((status, output));
            }
            if start.elapsed().as_secs() >= timeout {
                let _ = child.kill();
                break Err(anyhow!("timeout after {}s", timeout));
            }
            std::thread::sleep(Duration::from_millis(50));
        };

        match status_output {
            Ok((status, output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let combined = if stderr.is_empty() {
                    stdout.clone()
                } else if stdout.is_empty() {
                    stderr.clone()
                } else {
                    format!("{}\n{}", stdout, stderr)
                };
                Ok(json!({
                    "command": command,
                    "executed": true,
                    "exit_status": status.code(),
                    "stdout": stdout,
                    "stderr": stderr,
                    "output": combined,
                }))
            }
            Err(e) => Ok(json!({
                "command": command,
                "executed": false,
                "error": e.to_string(),
            })),
        }
    }
}

pub struct FetchUrlTool;
impl Tool for FetchUrlTool {
    fn name(&self) -> &'static str {
        "fetch_url"
    }
    fn description(&self) -> &'static str {
        "Fetch content from an HTTP/HTTPS URL with optional method, headers, body, and timeout. Returns status, headers, and text (truncated)."
    }
    fn params(&self) -> Vec<ParamBuilder> {
        vec![
            ParamBuilder::new("url")
                .type_of("string")
                .description("The URL to fetch (http or https)"),
            ParamBuilder::new("method")
                .type_of("string")
                .description("HTTP method (default GET)"),
            ParamBuilder::new("headers")
                .type_of("object")
                .description("Optional headers as key-value object"),
            ParamBuilder::new("body")
                .type_of("string")
                .description("Optional request body for POST/PUT/PATCH"),
            ParamBuilder::new("timeout_sec")
                .type_of("integer")
                .description("Request timeout in seconds (default 10)"),
            ParamBuilder::new("max_bytes")
                .type_of("integer")
                .description("Maximum response bytes to capture (default 200000)"),
        ]
    }
    fn required_params(&self) -> &'static [&'static str] {
        &["url"]
    }
    fn execute_blocking(&self, args: Value) -> Result<Value> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'url'"))?;
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(anyhow!("Only http/https URLs are allowed"));
        }
        let method = args
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("GET")
            .to_uppercase();
        let timeout = args
            .get("timeout_sec")
            .and_then(|v| v.as_u64())
            .unwrap_or(10);
        let max_bytes = args
            .get("max_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(200_000) as usize;
        let body = args
            .get("body")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let mut headers = HeaderMap::new();
        if let Some(hv) = args.get("headers").and_then(|v| v.as_object()) {
            for (k, v) in hv {
                if let Some(val_str) = v.as_str() {
                    if let Ok(name) = HeaderName::from_str(k) {
                        if let Ok(val) = HeaderValue::from_str(val_str) {
                            headers.insert(name, val);
                        }
                    }
                }
            }
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .connect_timeout(Duration::from_secs(timeout))
            .build()?;

        let req_builder = match method.as_str() {
            "GET" => client.get(url),
            "POST" => client.post(url),
            "PUT" => client.put(url),
            "PATCH" => client.patch(url),
            "DELETE" => client.delete(url),
            "HEAD" => client.head(url),
            _ => return Err(anyhow!("Unsupported method")),
        };
        let mut req = req_builder.headers(headers);
        if let Some(b) = body {
            req = req.body(b);
        }

        let resp: Response = req
            .send()
            .with_context(|| format!("Request failed for {}", url))?;
        let status = resp.status().as_u16();
        let final_url = resp.url().to_string();
        let mut resp_headers = serde_json::Map::new();
        for (name, value) in resp.headers().iter() {
            resp_headers.insert(name.to_string(), json!(value.to_str().unwrap_or("")));
        }
        let mut text = resp.text().unwrap_or_default();
        let truncated = text.len() > max_bytes;
        if truncated {
            text.truncate(max_bytes);
        }
        Ok(json!({
            "url": url,
            "final_url": final_url,
            "status": status,
            "headers": resp_headers,
            "truncated": truncated,
            "text": text,
        }))
    }
}
