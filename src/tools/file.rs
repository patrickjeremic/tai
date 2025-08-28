use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use globset::{Glob, GlobSetBuilder};
use ignore::WalkBuilder;
use llm::builder::ParamBuilder;
use llm::chat::ParameterProperty;
use regex::RegexBuilder;

use crate::tools::dir::resolve_path;

use super::Tool;

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
