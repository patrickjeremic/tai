use once_cell::sync::Lazy;
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style as SynStyle, Theme, ThemeSet};
use syntect::parsing::{SyntaxReference, SyntaxSet};

use nu_ansi_term::{Color, Style};

static SYNTAXES: Lazy<SyntaxSet> = Lazy::new(SyntaxSet::load_defaults_newlines);
static THEMES: Lazy<ThemeSet> = Lazy::new(ThemeSet::load_defaults);
static THEME: Lazy<Theme> = Lazy::new(|| {
    THEMES
        .themes
        .get("base16-ocean.dark")
        .or_else(|| THEMES.themes.get("Solarized (dark)"))
        .cloned()
        .or_else(|| THEMES.themes.values().next().cloned())
        .unwrap_or_default()
});

fn highlight(code: &str, lang: Option<&str>) -> String {
    let syntax: &SyntaxReference = match lang.and_then(|l| SYNTAXES.find_syntax_by_token(l)) {
        Some(s) => s,
        None => SYNTAXES
            .find_syntax_by_extension("txt")
            .unwrap_or_else(|| SYNTAXES.find_syntax_plain_text()),
    };
    let mut h = HighlightLines::new(syntax, &THEME);
    let mut out = String::new();
    for line in code.lines() {
        let ranges: Vec<(SynStyle, &str)> = h.highlight_line(line, &SYNTAXES).unwrap_or_default();
        let escaped = syntect::util::as_24_bit_terminal_escaped(&ranges, false);
        out.push_str(&escaped);
        out.push('\n');
    }
    out
}

pub fn render_markdown_to_terminal(input: &str) -> String {
    let mut out = String::new();
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_TASKLISTS);
    let parser = Parser::new_ext(input, opts);

    let mut list_stack: Vec<(bool, usize)> = Vec::new();
    let mut in_code_block: Option<String> = None;
    let mut code_buffer = String::new();
    let mut heading_level: Option<HeadingLevel> = None;
    let mut _in_paragraph = false;

    for ev in parser {
        match ev {
            Event::Start(tag) => match tag {
                Tag::Paragraph => {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    _in_paragraph = true;
                }
                Tag::Heading(level, _, _) => {
                    heading_level = Some(level);
                    if !out.is_empty() {
                        out.push('\n');
                    }
                }
                Tag::BlockQuote => {
                    if !out.ends_with('\n') {
                        out.push('\n');
                    }
                }
                Tag::List(start) => {
                    list_stack.push((start.is_none(), start.unwrap_or(1) as usize));
                }
                Tag::Item => {
                    let depth = list_stack.len().saturating_sub(1);
                    let indent = "  ".repeat(depth);
                    if let Some((unordered, idx)) = list_stack.last_mut() {
                        if *unordered {
                            out.push_str(&format!("{}- ", indent));
                        } else {
                            out.push_str(&format!("{}{}. ", indent, *idx));
                            *idx += 1;
                        }
                    }
                }
                Tag::CodeBlock(kind) => {
                    in_code_block = Some(match kind {
                        CodeBlockKind::Fenced(lang) => lang.to_string(),
                        CodeBlockKind::Indented => String::new(),
                    });
                    if !out.ends_with('\n') {
                        out.push('\n');
                    }
                }
                Tag::Emphasis
                | Tag::Strong
                | Tag::Strikethrough
                | Tag::Link(_, _, _)
                | Tag::Image(_, _, _)
                | Tag::Table(_)
                | Tag::TableHead
                | Tag::TableRow
                | Tag::TableCell
                | Tag::FootnoteDefinition(_) => {}
            },
            Event::End(tag) => match tag {
                Tag::Paragraph => {
                    _in_paragraph = false;
                    out.push('\n');
                    out.push('\n');
                }
                Tag::Heading(_, _, _) => {
                    out.push('\n');
                    out.push('\n');
                    heading_level = None;
                }
                Tag::BlockQuote => {
                    out.push('\n');
                }
                Tag::List(_) => {
                    out.push('\n');
                    let _ = list_stack.pop();
                }
                Tag::Item => {
                    if !out.ends_with('\n') {
                        out.push('\n');
                    }
                }
                Tag::CodeBlock(_) => {
                    let lang = in_code_block.take().unwrap_or_default();
                    let highlighted = highlight(
                        &code_buffer,
                        if lang.trim().is_empty() {
                            None
                        } else {
                            Some(lang.trim())
                        },
                    );
                    out.push_str(&highlighted);
                    code_buffer.clear();
                    out.push('\n');
                }
                Tag::Emphasis
                | Tag::Strong
                | Tag::Strikethrough
                | Tag::Link(_, _, _)
                | Tag::Image(_, _, _)
                | Tag::Table(_)
                | Tag::TableHead
                | Tag::TableRow
                | Tag::TableCell
                | Tag::FootnoteDefinition(_) => {}
            },
            Event::Text(text) => {
                if in_code_block.is_some() {
                    code_buffer.push_str(&text);
                } else if let Some(level) = heading_level {
                    let style = match level {
                        HeadingLevel::H1 => Style::new().bold().underline().fg(Color::Cyan),
                        HeadingLevel::H2 => Style::new().bold().fg(Color::Cyan),
                        HeadingLevel::H3 => Style::new().bold().fg(Color::LightCyan),
                        _ => Style::new().bold(),
                    };
                    out.push_str(&style.paint(text.as_ref()).to_string());
                } else {
                    out.push_str(text.as_ref());
                }
            }
            Event::Code(text) => {
                let style = Style::new().fg(Color::Yellow);
                out.push_str(&style.paint(format!("`{}`", text.as_ref())).to_string());
            }
            Event::Html(html) => {
                out.push_str(html.as_ref());
            }
            Event::SoftBreak => {
                out.push('\n');
            }
            Event::HardBreak => {
                out.push('\n');
            }
            Event::Rule => {
                out.push_str(&Style::new().fg(Color::DarkGray).paint("────").to_string());
                out.push('\n');
            }
            Event::TaskListMarker(checked) => {
                if checked {
                    out.push_str("[x] ");
                } else {
                    out.push_str("[ ] ");
                }
            }
            Event::FootnoteReference(name) => {
                out.push_str(&format!("[{}]", name.as_ref()));
            }
        }
    }

    if !code_buffer.is_empty() {
        let highlighted = highlight(&code_buffer, None);
        out.push_str(&highlighted);
    }

    out
}
