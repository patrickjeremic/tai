
# 🤖 TAI - Terminal AI Assistant

> Your intelligent command-line companion for getting things done faster

TAI is a powerful terminal-based AI assistant that bridges the gap between natural language and command-line operations. Ask questions, get commands, and execute them with confidence - all from your terminal.

## ✨ Features

- **🧠 Smart Command Generation** - Ask for what you want, get the exact command
- **📋 Clipboard Integration** - Copy commands before executing
- **📚 Context-Aware** - Load project-specific knowledge automatically  
- **⚙️ Flexible Configuration** - Customize model, temperature, and global contexts
- **📖 Conversation History** - Remembers previous interactions for better context
- **🔄 Interactive Mode** - Type `tai` and start chatting
- **🎯 Multi-line Support** - Perfect for complex queries

## 🚀 Quick Start

```bash
# Install TAI
cargo install --path .

# Basic usage
tai "list all running processes"

# Interactive mode
tai
> how do I find large files?
> 

# Use project context
tai --context rust "optimize my build"
```

## 📦 Installation

### Prerequisites
- Rust 1.70+ 
- An Anthropic API key (or local Ollama setup)

### Build from source
```bash
git clone https://github.com/yourusername/tai
cd tai
cargo build --release
cp target/release/tai ~/.local/bin/  # or your preferred PATH location
```

### Setup API Key
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
# Or configure it in TAI's config system
tai config anthropic_api_key "your-api-key-here"
```

## 🎯 Usage

### Basic Commands

```bash
# Ask a question
tai "how do I compress a folder with tar?"

# Execute a task  
tai "show me all Python files larger than 1MB"

# Clear conversation history
tai --clear-history
```

### Interactive Mode

```bash
tai
> I need to find all files modified in the last 24 hours
> that contain the word "TODO" in them
> 
```

### Context System

```bash
# Use specific context
tai --context urbit "how do I commit my desk changes?"

# Skip context loading  
tai --nocontext "what's the current time?"

# Create a context file
echo "# My Project Context..." > .context.tai
tai "deploy my application"  # automatically uses local context
```

### Configuration

```bash
# View all settings
tai config

# Set model globally
tai config model claude-3-opus --global

# Set temperature for current project
tai config temperature 0.8

# Add global contexts
tai config global_contexts "rust,docker,kubernetes"
```

## 🏗️ Context System

TAI supports multiple levels of context to make interactions more relevant:

### Local Context
Create `.context.tai` in your project directory:
```markdown
# My Web App
- Built with Next.js and TypeScript  
- Database: PostgreSQL with Prisma
- Deployed on Vercel
- Uses TailwindCSS for styling
```

### Named Contexts
Store reusable contexts in `~/.config/tai/context/`:
```bash
# ~/.config/tai/context/docker.context.tai
# ~/.config/tai/context/kubernetes.context.tai
# ~/.config/tai/context/rust.context.tai
```

### Global Contexts
Configure contexts to load automatically:
```bash
tai config global_contexts "docker,rust"
```

## ⚙️ Configuration

TAI uses a hierarchical configuration system:

1. **Environment variables** (highest priority)
2. **Local config** (`.config.tai` in project/git root)  
3. **Global config** (`~/.config/tai/config.tai`)

### Configuration Options

| Key | Description | Example |
|-----|-------------|---------|
| `model` | AI model to use | `claude-3-5-sonnet-latest` |
| `temperature` | Response creativity (0.0-2.0) | `0.7` |
| `max_tokens` | Maximum response length | `1500` |
| `anthropic_api_key` | API key (fallback) | `sk-ant-...` |
| `global_contexts` | Always-loaded contexts | `rust,docker` |

### Example Config File

```toml
# ~/.config/tai/config.tai
model = "claude-3-5-sonnet-latest"
temperature = 0.1
max_tokens = 2000
global_contexts = ["rust", "git", "docker"]
```

## 🔧 Command Reference

### Execution Options
When TAI suggests a command, you have three choices:
- **Y** - Execute the command
- **n** - Cancel execution  
- **c** - Copy to clipboard

### Command-line Flags
```bash
tai [OPTIONS] [MESSAGE...]

Options:
    --context <NAME>     Load specific context
    --nocontext         Skip context loading
    --clear-history     Clear conversation history
    
Subcommands:
    config              Manage configuration
```

### Config Subcommand
```bash
tai config                    # Show all settings
tai config <key>              # Get specific value  
tai config <key> <value>      # Set locally
tai config <key> <value> --global  # Set globally
```

## 📁 File Structure

```
~/.config/tai/
├── config.tai                 # Global configuration
└── context/
    ├── rust.context.tai       # Rust development context
    ├── docker.context.tai     # Docker context
    └── urbit.context.tai      # Urbit context

# In your project
.config.tai                    # Project configuration  
.context.tai                   # Project context
```

## 🤝 Contributing

We welcome contributions! Here are some ways to help:

- 🐛 **Report bugs** - Found an issue? Let us know!
- 💡 **Suggest features** - Have ideas for improvements?  
- 📝 **Improve docs** - Help make the documentation clearer
- 🔧 **Submit PRs** - Code contributions are greatly appreciated

### Development Setup

```bash
git clone https://github.com/yourusername/tai
cd tai
cargo build
cargo test
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built with [rllm](https://github.com/rllm-project/rllm) for AI provider abstraction
- Uses [clap](https://github.com/clap-rs/clap) for elegant CLI parsing
- Clipboard functionality powered by [arboard](https://github.com/1Password/arboard)

---

<div align="center">

**[⭐ Star this repo](https://github.com/yourusername/tai) if TAI helps you be more productive!**

Made with ❤️ for developers who love the terminal

</div>
