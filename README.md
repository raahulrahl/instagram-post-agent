<p align="center">
  <img src="https://raw.githubusercontent.com/getbindu/create-bindu-agent/refs/heads/main/assets/light.svg" alt="bindu Logo" width="200">
</p>

<h1 align="center">Instagram Post Agent</h1>
<h3 align="center">AI-Powered Instagram Marketing Content Generator</h3>

<p align="center">
  <strong>An intelligent multi-agent system that creates complete Instagram marketing campaigns—analyzing products, researching competitors, generating ad copy, and designing visual concepts—all in one seamless workflow.</strong>
</p>

<p align="center">
  <a href="https://github.com/Paraschamoli/instagram-post-agent/actions/workflows/main.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/Paraschamoli/instagram-post-agent/main.yml?branch=main" alt="Build Status">
  </a>
  <a href="https://github.com/Paraschamoli/instagram-post-agent/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Paraschamoli/instagram-post-agent" alt="License">
  </a>
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/framework-bindu-purple" alt="Built with Bindu">
</p>

---

## 📖 Overview

The **Instagram Post Agent** transforms product websites and marketing briefs into complete Instagram campaigns. Using a sophisticated multi-agent **CrewAI** system, it performs market analysis, generates compelling ad copy, and creates detailed visual concepts—all tailored to your brand and target audience.

**Key Capabilities:**
- 🔍 **Market Intelligence**: Analyzes websites and competitors to understand positioning
- ✍️ **Creative Copywriting**: Generates multiple ad copy options with hashtags and CTAs
- 🎨 **Visual Concept Design**: Creates detailed photograph descriptions for visual content
- 🎯 **Audience Targeting**: Adapts content tone and style to specific demographics
- ⚡ **Multi-Agent Collaboration**: 5 specialized AI agents work in sequence for optimal results

[Postman Collection](https://raahul-1409c5b4-717533.postman.co/workspace/getbindu's-Workspace~44eb7cfe-a752-4114-8a1a-631395f07bf1/collection/50606358-08d63179-1b6c-4d68-9d56-175631495915?action=share&creator=50606358)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys for required services (all have free tiers available)

### Installation

```bash
# Clone the repository
git clone https://github.com/Paraschamoli/instagram-post-agent.git
cd instagram-post-agent

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
```

### Configuration
Edit `.env` and add your API keys:

| Key | Description | Required | Get From |
| --- | --- | --- | --- |
| `OPENROUTER_API_KEY` | For LLM calls (GPT-4o recommended) | ✅ Yes | OpenRouter |
| `SERPER_API_KEY` | For search functionality | ✅ Yes | Serper.dev |
| `BROWSERLESS_API_KEY` | For web scraping | ✅ Yes | Browserless.io |
| `MEM0_API_KEY` | For memory features | Optional | Mem0.ai |
| `OPENAI_API_KEY` | Alternative to OpenRouter | Optional | OpenAI |

### Run the Agent

```bash
# Start the agent
uv run python -m instagram_post_agent

# Agent will be available at http://localhost:3773
```

---

## 💡 Usage

### Example Queries

```bash
# Basic product marketing
"Create Instagram content for https://ecobags.com sustainable tote bags"

# Detailed campaign with target audience
"Generate Instagram ads for fitness app at https://fitapp.com, target fitness enthusiasts 25-40"

# Luxury brand campaign
"Create luxury Instagram campaign for skincare line at https://luxuryskincare.com, target women 30-50"
```

### How It Works
1. **Ingestion**: The agent receives your product website and details.
2. **Analysis**: Specialized agents analyze the website and competitors.
3. **Strategy**: Marketing strategist develops campaign approach.
4. **Copy Creation**: Content creator generates multiple ad copy options.
5. **Visual Design**: Photographer and creative director create visual concepts.
6. **Compilation**: All elements are combined into a complete marketing package.

### Output Structure

```text
**Instagram Marketing Content**

**Ad Copy Options:**
Option 1: [Engaging ad copy with hashtags and CTA]
Option 2: [Alternative approach]
Option 3: [Creative variation]

**Photograph Descriptions:**
Photograph 1: [Detailed visual concept description]
Photograph 2: [Alternative visual approach]
Photograph 3: [Creative visual concept]

**Marketing Parameters:**
- Product: [Product name]
- Target Audience: [Audience description]
- Campaign Tone: [Tone/voice]
- Key Message: [Core message]
- Call to Action: [CTA]
- Hashtags: [Relevant hashtags]
```

---

## 🔌 API Usage

The agent exposes a RESTful API compatible with the Bindu protocol.

### Send Message Endpoint
`POST http://localhost:3773/chat`

**Request:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Create Instagram content for https://example.com eco-friendly product"
    }
  ]
}
```

**Response:**

```json
{
  "response": "**Instagram Marketing Content**\n\n**Ad Copy Options:**\nOption 1: Discover sustainable living..."
}
```

For complete API documentation, visit:
[📚 Bindu API Reference](https://docs.getbindu.com)

---

## 🎯 Skills

**instagram-post (v1.0.0)**

**Primary Capability:**
Generating complete Instagram marketing campaigns from product information.

**Features:**
*   Website analysis and competitor research
*   Multi-option ad copy generation
*   Visual concept design with detailed descriptions
*   Hashtag recommendation and call-to-action optimization
*   Target audience analysis and tone adaptation

**Best Used For:**
*   Product launches and marketing campaigns
*   Social media content creation
*   Brand awareness campaigns
*   Visual content planning
*   Competitor analysis and positioning

**Not Suitable For:**
*   Actual image generation (use DALL-E/Midjourney)
*   Video content creation
*   Real-time social media posting
*   Analytics and performance tracking

---

## 🐳 Docker Deployment

### Local Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Agent will be available at http://localhost:3773
```

### Docker Configuration
Ensure your `.env` file is populated before building. The Docker container maps port 3773 by default.

```yaml
version: '3.8'
services:
  agent:
    build: .
    ports:
      - "3773:3773"
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
      - BROWSERLESS_API_KEY=${BROWSERLESS_API_KEY}
```

---

## 🌐 Deploy to bindus.directory

Make your agent discoverable worldwide on the Internet of Agents.

### Setup GitHub Secrets
Go to your repo **Settings > Secrets and variables > Actions**.

Add the following secrets:
*   `BINDU_API_TOKEN`: Your API key from bindus.directory.
*   `DOCKERHUB_TOKEN`: Your Docker Hub access token.
*   `DOCKERHUB_USERNAME`: Your Docker Hub username.

### Deploy

```bash
# Push to main to trigger automatic deployment
git push origin main
```

---

## 🛠️ Development

### Project Structure

```text
instagram-post-agent/
├── instagram_post_agent/
│   ├── main.py                     # 🧠 Core logic (CrewAI orchestration)
│   ├── skills/                     # Bindu skill definitions
│   └── agent_config.json           # Agent metadata
├── tests/
│   └── test_main.py                # Pytest suite
├── .env.example                    # Env var template
├── pyproject.toml                  # Dependencies (uv)
└── Dockerfile                      # Production build definition
```

### Running Tests

```bash
# Run unit tests
uv run pytest

# Run with coverage
uv run pytest --cov=instagram_post_agent tests/
```

---

## 🤝 Contributing

We welcome contributions! Whether it's adding new marketing templates, improving the agent prompts, or enhancing the visual concept generation.

1.  Fork the repo.
2.  Create a branch: `git checkout -b feature/visual-enhancements`
3.  Commit changes.
4.  Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Powered by Bindu

Built with the **Bindu Agent Framework**.

**Why Bindu?**
*   ⚡ **Zero-config setup**: Focus on logic, not infrastructure.
*   🛠️ **Production-ready**: Built-in HTTP server, protocol handling, and dockerization.
*   🌐 **Interoperable**: Ready for the Internet of Agents.

<p align="center">
  <strong>Built with ❤️ by Paras Chamoli</strong>
</p>

<p align="center">
  <a href="https://github.com/Paraschamoli/instagram-post-agent/stargazers">⭐ Star this repo</a> •
  <a href="https://discord.gg/3w5zuYUuwt">💬 Join Discord</a> •
  <a href="https://bindus.directory">🌐 Agent Directory</a>
</p>
