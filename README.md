# Benchmark Intelligence Agent

> Automatically track trending benchmarks across LLMs, VLMs, and audio-language models

[![Status](https://img.shields.io/badge/status-ready-success)](https://github.com) [![Python](https://img.shields.io/badge/python-3.9+-blue)](https://python.org) [![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

---

## 📊 Latest Report

**[View Latest Benchmark Report →](agents/benchmark_intelligence/reports/trending_benchmarks_20260411_000851.md)**

**Key Findings** (2026-04-11):
- **263 unique benchmarks** discovered from 11,281 mentions
- **167 models** analyzed from 18 major labs
- **Vision + Text extraction** - both sources used for complete coverage
- **Top benchmarks**: MMLU Pro (57 models), MMLU (57), GPQA Diamond (48)
- **14 categories**: Vision (61 benchmarks), Coding (37), Knowledge (25), Math (24)
- **Unicode normalization**: τ²-Bench = τ2-Bench merged successfully
---

## 🎯 What This Does

This AI agent automatically:

1. **Discovers trending models** from major labs (Qwen, Meta, Mistral, Google, Microsoft, etc.)
2. **Extracts benchmarks** from model cards and arXiv papers using AI (text + vision extraction)
3. **Analyzes charts & figures** using Claude vision AI to extract benchmarks from PDFs
4. **Consolidates variations** (GSM8K ≈ gsm8k ≈ GSM-8K, τ²-Bench = τ2-Bench) using Unicode normalization and AI validation
5. **Classifies benchmarks** into categories using Claude AI
6. **Tracks trends** over time with SQLite caching and snapshots
7. **Generates reports** showing evolution and emerging patterns

**Run it monthly** to stay current with the AI evaluation landscape.

---

## 🚀 Quick Start

### Execution Modes

The agent supports **2 execution paths** with **7 individual stages** or a full pipeline:

#### 1. **Python Direct Execution** (Recommended for Development)

```bash
# Full pipeline (all 6 stages)
python -m agents.benchmark_intelligence.main generate

# Individual stages (for debugging/development)
python -m agents.benchmark_intelligence.main filter_models
python -m agents.benchmark_intelligence.main find_docs
python -m agents.benchmark_intelligence.main parse_docs --concurrency 30
python -m agents.benchmark_intelligence.main consolidate_benchmarks
python -m agents.benchmark_intelligence.main categorize_benchmarks
python -m agents.benchmark_intelligence.main report
```

#### 2. **Ambient Workflow Execution** (Recommended for Production)

```bash
# Full pipeline
/benchmark_intelligence.generate

# Individual stages
/benchmark_intelligence.filter_models
/benchmark_intelligence.find_docs
/benchmark_intelligence.parse_docs --concurrency 30
/benchmark_intelligence.consolidate_benchmarks --from-db
/benchmark_intelligence.categorize_benchmarks
/benchmark_intelligence.report
```

### Setup

#### On Ambient Code Platform (Recommended)

```bash
# 1. Set HuggingFace token in Workspace Settings → Environment Variables
# HF_TOKEN = "hf_..."

# 2. Run via Ambient workflow
/benchmark_intelligence.generate

# Or via Python
cd /workspace/repos/trending_benchmarks
python -m agents.benchmark_intelligence.main generate
```

#### On Other Platforms

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys
export HF_TOKEN="your_huggingface_token"
export ANTHROPIC_API_KEY="your_claude_key"  # Not needed on Ambient

# 3. Run
python -m agents.benchmark_intelligence.main generate
```

**Expected runtime**: ~50-60 minutes for 65 models (with AI extraction, default concurrency: 20)

---

## 📚 Documentation & Configuration

### Core Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| **[benchmark_taxonomy.md](benchmark_taxonomy.md)** | Complete reference of 30+ benchmarks | Root |
| **[categories.yaml](categories.yaml)** | 13 benchmark categories & definitions | Root |
| **[config.yaml](config.yaml)** | Target labs/organizations to track | Config |

### Reports & Data

| Resource | Description |
|----------|-------------|
| **[Latest Report](agents/benchmark_intelligence/reports/trending_benchmarks_20260411_000851.md)** | Most recent benchmark intelligence |
| **[All Reports](agents/benchmark_intelligence/reports/)** | Historical snapshots |
| **[SQLite Database](benchmark_cache.db)** | Queryable cache (see below) |

---

## 💾 Caching System

The agent uses **SQLite** for intelligent caching with change detection:

### Database Schema

```
benchmark_cache.db
├── models           # Model metadata (name, lab, release_date, downloads)
├── benchmarks       # Unique benchmarks with categories
├── model_benchmarks # Benchmark scores/results per model
├── documents        # Cached model cards & docs (content-hash tracking)
└── snapshots        # Temporal snapshots for trend analysis
```

### How Caching Works

1. **Content-hash tracking**: Models are only reprocessed if their model card changes
2. **Incremental updates**: Subsequent runs only process new/changed models
3. **Historical snapshots**: Trend analysis without re-fetching old data
4. **Queryable**: Use SQL for custom analysis

### Query Examples

```bash
sqlite3 benchmark_cache.db

# Show all discovered models
SELECT id, lab, release_date, downloads, likes
FROM models
ORDER BY downloads DESC LIMIT 20;

# Top benchmarks by usage
SELECT b.canonical_name, COUNT(DISTINCT mb.model_id) as model_count, b.categories
FROM benchmarks b
JOIN model_benchmarks mb ON b.id = mb.benchmark_id
GROUP BY b.canonical_name
ORDER BY model_count DESC
LIMIT 15;

# Models released in last 12 months
SELECT id, lab, release_date, downloads
FROM models
WHERE release_date >= date('now', '-12 months')
ORDER BY release_date DESC;

# Benchmark trend over time
SELECT s.timestamp, s.benchmark_count, s.model_count
FROM snapshots s
ORDER BY s.timestamp;
```

### Cache Location

- **File**: `benchmark_cache.db` (in project root)
- **Size**: ~240KB (current)
- **Backed up**: Yes (snapshots table tracks history)

---

## 📊 What You Get

### 📝 Generated Reports

**7 automated sections**:

1. **Executive Summary**: Models & benchmarks tracked
2. **Trending Models**: Sorted by release date & significance
3. **Most Common Benchmarks**: All-time + monthly trends
4. **Emerging Benchmarks**: Recently introduced (<90 days)
5. **Category Distribution**: Breakdown by type (charts)
6. **Lab Insights**: Per-lab statistics & preferences
7. **Temporal Trends**: Evolution over time

### 📁 Historical Tracking

Timestamped reports in `agents/benchmark_intelligence/reports/`:
```
reports/
├── trending_benchmarks_20260410_155422.md  # Latest
└── ...
```

---

## 🏗️ Architecture

```
Discover Models (HuggingFace API)
    ↓
Check Cache (content-hash comparison)
    ↓
Parse Documents (model cards, arXiv papers - if changed)
    ↓
Extract Benchmarks (Claude AI: text + vision for charts/figures)
    ↓
Consolidate Names (Unicode normalization + fuzzy matching + AI validation)
    ↓
Classify Benchmarks (multi-label AI categorization)
    ↓
Store in SQLite Cache
    ↓
Create Temporal Snapshot
    ↓
Generate Markdown Report
```

**Key Components**:
- **HuggingFace Client**: Official `huggingface_hub` library
- **Universal Claude Client**: Auto-detects Ambient/Vertex AI/Anthropic API
- **AI Extraction**: Claude-powered parsing of model cards and arXiv papers (text + vision)
- **Vision AI**: Extracts benchmarks from charts/figures in PDFs using Claude vision API
- **PDF Processing**: `pdfplumber` for embedded image extraction from research papers
- **Cache Manager**: SQLite with content-hash change detection
- **Smart Consolidation**: Unicode normalization + AI validation ("MMLU", "MMLU-Pro", τ²-Bench = τ2-Bench)

---

## 📚 Benchmark Taxonomy

### Categories (13)

**Knowledge** • **Reasoning** • **Math** • **Code** • **Vision** • **Audio** • **Multilingual** • **Safety** • **Long-Context** • **Instruction-Following** • **Tool-Use** • **Agent** • **Domain-Specific**

### Top Benchmarks Tracked (30+)

**Knowledge**: MMLU, MMLU-Pro, C-Eval, CMMLU, TriviaQA, GPQA
**Math**: GSM8K, MATH, AIME, Gaokao
**Code**: HumanEval, MBPP, LiveCodeBench, CFBench
**Vision**: MMMU, CMMMU, VQAv2, DocVQA, AI2D
**Reasoning**: ARC, BBH, HellaSwag, PIQA, WinoGrande, BoolQ
**Safety**: TruthfulQA, RewardBench
**Multimodal**: Open LLM Leaderboard, Arena-Hard

**See [benchmark_taxonomy.md](benchmark_taxonomy.md) for complete reference with definitions.**

---

## 🎯 Target Labs (15)

- **Qwen** • **01-ai** (Yi)
- **meta-llama** • **mistralai** • **google**
- **microsoft** • **anthropic**
- **alibaba-pai** • **tencent** • **deepseek-ai**
- **OpenGVLab** • **THUDM** (ChatGLM)
- **baichuan-inc** • **internlm**
- **MinimaxAI**

Configure in [`config.yaml`](config.yaml)

---

## ⚙️ Configuration

### Discovery Settings

Edit [`config.yaml`](config.yaml):

```yaml
discovery:
  models_per_lab: 15           # Models to fetch per lab
  sort_by: "downloads"         # downloads | trending | lastModified
  filter_tags: []              # Task filters (empty = all)
  min_downloads: 1000          # Minimum popularity threshold
  date_filter_months: 12       # Only models from last N months
  exclude_tags:                # Skip these model types
    - "time-series-forecasting"
    - "fill-mask"

# Concurrency settings
parallelization:
  max_concurrent_document_fetches: 5
  enabled: true
  timeout_per_document_seconds: 60

# Rate limiting (prevents API 429 errors)
rate_limiting:
  huggingface:
    requests_per_minute: 60
    max_retries: 5
    initial_backoff_seconds: 2.0
  anthropic:
    requests_per_minute: 50
    max_retries: 5
  arxiv:
    requests_per_minute: 30
    max_retries: 3
```

### Concurrency Settings

**Default**: 20 concurrent workers for document parsing

**Adjust based on your needs**:

```bash
# Low concurrency (safer, slower)
python -m agents.benchmark_intelligence.main parse_docs --concurrency 10

# High concurrency (faster, may hit rate limits)
python -m agents.benchmark_intelligence.main parse_docs --concurrency 50

# Ambient workflow
/benchmark_intelligence.parse_docs --concurrency 30
```

### JSON Output Locations

All outputs are saved to `agents/benchmark_intelligence/outputs/`:

| Stage | Output File | Schema |
|-------|-------------|--------|
| **filter_models** | `filtered_models/models_YYYYMMDD_HHMMSS.json` | `[{id, author, downloads, likes, tags, created_at}]` |
| **find_docs** | `docs/docs_YYYYMMDD_HHMMSS.json` | `[{model_id, documents: [{type, url, found}]}]` |
| **parse_docs** | `parsed/parsed_YYYYMMDD_HHMMSS.json` | `[{model_id, benchmarks: [{name, score, metric}]}]` |
| **consolidate** | `consolidated/benchmarks_YYYYMMDD_HHMMSS.json` | `[{canonical_name, occurrences, models: [...]}]` |
| **categorize** | `categorized/categorized_YYYYMMDD_HHMMSS.json` | `[{benchmark_name, category, subcategory, confidence}]` |
| **report** | `reports/report_YYYYMMDD_HHMMSS.md` | Markdown report |

### Categories & Taxonomy

- **Categories**: Edit [`categories.yaml`](categories.yaml) at root
- **Taxonomy**: Update [`benchmark_taxonomy.md`](benchmark_taxonomy.md) at root

### Scheduling

**Monthly runs** (recommended):

```bash
# Via cron (automatically configured)
0 9 1 * * cd /workspace/repos/trending_benchmarks && /benchmark_intelligence.generate

# Or manual
python -m agents.benchmark_intelligence.main generate
```

---

## 🔧 Advanced Usage

### Dry Run (Test Mode)

```bash
python -m agents.benchmark_intelligence.main --dry-run --verbose
```

### Specific Labs Only

```bash
python -m agents.benchmark_intelligence.main \
  --labs Qwen,meta-llama,mistralai
```

### Force Refresh (Ignore Cache)

```bash
# Clear cache and start fresh
rm benchmark_cache.db
python -m agents.benchmark_intelligence.main
```

### Custom Date Range

```bash
# Edit config.yaml:
discovery:
  date_filter_months: 24  # Last 2 years
```

---

## 🛠️ Technical Stack

**Language**: Python 3.9+
**APIs**: HuggingFace Hub, Anthropic Claude (or Vertex AI)
**Storage**: SQLite
**AI**: Claude Sonnet 4 for intelligent extraction & classification
**Format**: Markdown, YAML, JSON

**Dependencies** (7):
- `huggingface_hub` - Model discovery
- `anthropic` - AI-powered parsing (or Vertex AI on Ambient)
- `pdfplumber` - PDF parsing and image extraction
- `pyyaml` - Configuration
- `requests` - HTTP
- `beautifulsoup4` - HTML parsing
- `python-dateutil` - Date handling

---

## 📖 Complete Documentation

| Document | Purpose |
|----------|---------|
| [AMBIENT_QUICKSTART.md](AMBIENT_QUICKSTART.md) | Get started on Ambient platform |
| [agents/.../README.md](agents/benchmark_intelligence/README.md) | Full technical documentation |
| [config.yaml](config.yaml) | Configuration reference |
| [specs/001-.../spec.md](specs/001-benchmark-intelligence/spec.md) | Complete feature specification |

---

## 🐛 Troubleshooting

### "HF_TOKEN not set"
```bash
export HF_TOKEN="hf_your_token"
```
Get token: https://huggingface.co/settings/tokens

### "ANTHROPIC_API_KEY not set"
Only needed outside Ambient. Get key: https://console.anthropic.com
On Ambient: Uses native Vertex AI Claude support (no key needed)

### Getting irrelevant models?
Edit `config.yaml` → remove labs that produce noise (e.g., "huggingface" org gets time-series models)

### Models from wrong time period?
Edit `config.yaml` → `date_filter_months: 12` (or higher)

### Cache corruption
```bash
rm benchmark_cache.db
# Re-run will rebuild from scratch
```

### Common Concurrency Issues

#### 429 Rate Limit Errors

**Symptom**: "Too many requests" errors from APIs

**Solutions**:
1. **Reduce concurrency**:
   ```bash
   python -m agents.benchmark_intelligence.main parse_docs --concurrency 10
   ```

2. **Adjust rate limits** in `config.yaml`:
   ```yaml
   rate_limiting:
     huggingface:
       requests_per_minute: 30  # Lower = safer
   ```

3. **Rate limiter automatically retries** with exponential backoff

#### Timeout Errors

**Symptom**: "Connection timeout" or "Read timeout"

**Solutions**:
1. **Increase timeout** in `config.yaml`:
   ```yaml
   parallelization:
     timeout_per_document_seconds: 120  # Default: 60
   ```

2. **Reduce concurrent fetches**:
   ```yaml
   parallelization:
     max_concurrent_document_fetches: 3  # Default: 5
   ```

#### Memory Issues

**Symptom**: Process killed or "Out of memory"

**Solutions**:
1. **Lower concurrency** (fewer workers = less memory):
   ```bash
   python -m agents.benchmark_intelligence.main parse_docs --concurrency 5
   ```

2. **Process in batches** (run individual stages separately)

#### Connection Pool Exhausted

**Symptom**: "No available connections" or hanging requests

**Solutions**:
1. **Connection pool auto-manages** resources
2. **Check logs** for specific errors
3. **Restart** with lower concurrency

#### Resuming After Interruption

**The pipeline is resumable!** Hash cache prevents re-processing:

```bash
# If interrupted, just re-run the same command
python -m agents.benchmark_intelligence.main generate

# Hash cache will skip already-processed documents
# Only new/changed documents will be processed
```

---

## 📋 Requirements

- Python 3.9 or higher
- HuggingFace account (for API token)
- Anthropic API key (for Claude) OR Ambient Code Platform
- Internet connection
- ~500MB disk space (for cache)

---

## 📜 License

Apache 2.0 - See [LICENSE](LICENSE) file

---

## 🔗 Links

- **[Latest Report](agents/benchmark_intelligence/reports/trending_benchmarks_20260411_000851.md)** ⭐
- **[Feature Specification](specs/001-benchmark-intelligence/spec.md)** - Complete requirements
- **[Benchmark Taxonomy](benchmark_taxonomy.md)** - Complete reference
- **[Categories](categories.yaml)** - Category definitions
- [HuggingFace Hub](https://huggingface.co)
- [Anthropic Claude](https://anthropic.com)

---

## 📊 Status

**Version**: 1.1.0
**Status**: ✅ Production Ready
**Last Run**: 2026-04-11
**Models**: 167 | **Benchmarks**: 263 | **Categories**: 14
**Features**: Vision AI extraction • Unicode normalization • AI validation • Multi-source parsing

---

**Built with ❤️ using AI • Powered by Claude & HuggingFace**
