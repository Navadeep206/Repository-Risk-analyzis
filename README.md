# Repository Risk Intelligence

A Python-based utility suite designed to extract, analyze, and monitor risks in software repositories. It supports repository cloning, metadata analytics, commit history extraction, contributor contribution tracking, and code change complexity analyses.

## Directory Structure

```text
repository-risk-intelligence/
├── data/
│   ├── raw/           # Raw mined repository logs and exports
│   ├── processed/     # Extracted and structured datasets (CSV, Parquet, etc.)
│   └── repositories/  # Local clones of analyzed git repositories (ignored by Git)
├── notebooks/         # Jupyter Notebooks for interactive data analysis
├── src/
│   ├── clone_repo.py             # Utility to clone repositories for analysis
│   ├── repository_stats.py       # High-level repo health & metrics compiler
│   ├── commit_extractor.py       # Detailed commit log mining
│   ├── contributor_extractor.py  # Ownership distribution & bus factor analysis
│   └── modification_extractor.py  # File-level churn & complexity tracking
├── tests/             # Unit and integration tests
├── requirements.txt   # Project dependencies
├── README.md          # Project documentation
└── .gitignore         # Version control exclusion file
```

## Features Roadmap

1. **Clone Repo**: Automated target repository retrieval.
2. **Repository Stats**: Quick stats like total commits, active contributors, file count, and directory depth.
3. **Commit Extractor**: Extract commit hash, author, date, message, and modifications.
4. **Contributor Extractor**: Track contribution metrics per developer, identify core maintainers, and estimate the "Bus Factor".
5. **Modification Extractor**: File-level change counts, code line churn (lines added/deleted), and cyclomatic complexity changes.

## Setup & Quick Start

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run analyses:
   Check the command line tools in the `src/` directory.
