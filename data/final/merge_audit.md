# Merge Operations Integrity Audit

This document audits all data merging operations within the dataset builder pipeline.

## 1. Metric Merger Audit (`quality_metrics/metric_merger.py`)
- **Action**: Combines `python_metrics.csv`, `javascript_metrics.csv`, and `typescript_metrics.csv` via concatenation (`pd.concat`).
- **Keys**: No explicit keys are merged here since it is a row concatenation of unified schemas.
- **Key Mismatch Risks**: None.

## 2. Multi-Repository Miner Audit (`multi_repository_miner.py`)
- **Action**: Compiles stats per repository into `repositories_metadata.csv` and merges quality dataframes.
- **Keys**: 
  - `repository_name`
- **Integrity**: Correctly maps dominant language, total LOC, commit counts, and unique contributors.

## 3. Data Merge Stage Audit (`merge_repository_data.py`)
- **Action**: Combines static quality metrics (`quality_metrics.csv`) with git modifications (`{repo}_modifications.csv`).
- **Keys**:
  - `file_path`
  - `repository_name`
- **Method**: Left Join. `df_quality` is the base dataframe; modifications features are grouped by `file_path` and left-joined.
- **Integrity & Drop-offs**:
  - Because it is a **Left Join**, no files are removed from the quality metrics during the merge (any files without git modifications will simply have modification counts set to 0 and other git features filled with default values).
  - Path casing and format: both files resolve paths relative to the repository base directory using standard Unix forward slashes (`/`), matching exactly.

## 4. Key Matching Check
| Join Stage | Left Key | Right Key | Join Type | Case-Sensitive | Path Casing / Separator | Row Drops? |
| --- | --- | --- | --- | --- | --- | --- |
| Quality metrics merge | N/A (row concat) | N/A (row concat) | Concatenation | N/A | Unified forward slashes | No |
| History + Quality merge | `file_path` | `file_path` | Left Join | Yes | Unified forward slashes | No |
