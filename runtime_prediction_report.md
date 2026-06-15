# Repository Risk Intelligence — Runtime Prediction and Progress Tracking Report

This document outlines the design, architecture, and mathematical modeling of the real-time progress tracking and runtime estimation system implemented in the Repository Risk Analyzer.

---

## 1. System Architecture and Pipeline Stages

To deliver a premium, responsive UX, the analysis execution is divided into eight pipeline stages, modeled after enterprise tools like GitHub Actions and Datadog. 

The stages are defined as follows:

| Stage | Name | Description | Progress Weight |
|---|---|---|---|
| **Stage 1** | Clone Repository | Git shallow clone of repository code base | 20% |
| **Stage 2** | Mine Commit History | Traverse commits using `pydriller` to extract authorship and metadata | 35% |
| **Stage 3** | Mine Contributors | Calculate contributor counts, bus factor, and ownership concentrations | 10% |
| **Stage 4** | Mine File Modifications | Iterate over git history per file to compute frequency and churn | 5% |
| **Stage 5** | Compute Quality Metrics | AST parsing of python/JS/TS files to extract LOC, complexity, maintainability | 20% |
| **Stage 6** | Feature Engineering | Align metadata and engineering metrics into a feature matrix | 5% |
| **Stage 7** | Risk Prediction | Execute ML model inference using production XGBoost weights | 3% |
| **Stage 8** | Generate Report | Finalize the assessment and compile the download ready PDF report | 2% |

### Weighted Progress Calculation

The overall progress bar displays the weighted sum of individual stage progresses:

$$\text{Overall Progress} = \sum_{i=1}^{8} W_i \times P_i$$

Where:
- $W_i$ is the weight of Stage $i$ (e.g., $W_2 = 0.35$).
- $P_i$ is the progress of Stage $i$ represented as a fraction between $0.0$ and $1.0$.

---

## 2. Repository Statistics Heuristics

Before mining begins, metadata is fetched from the public GitHub API. If rate limits are exceeded, the system falls back to a set of standardized, scale-sensitive heuristics.

### Heuristic Definitions

1. **Age in Days ($D$)**:
   Calculated by finding the span between the current system time and the repository's creation timestamp:
   $$D = \text{Current Time} - \text{Created At Time}$$

2. **File Count Estimate ($F$)**:
   Estimated from the repository disk size in kilobytes ($S_{\text{kb}}$):
   $$F = \max(10, \min(100000, \lfloor S_{\text{kb}} \times 0.05 \rfloor))$$

3. **Commit Count Estimate ($C$)**:
   Since the codebase uses a shallow clone of `depth=200`, the actual analyzed commits are capped. However, the total repository commits estimate is modeled using forks count ($K$) and age ($D$):
   $$C = \max(10, \lfloor K \times 2.5 + D \times 0.1 + 20 \rfloor)$$

4. **Contributor Count Estimate ($U$)**:
   Estimated using stargazers ($S_{\text{stars}}$) and forks ($K$):
   $$U = \max(1, \lfloor \sqrt{S_{\text{stars}}} \times 0.8 + K \times 0.05 + 2 \rfloor)$$

---

## 3. Real-Time Git Clone Parsing

To capture download status in real-time, `git clone` is executed in a subprocess with the `--progress` flag enabled:

```bash
git clone --progress --depth 200 --single-branch https://github.com/owner/repo local_path
```

### Stream Parsing Regex Mechanics

Git clone outputs progress updates to `stderr` in standard blocks (e.g., *Counting objects*, *Compressing objects*, *Receiving objects*, *Resolving deltas*). The `GitCloneParser` reads `stderr` line-by-line using a non-blocking stream:

- **Stage Percent**: Parsed using the pattern `r"(\d+)%"`.
- **Downloaded Size**: Extracted from the size portion before the pipe `|` symbol:
  $$S_{\text{bytes}} \text{ matching } r"([\d\.]+)\s*([KMGT]i?B)"$$
- **Download Speed**: Extracted from the speed portion after the pipe:
  $$V_{\text{speed}} \text{ matching } r"\|\s*([\d\.]+)\s*([KMGT]i?B/s)"$$

Remaining download size ($S_{\text{rem}}$) is estimated based on the current percentage ($P_{\text{clone}}$):

$$S_{\text{total\_est}} = \frac{S_{\text{downloaded}}}{P_{\text{clone}}}$$
$$S_{\text{rem}} = \max(0, S_{\text{total\_est}} - S_{\text{downloaded}})$$

---

## 4. Execution Timing & Regression Engine

The prediction engine estimates the analysis runtime ($T$) in seconds using two approaches: a baseline heuristic model and a linear regression model trained dynamically on previous runs.

### Model 1: Heuristic Baseline
Used when there are fewer than 3 historical runs in the database.
$$T_{\text{heuristic}} = \beta_0 + \beta_1 F + \beta_2 C + \beta_3 U$$
- $\beta_0 = 3.0$ seconds (base setup / clone overhead)
- $\beta_1 = 0.015$ seconds/file (static analysis LOC parsing)
- $\beta_2 = 0.020$ seconds/commit (git history traversal)
- $\beta_3 = 0.050$ seconds/contributor (entropy calculation)

### Model 2: Linear Regression Model (Normal Equations)
When $N \ge 3$ run histories are available in `data/runtime_history.csv`, a regression model is fit:
$$y = X \beta + \epsilon$$

Where:
- $y \in \mathbb{R}^N$ is the vector of actual measured run times.
- $X \in \mathbb{R}^{N \times 4}$ is the feature matrix containing a bias column (1s), files count, commits count, and contributors count.
- $\beta \in \mathbb{R}^4$ is the parameter vector representing setup overhead, time-per-file, time-per-commit, and time-per-contributor.

The parameter vector is computed analytically using the normal equations (via least-squares solver):
$$\beta = (X^T X)^{-1} X^T y$$

To prevent unstable predictions (e.g., negative or excessively low times), predictions are bound:
$$T_{\text{final}} = \max(3.0, \max(0.3 \times T_{\text{heuristic}}, X_{\text{new}} \beta))$$

---

## 5. Watchdog Stall Detector

To prevent the UI from appearing frozen, a watchdog timer monitors the delta since the last progress update:

$$\Delta t = t_{\text{current}} - t_{\text{last\_activity}}$$

If $\Delta t > 60$ seconds, the watchdog triggers a warning state:
- An orange warning banner is injected into the pipeline stages UI: **"Analysis appears slower than expected."**
- This ensures users are notified of slow network fetches or disk bottlenecks without the page freezing.

---

## 6. History Database

Runs are saved locally to `data/runtime_history.csv` with the schema:
- `repository_name`: Owner/Name of analyzed repository.
- `files`: Actual number of files analyzed.
- `commits`: Actual number of commits traversed.
- `contributors`: Actual number of contributors.
- `runtime`: Measured duration of pipeline run in seconds.
- `timestamp`: Time of analysis (ISO format, UTC).
