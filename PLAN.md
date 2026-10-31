# `PLAN.md` — Project ResearchRadar: Automated Knowledge Manifold Optimization for Robotics

## 1. System Role & Execution Model

* **Architect / Supervisory AI:** Designs the mathematical structures, formulates discovery objectives, validates algorithmic fidelity to the KMO and EDGE literature, and verifies experimental validity.
* **Pi Agent (The "Hands"):** The local coding agent. Pi executes file I/O, configures local dependencies, calls arXiv/Ollama APIs, and implements modular Python scripts under the direction of the Supervisory AI.

---

## 2. Project Objective

Build an automated, featherweight "Research Radar" that:
1. Automatically harvests scientific literature across three conflicting robotics paradigms:
   * **Regime A:** Classical System Identification (SysID) (analytical/grey-box models, least squares, Kalman filtering).
   * **Regime B:** Domain Randomization & Sim-to-Real (stochastic physics perturbation, transfer bounds).
   * **Regime C:** Deep Reinforcement Learning & Stochastic Policies (PPO, policy gradient, sample efficiency).
2. Generates dense embeddings using **`F2LLM-4B`** running locally in **Ollama** (with **`F2LLM-0.6B`** for rapid unit tests).
3. Constructs a 2D continuous **Knowledge Manifold** using Multidimensional Scaling (MDS) and Smoothed Particle Hydrodynamics (SPH) kernel interpolation.
4. Executes a **4-Objective Pareto Optimization** ($P_{\text{boundary}}, W, G, S$) across a discrete $21 \times 21$ evaluation grid to discover non-dominated **Supported Knowledge Gaps**.
5. Outputs:
   * An **interactive 2D HTML scatter plot** (Plotly) showing the paper landscape, regime contours, and clickable Pareto gaps.
   * A **unified Markdown report** (`RADAR_REPORT.md`) containing ranked gap profiles, supporting anchor citations, local keyword provenance, and a verbatim **evidence-constrained LLM prompt** ready to paste into Claude or GPT.

---

## 3. Directory Structure

```text
research-radar/
├── PLAN.md                     # This master plan
├── requirements.txt            # Python dependencies
├── config.yaml                 # Query parameters, Ollama settings, grid size
├── src/
│   ├── __init__.py
│   ├── harvest.py              # Automated arXiv query engine
│   ├── embed.py                # Local Ollama embedding client (F2LLM)
│   ├── manifold.py             # 2D projection, SPH field, Jacobian gradient
│   ├── pareto.py               # 4-Objective Pareto dominance search
│   └── report.py               # Plotly interactive UI & Markdown generator
├── data/
│   ├── corpus.json             # Downloaded papers (metadata + abstracts)
│   └── embeddings.npy          # Cached (N, D) embedding matrix
└── output/
    ├── radar_map.html          # Interactive 2D visualization
    └── RADAR_REPORT.md         # Ranked gaps, anchor papers, and LLM prompt payload
```

---

## 4. Phased Implementation Roadmap for Pi Agent

### Phase 1: Environment & Local Inference Setup
* **Objective:** Establish the workspace and verify local communication with the Ollama embedding service.
* **Tasks for Pi Agent:**
  1. Create the project directory structure.
  2. Write `requirements.txt`:
     ```text
     requests>=2.31.0
     numpy>=1.24.0
     scipy>=1.10.0
     scikit-learn>=1.3.0
     plotly>=5.18.0
     pyyaml>=6.0
     arxiv>=2.1.0
     ```
  3. Verify Ollama endpoint availability (`http://localhost:11434/api/embed` or `/api/embeddings`) for `codefuse-ai/F2LLM-4B` (or `F2LLM-0.6B` fallback).
  4. Write `config.yaml` specifying search terms, sample sizes (e.g., 40 papers per regime = 120 total), and Ollama model tag.

### Phase 2: Literature Harvesting (`src/harvest.py`)
* **Objective:** Programmatically query and cache structured literature data without manual file downloads.
* **Tasks for Pi Agent:**
  1. Use the `arxiv` Python client to query three distinct topics:
     * *Query A (SysID):* `abs:"system identification" AND (cat:cs.RO OR cat:eess.SY)`
     * *Query B (Sim-to-Real / DR):* `abs:"domain randomization" AND abs:"sim-to-real" AND cat:cs.RO`
     * *Query C (RL Control):* `abs:"reinforcement learning" AND abs:"manipulation" AND cat:cs.RO`
  2. Normalize and deduplicate results.
  3. Extract per paper: `id`, `title`, `abstract`, `authors`, `year`, `regime_label`.
  4. Save structured metadata to `data/corpus.json`.

### Phase 3: Ollama Embedding Pipeline (`src/embed.py`)
* **Objective:** Transform text into a frozen semantic representation array using the local `F2LLM` model.
* **Tasks for Pi Agent:**
  1. Format text representations as: `"Title: " + title + " | Abstract: " + abstract`.
  2. Send batch requests to Ollama's local embedding endpoint.
  3. Cache the output tensor as `data/embeddings.npy` (Shape: $[N, 4096]$ or $[N, 2048]$ depending on model scale).
  4. Implement an MD5 cache check so re-running does not re-embed unmodified papers.

### Phase 4: Knowledge Manifold & Discovery Fields (`src/manifold.py`)
* **Objective:** Translate high-dimensional vectors into a continuous 2D metric space and construct the 4 KMO fields.
* **Tasks for Pi Agent:**
  1. **Coordinate Projection:** Apply Metric MDS (`sklearn.manifold.MDS`) to embed documents onto a bounded chart $\Omega = [-1, 1]^2$.
  2. **Grid Discretization:** Construct a regular $21 \times 21$ meshgrid (441 candidate discovery points).
  3. **SPH Kernel Interpolation:** For any coordinate $x \in [-1, 1]^2$, compute continuous document weights using a Gaussian/cubic smoothing kernel:
     $$w_i(x) = \exp\left(-\frac{\|x - x_i\|^2}{2h^2}\right), \quad \tilde{w}_i(x) = \frac{w_i(x)}{\sum_j w_j(x)}$$
  4. **Formulate the 4 Discovery Objectives:**
     * **$P_{\text{boundary}}(x)$ (Persistent Boundary):** Calculate normalized cluster entropy across regimes ($H(x)$) and probability difference between the top-two regimes ($\Delta p(x)$). Aggregate via harmonic mean:
       $$B(x) = 0.65 N_+(H(x)) + 0.35 N_-(\Delta p(x))$$
     * **$W(x)$ (Band-Pass White Space):** Compute nearest-document distance $d_{nn}(x)$. Apply empirical quantile band-pass filter ($q_{20}, q_{60}, q_{85}, q_{98}$) so neither incremental ($< q_{20}$) nor wild extrapolation ($> q_{98}$) wins:
       $$W(x) = \begin{cases} 0 & d_{nn} \le q_{20} \text{ or } d_{nn} \ge q_{98} \\ 1.0 & q_{60} \le d_{nn} \le q_{85} \\ \text{linear ramp} & \text{otherwise} \end{cases}$$
     * **$G(x)$ (Semantic Gradient):** Compute the spatial Jacobian magnitude of the interpolated semantic field via central differences:
       $$G(x) = \|\nabla v(x)\|_F$$
     * **$S(x)$ (Evidential Support):** Compute inverse Simpson concentration $N_{\text{eff}}(x) = \frac{1}{\sum_i \tilde{w}_i(x)^2}$ and distance to the 5th nearest paper $d_5(x)$.

### Phase 5: Exact Pareto Dominance Search (`src/pareto.py`)
* **Objective:** Extract non-dominated points representing distinct trade-off archetypes.
* **Tasks for Pi Agent:**
  1. For each point $x_a$ in the 441-point grid, test against all $x_b$:
     $$x_a \succ x_b \iff (\forall j: F_j(x_a) \ge F_j(x_b)) \land (\exists j: F_j(x_a) > F_j(x_b))$$
     where $F = [P_{\text{boundary}}, W, G, S]$.
  2. Filter out mathematical edge artifacts by enforcing the practical Pareto constraint: $W(x) > 0$.
  3. Rank practical Pareto solutions by balanced compromise distance to the ideal point $(1, 1, 1, 1)$.

### Phase 6: Reporting & Evidence Verbalization (`src/report.py`)
* **Objective:** Produce interactive visual maps and an auditable hypothesis prompt payload.
* **Tasks for Pi Agent:**
  1. **Plotly HTML (`output/radar_map.html`):**
     * Plot existing papers as colored circles (color = regime label).
     * Plot practical Pareto gap points as glowing star markers.
     * Attach hover tooltips: paper titles, authors, gap scores ($P, W, G, S$).
  2. **Unified Markdown Report (`output/RADAR_REPORT.md`):**
     * Summarize corpus breakdown and non-dominated discovery points.
     * Profile the Top 3 candidate gaps: list the 5 closest contributing anchor papers.
     * Run local TF-IDF on the anchor subset to extract shared regional vocabulary ($df \ge 4$) vs. speculative terms ($df \le 3$).
     * Append the **verbatim KMO Appendix A prompt template** pre-filled with the extracted evidence package.

---

## 5. Direct Prompts to Drive the Pi Agent

To initiate the coding steps, feed these direct prompts to your Pi agent:

### Prompt for Step 1 (Setup & Harvester)
> *"Pi, initialize the research-radar repository following PLAN.md Section 4, Phase 1 and Phase 2. Create `requirements.txt`, install dependencies, write `config.yaml`, and implement `src/harvest.py` to pull 40 papers each for SysID, Domain Randomization, and RL Control using the `arxiv` library. Save the parsed corpus to `data/corpus.json`."*

### Prompt for Step 2 (Ollama Embedder)
> *"Pi, implement `src/embed.py` according to Phase 3. Connect to local Ollama on port 11434 and embed the `title + abstract` pairs from `data/corpus.json` using model `f2llm-4b`. Save the normalized embeddings to `data/embeddings.npy` with progress logging and an MD5 cache."*

### Prompt for Step 3 (Manifold, Pareto & Reporting)
> *"Pi, implement `src/manifold.py`, `src/pareto.py`, and `src/report.py` following Phases 4, 5, and 6. Project embeddings to 2D using MDS, compute the 4 KMO discovery fields across a 21x21 grid, extract the practical Pareto front ($W > 0$), and generate `output/radar_map.html` (Plotly) along with `output/RADAR_REPORT.md` with pre-filled LLM evidence packages."*