# 💎 Bayes Watch: The Case of the Vanishing Sapphire

**A Bayesian Network-based mystery solver, built with `pgmpy` and `streamlit`.**

> *Because every clue is a witness, and Bayes is the interrogator.*

Bayes Watch is an interactive detective game that uses a real probabilistic
inference engine — not scripted logic — to decide "whodunit." Enter clues
about the crime, and watch the model update its belief about each suspect's
guilt live, using exact Bayesian inference.

---

## 📖 The Case File

On the night of the Ravenswood Masquerade Gala, the legendary **Star of
Midnight** sapphire vanished from the manor's private vault. The doors were
locked, the guests were dancing — and by midnight, the jewel was gone.

**Suspects:**

| | Suspect | Profile |
|---|---|---|
| 👑 | **Lady Odalys Voss** — *The Heiress* | A distant cousin locked in a decades-long feud over who truly inherits the Voss jewels. Had vault access as guest of honor. |
| 🎩 | **Jenkins** — *The Butler* | Thirty years of loyal service, and the only staff member holding a master key to the vault. Claims he heard nothing all night. |
| 🦊 | **The Velvet Fox** — *The Jewel Thief* | A never-caught professional, rumored to be working the region's high-society events this season. *(No formal "motive" clue exists for her — for a professional, the jewels themselves are motive enough.)* |

Your job: enter whatever evidence you've gathered, click **Solve the
Mystery**, and let the network tell you who's really guilty.

---

## ✨ Features

- **Real Bayesian inference, not a scripted quiz.** Every posterior
  probability is computed exactly via `pgmpy`'s Variable Elimination
  algorithm — nothing is hard-coded per scenario.
- **Transparent reasoning.** The network structure is rendered live with
  Graphviz, and an expandable panel lets you inspect the raw conditional
  probability tables (CPDs) behind every arrow.
- **8 independent clues.** Forced entry, individual motive, individual
  alibi, fingerprint matches, and security footage — each one reshapes the
  posterior the moment it's entered.
- **Graceful handling of missing evidence.** Every clue defaults to
  `Unknown`, which is correctly excluded from inference (marginalized over)
  rather than guessed at.
- **Instant, visual feedback.** Results are shown as both a probability
  table and a live-updating bar chart, plus a plain-language verdict banner
  ("Case Closed," "Too close to call," etc.).
- **One-click reset.** Snap every clue back to `Unknown` and start the
  investigation over.

---

## 🧠 How It Works

Bayes Watch models the mystery as a **Discrete Bayesian Network** with a
Naive-Bayes-style topology: a single hidden cause (`GuiltyParty`) with a
direct edge to every observable clue.

```
                         ┌───────────────┐
                         │  GuiltyParty  │
                         │   {A, B, C}   │
                         └───────┬───────┘
        ┌───────────┬───────────┼───────────┬───────────┬───────────┐
        ▼           ▼           ▼           ▼           ▼           ▼
  ForcedEntry   MotiveA/B   AlibiA/B/C  Fingerprints  SecurityFootage
```

Because every clue is conditionally independent of every other clue *given*
the guilty party's identity, the model needs only nine small probability
tables instead of one impossibly large joint table — the same structural
trick used by a Naive Bayes spam filter, just aimed at detective work
instead of email.

| Node | Meaning | States |
|---|---|---|
| `GuiltyParty` | Identity of the true culprit | `A`, `B`, `C` |
| `ForcedEntry` | Was the vault forced open? | `Yes`, `No` |
| `MotiveA` | Strong motive for the Heiress? | `Yes`, `No` |
| `MotiveB` | Strong motive for the Butler? | `Yes`, `No` |
| `AlibiA` | Alibi for the Heiress? | `Yes`, `No` |
| `AlibiB` | Alibi for the Butler? | `Yes`, `No` |
| `AlibiC` | Alibi for the Jewel Thief? | `Yes`, `No` |
| `Fingerprints` | Whose prints were recovered? | `None`, `A`, `B`, `C` |
| `SecurityFootage` | Was the footage useful? | `Yes`, `No` |

Given whatever subset of these you observe, the app queries
`P(GuiltyParty | Evidence)` and reports the exact posterior — not an
approximation — because the network is small enough for Variable
Elimination to solve directly.

---

## 🖥️ Screenshots

**The case file, and the network the engine reasons with:**

The sidebar introduces the scenario and suspects, while the main panel
renders the live Bayesian network graph and lets you peek at the
underlying probability tables.

**Entering clues and solving the mystery:**

Selectors for every clue live in the sidebar (grouped under *Scene of the
Crime*, *Motive*, *Alibis*, and *Forensics*), each defaulting to
`Unknown`. Before any evidence is submitted, the main panel shows the
uniform 33.33% / 33.33% / 33.33% prior for the three suspects, updating
instantly to the posterior distribution — table, bar chart, and verdict —
once you click **Solve the Mystery**.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Probabilistic model & inference | [`pgmpy`](https://pgmpy.org/) (`DiscreteBayesianNetwork`, `TabularCPD`, `VariableElimination`) |
| Web interface | [`streamlit`](https://streamlit.io/) |
| Network visualization | [`graphviz`](https://graphviz.org/) *(optional — app runs without it, just without the diagram)* |
| Data handling / display | `pandas` |

---

## 📂 Project Structure

```
bayes-watch/
├── main.py                     # Streamlit front end (UI, widgets, results display)
├── supports/
│   ├── __init__.py
│   └── mystery_solver.py       # Bayesian network structure, CPDs, and helper functions
└── requirements.txt
```

`main.py` handles nothing but layout and interaction — it imports
`build_bayesian_network()`, `create_graphviz_plot()`, and
`cpd_to_dataframe()` from `supports/mystery_solver.py`, which owns all of
the actual probabilistic model. This split means the entire case (suspect
names, clue wording, even the CPD values) can be re-themed without
touching the reasoning engine.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- (Optional, for the network diagram) the [Graphviz system
  package](https://graphviz.org/download/) installed and on your `PATH`,
  in addition to the `graphviz` Python bindings.

### Installation

```bash
git clone <repository-url>
cd bayes-watch
pip install -r requirements.txt
```

**`requirements.txt`**
```
streamlit
pgmpy
pandas
graphviz
```

### Run the app

```bash
streamlit run main.py
```

Then open the local URL Streamlit prints in your browser (typically
`http://localhost:8501`).

---

## 🕵️ Usage

1. Read the case file in the sidebar to meet the three suspects.
2. Expand **"Peek at the Math"** if you want to see the raw conditional
   probability tables before diving in.
3. Select whatever clues you've gathered — leave anything you don't know
   as `Unknown`.
4. Click **Solve the Mystery**.
5. Read the updated posterior probabilities, the bar chart, and the
   verdict banner.
6. Click **Reset All Clues** to start a fresh investigation.

---

## 👥 Team — CSE440, Section 1, Group 8

| Name | ID | Contribution |
|---|---|---|
| Md. Nazibul Islam Nabil | 2222456042 | Model construction & inference-engine bootstrap, Graphviz rendering, CPD viewer |
| Tasnif Gaffar Pronoy | 2222590042 | Query execution, results table, bar chart, verdict-banner logic |
| Nazifa Tahsin | 2132652642 | Sidebar case file & evidence-input widgets, reset logic |
| Zinat Shaharin Mim | 2111301042 | App header, page configuration, theming constants |
| Jannat-a-habib-baishakhi | 2131069642 | Default prior-probability view |

All members jointly contributed to the Bayesian network design (structure
and CPDs) and end-to-end testing.

---

## ⚠️ Limitations

- Uses a simplified **naive-Bayes** structure — all clues are assumed
  conditionally independent given the guilty party, even though some
  real-world clues (e.g., forced entry and fingerprints) could plausibly
  be correlated.
- All probabilities were **hand-authored from narrative logic**, not
  learned from data, so results reflect the model's internal assumptions
  rather than objective fact.
- Scoped to a **single fixed scenario** with three suspects and eight
  evidence types.

## 🔭 Future Work

- Support dependencies between evidence nodes (beyond naive Bayes).
- Learn CPDs from data with sensitivity analysis, if example outcomes
  become available.
- Add approximate inference (e.g., MCMC) for larger, denser future cases.
- Let users author entirely new mystery scenarios via a config file
  instead of code changes.

---

## 📚 References

- pgmpy: Probabilistic Graphical Models using Python — https://pgmpy.org/
- Streamlit Documentation — https://streamlit.io/
- Koller & Friedman, *Probabilistic Graphical Models: Principles and
  Techniques*, MIT Press, 2009.

---

*Built with `pgmpy`, `streamlit`, and `graphviz` as a course project for
CSE440 (Artificial Intelligence), North South University.*
