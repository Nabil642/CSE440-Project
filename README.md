# 💎 Bayes Watch: The Case of the Vanishing Sapphire

**A Bayesian Network mystery solver, built with `pgmpy` + `streamlit`.**

Enter clues about a jewel heist. Watch a Bayesian Network turn them into a live-updating probability of guilt for each suspect — no black box, every table is on display.

> Built as an AI coursework project exploring probabilistic graphical models, and adapted from the [ClueChain](https://github.com/nipunsaif/ClueChain) mystery-solver concept.

---

## Overview

On the night of the Ravenswood Masquerade Gala, the legendary **Star of Midnight** sapphire vanished from the manor's private vault. The doors were locked, the guests were dancing, and by midnight the jewel was gone.

**Bayes Watch** casts you as the detective. You feed it clues — was the vault forced open? does a suspect have an alibi? whose fingerprints turned up? — and a Bayesian Network computes, in real time, the posterior probability that each suspect is the culprit. It's the same underlying idea as a spam filter or a medical diagnosis tool, just pointed at a locked-room mystery instead.

## 🕵️ The Case

| Suspect | Role | Why they're a suspect |
|---|---|---|
| 👑 **Lady Odalys Voss** | The Heiress | A distant cousin locked in a decades-long inheritance feud; had vault access as guest of honor |
| 🎩 **Jenkins** | The Butler | Thirty years of service, holds the only master key, claims he heard nothing |
| 🦊 **The Velvet Fox** | The Jewel Thief | A never-caught professional, rumored to be working the region's high-society events this season |

## ✨ Features

- **Bayesian Inference** — every clue you enter is run through exact inference (`VariableElimination`) to compute `P(GuiltyParty | Evidence)`, not a heuristic score.
- **8 wired-in clues** — forced entry, motive (for the two insiders), alibis for all three suspects, fingerprint matches, and security footage all feed the network.
- **Live network diagram** — a Graphviz view of the actual graph structure, styled in midnight-navy and gold, with the "who did it?" node visually spotlighted as the hidden root cause.
- **"Peek at the Math" panel** — an expander that prints every raw Conditional Probability Table (CPD) so the reasoning is never a black box.
- **Verdict banner** — a plain-language read of the numbers ("Case Closed!" past 60% confidence, or "too close to call" when the top two suspects are nearly tied).
- **One-click reset** — clear every clue back to "Unknown" without reloading the page.

## 🧠 How It Works, Briefly

The network has one hidden variable, `GuiltyParty` (A / B / C), and eight observable clue variables that all branch directly off it — nothing connects clue-to-clue. That "one cause, many effects" shape is the same graph a **Naive Bayes classifier** uses; here the "class" is the culprit and the "features" are the clues.

Each clue's Conditional Probability Table encodes a piece of detective logic — e.g. a professional thief is far more likely to force the vault than an insider with a key. Bayes' theorem is what turns those tables around:

```
P(GuiltyParty | Evidence)  ∝  P(Evidence | GuiltyParty) · P(GuiltyParty)
```

`pgmpy`'s `VariableElimination` computes this exactly, using the graph's conditional-independence structure to avoid building the full joint probability table by hand.

## 🚀 Usage

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/bayes-watch.git
cd bayes-watch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run main.py
```

Streamlit will open the app in your browser (default: `http://localhost:8501`). Enter clues in the sidebar and click **Solve the Mystery**.

## 📁 Project Structure

```
bayes-watch/
│
├── main.py                    # Streamlit UI: layout, sidebar inputs, results display
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── .streamlit/
│   └── config.toml            # App theme (midnight navy + gold)
│
└── supports/
    ├── __init__.py            # Marks 'supports' as a Python package
    └── mystery_solver.py      # The Bayesian Network: structure, CPDs, and graph rendering
```

## 🛠 Requirements

Core dependencies actually used by the app:

- **`streamlit`** — web application interface
- **`pgmpy`** — Bayesian Network construction and exact inference
- **`pandas`** — formatting probability tables for display
- **`graphviz`** *(optional)* — renders the network diagram; the app still runs without it, just without the picture. Also requires the [Graphviz system binaries](https://graphviz.org/download/) to be installed separately from the Python package.

`requirements.txt` also carries a few extras (`numpy`, `networkx`, `matplotlib`, `ipykernel`, `jupyter`) inherited from the original project scaffolding — handy if you want to prototype the network in a notebook, but not required to run `main.py` itself.

Python 3.9+ recommended.

## 🔭 Possible Extensions

Ideas for taking this further:

- Add a `MotiveC` node or other clues specific to the Velvet Fox
- Let users pick between multiple mystery scenarios (a `data/` folder of alternate cases already exists as a starting point)
- Add a "sensitivity analysis" view showing which clue would most shift the verdict if answered
- Persist a case log so past investigations can be reviewed

## 🤝 Contributing

1. Fork this repository
2. Create a branch for your feature or fix
3. Commit your changes
4. Push to your fork
5. Open a pull request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
