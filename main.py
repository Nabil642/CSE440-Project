# ============================================================================
# PART 1 - Contributor: Zinat Shaharin Mim
# ============================================================================
# main.py
# ============================================================================
#  BAYES WATCH: THE CASE OF THE VANISHING SAPPHIRE
#  A Bayesian Network mystery solver — Streamlit front end.
# ============================================================================
# This file is ONLY concerned with the interface: laying out the page,
# collecting the player's clues, and displaying results. All of the actual
# Bayesian Network (structure + probability tables) lives in
# supports/mystery_solver.py, which this file imports and calls.

import streamlit as st
import pandas as pd
from pgmpy.inference import VariableElimination

# --- Import the game engine (structure + CPDs live in supports/) -----------
try:
    from supports.mystery_solver import (
        build_bayesian_network,
        create_graphviz_plot,
        cpd_to_dataframe,
    )
except ModuleNotFoundError:
    st.error(
        "ERROR: Could not find the 'supports/mystery_solver.py' module. "
        "Please ensure the file structure is correct (main.py sits next to a "
        "'supports/' folder containing an empty '__init__.py')."
    )
    st.stop()
except ImportError as e:
    st.error(f"ERROR importing from 'supports.mystery_solver': {e}. Check dependencies within mystery_solver.py.")
    st.stop()

# Graphviz is optional — the app should still run without it, just without
# the network diagram.
try:
    import graphviz
except ImportError:
    st.warning(
        "Graphviz library not found. Visualization disabled. To enable, install it "
        "(`pip install graphviz`) and the Graphviz system tools "
        "(see https://graphviz.org/download/)."
    )
    graphviz = None


# ============================================================================
# THEME / FLAVOR CONSTANTS
# ============================================================================
# Centralizing the story text and suspect names here means the whole case
# could be re-themed again later just by editing this one block — the
# Bayesian logic in supports/mystery_solver.py never has to change.

APP_TITLE = "💎 Bayes Watch: The Case of the Vanishing Sapphire"

# Maps the internal suspect codes (used throughout the Bayesian network) to
# the flavorful display names shown in the UI.
SUSPECT_NAMES = {
    'A': 'Lady Odalys Voss — The Heiress',
    'B': 'Jenkins — The Butler',
    'C': 'The Velvet Fox — The Jewel Thief',
}
SUSPECT_EMOJI = {'A': '👑', 'B': '🎩', 'C': '🦊'}
SUSPECT_COLOR = {'A': '#D4AF37', 'B': '#9CA3AF', 'C': '#7C3AED'}  # gold / silver / velvet-purple


# ============================================================================
# PAGE CONFIG + HEADER
# ============================================================================
st.set_page_config(page_title="Bayes Watch: The Vanishing Sapphire", page_icon="💎", layout="wide")

st.title(APP_TITLE)
st.markdown("Using Bayesian Networks to catch whoever stole **The Star of Midnight**.")
st.markdown("---")


# ============================================================================
# PART 2 - Contributor: Md. Nazibul Islam Nabil
# ============================================================================
# ============================================================================
# BUILD MODEL AND INFERENCE ENGINE
# ============================================================================
# model      -> the DiscreteBayesianNetwork (structure + probability tables)
# inference  -> pgmpy's VariableElimination engine, which will do the actual
#               Bayesian math when we ask it a question below.
model = None
inference = None
model_built = False
graph_viz_object = None

try:
    model = build_bayesian_network()
    inference = VariableElimination(model)
    model_built = True
    if graphviz and model:
        graph_viz_object = create_graphviz_plot(model)

except ValueError as e:
    st.error(f"Error building the Bayesian Network: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred during model setup: {e}")


# ============================================================================
# DISPLAY NETWORK STRUCTURE
# ============================================================================
if model_built and graph_viz_object:
    st.subheader("🧭 The Network Behind the Investigation")
    st.markdown("""
    This graph is the entire "case file" the engine reasons with. Every arrow means
    *"the true culprit's identity statistically shapes this clue."*
    - An arrow from **Who Did It? → a clue box** means the guilty party's identity directly
      influences how that clue tends to turn out (e.g. a professional thief is more likely to
      force the vault than an insider with a key).
    - There are **no arrows between the clue boxes themselves** — the model assumes clues are
      only related to each other *through* who the culprit is. This "one hidden cause, many
      visible effects" shape is exactly the structure used in a **Naive Bayes classifier**,
      just applied to detective work instead of spam filtering.
    - A consequence of that shape: `ForcedEntry` is conditionally independent of `AlibiA`
      *given* `GuiltyParty` — once you know who did it, knowing whether the vault was forced
      tells you nothing extra about the Heiress's alibi.
    """)
    try:
        st.graphviz_chart(graph_viz_object)
    except Exception as e:
        st.error(f"Failed to render Graphviz chart: {e}")

    # Educational reveal: let the player peek at the actual probability
    # tables driving the graph above, instead of leaving the math a black box.
    with st.expander("🧮 Peek at the Math — Conditional Probability Tables"):
        st.markdown(
            "Each arrow above is backed by a small table like these. A column tells you "
            "*\"if this suspect were guilty, how likely is this outcome?\"* — the "
            "**opposite** direction from what we actually want. Entering evidence below "
            "and clicking **Solve** is what lets the engine flip these around into "
            "*\"given this evidence, how likely is each suspect?\"* via Bayes' theorem."
        )
        for cpd in model.get_cpds():
            parents = cpd.get_evidence()
            heading = f"P({cpd.variable} | {', '.join(parents)})" if parents else f"P({cpd.variable})  — the prior"
            st.markdown(f"**{heading}**")
            st.dataframe(cpd_to_dataframe(cpd), width='stretch')

    st.markdown("---")
elif model_built and not graphviz:
    st.markdown("---")


# ============================================================================
# PART 3 - Contributor: Nazifa Tahsin
# ============================================================================
# ============================================================================
# SIDEBAR — CASE FILE + EVIDENCE INPUT
# ============================================================================
st.sidebar.header("📜 The Case File")
st.sidebar.markdown(f"""
On the night of the Ravenswood Masquerade Gala, the legendary **Star of Midnight** sapphire
vanished from the manor's private vault. The doors were locked, the guests were dancing —
and by midnight, the jewel was gone.

**Suspects:**
- {SUSPECT_EMOJI['A']} **{SUSPECT_NAMES['A']}** — A distant cousin locked in a decades-long
  feud over who truly inherits the Voss jewels. Had vault access as guest of honor.
- {SUSPECT_EMOJI['B']} **{SUSPECT_NAMES['B']}** — Thirty years of loyal service, and the only
  staff member holding a master key to the vault. Claims he heard nothing all night.
- {SUSPECT_EMOJI['C']} **{SUSPECT_NAMES['C']}** — A never-caught professional, rumored to be
  working the region's high-society events this season. *(No formal "motive" clue exists for
  her — for a professional, the jewels themselves are motive enough.)*
""")

st.sidebar.markdown("---")
st.sidebar.header("🔍 Enter Clues (Evidence)")

# --- Widget default/reset plumbing ------------------------------------------
# Every evidence widget below is given an explicit `key=`, which is what
# lets this Reset button snap them all back to "Unknown" on demand.
EVIDENCE_WIDGET_KEYS = ["fe", "ma", "mb", "aa", "ab", "ac", "fp", "sf"]

# Seed each widget's session-state entry exactly once. Streamlit warns if a
# widget is given BOTH a fixed `index=` default AND has its value written via
# st.session_state (which the Reset button below does) — so instead we let
# session_state be the single source of truth and skip `index=` entirely on
# every selectbox further down.
for _key in EVIDENCE_WIDGET_KEYS:
    if _key not in st.session_state:
        st.session_state[_key] = "Unknown"


def reset_clues():
    """Callback: reset every evidence widget back to its 'Unknown' default."""
    for key in EVIDENCE_WIDGET_KEYS:
        st.session_state[key] = "Unknown"


st.sidebar.button("🔄 Reset All Clues", on_click=reset_clues, width='stretch')

# Generic Yes/No/Unknown evidence options. Selecting "Unknown" maps to
# Python None, which later gets filtered OUT of the evidence dictionary —
# i.e. "Unknown" means "don't tell the engine anything about this clue."
evidence_options = {'Yes': 'Yes', 'No': 'No', 'Unknown': None}

# Fingerprints has 4 possible outcomes instead of 2, so it gets its own map.
fp_options = {
    'None Found': 'None',
    f"Match {SUSPECT_NAMES['A']}": 'A',
    f"Match {SUSPECT_NAMES['B']}": 'B',
    f"Match {SUSPECT_NAMES['C']}": 'C',
    'Unknown': None,
}

st.sidebar.subheader("Scene of the Crime")
fe_input_display = st.sidebar.selectbox("1. Was the vault forced open?",
                                         options=list(evidence_options.keys()), key="fe")
fe_input = evidence_options[fe_input_display]

st.sidebar.subheader("Motive")
m_a_input_display = st.sidebar.selectbox(f"2. Strong motive for {SUSPECT_NAMES['A']}?",
                                          options=list(evidence_options.keys()), key="ma")
m_a_input = evidence_options[m_a_input_display]

m_b_input_display = st.sidebar.selectbox(f"3. Strong motive for {SUSPECT_NAMES['B']}?",
                                          options=list(evidence_options.keys()), key="mb")
m_b_input = evidence_options[m_b_input_display]

st.sidebar.subheader("Alibis")
a_a_input_display = st.sidebar.selectbox(f"4. Alibi for {SUSPECT_NAMES['A']}?",
                                          options=list(evidence_options.keys()), key="aa")
a_a_input = evidence_options[a_a_input_display]

a_b_input_display = st.sidebar.selectbox(f"5. Alibi for {SUSPECT_NAMES['B']}?",
                                          options=list(evidence_options.keys()), key="ab")
a_b_input = evidence_options[a_b_input_display]

a_c_input_display = st.sidebar.selectbox(f"6. Alibi for {SUSPECT_NAMES['C']}?",
                                          options=list(evidence_options.keys()), key="ac")
a_c_input = evidence_options[a_c_input_display]

st.sidebar.subheader("Forensics")
fp_input_display = st.sidebar.selectbox("7. Fingerprints found?",
                                         options=list(fp_options.keys()), key="fp")
fp_input = fp_options[fp_input_display]

sf_input_display = st.sidebar.selectbox("8. Useful security footage?",
                                         options=list(evidence_options.keys()), key="sf")
sf_input = evidence_options[sf_input_display]


# ============================================================================
# PART 4 - Contributor: Tasnif Gaffar Pronoy
# ============================================================================
# ============================================================================
# INFERENCE
# ============================================================================
st.header("🕵️ Inference Results")

if model_built:
    solve_button = st.button("Solve the Mystery", type="primary")

    if solve_button:
        # Build the evidence dictionary, filtering out anything left as
        # "Unknown" (None) — pgmpy should only see clues we actually have.
        evidence_dict = {}
        if fe_input is not None: evidence_dict['ForcedEntry'] = fe_input
        if m_a_input is not None: evidence_dict['MotiveA'] = m_a_input
        if m_b_input is not None: evidence_dict['MotiveB'] = m_b_input
        if a_a_input is not None: evidence_dict['AlibiA'] = a_a_input
        if a_b_input is not None: evidence_dict['AlibiB'] = a_b_input
        if a_c_input is not None: evidence_dict['AlibiC'] = a_c_input
        if fp_input is not None: evidence_dict['Fingerprints'] = fp_input
        if sf_input is not None: evidence_dict['SecurityFootage'] = sf_input

        st.subheader("Evidence Considered")
        if not evidence_dict:
            st.write("No specific clues entered. Showing prior probabilities.")
        else:
            st.json(evidence_dict)

        try:
            if inference is None:
                st.error("Inference engine not initialized. Cannot solve.")
            else:
                # -----------------------------------------------------------
                # THE CORE BAYESIAN STEP
                # -----------------------------------------------------------
                # inference.query(...) asks pgmpy for the POSTERIOR
                # distribution P(GuiltyParty | evidence_dict). Under the
                # hood, VariableElimination:
                #   1. Starts from the joint distribution implied by every
                #      CPD multiplied together (the chain rule of
                #      probability applied along the graph's edges).
                #   2. "Clamps" each evidence variable to its observed value.
                #   3. Sums out (marginalizes) every variable that ISN'T
                #      GuiltyParty, working through the network in the most
                #      efficient order it can find — instead of building
                #      the full joint table (which would be much larger),
                #      it exploits the conditional independencies baked
                #      into the graph to eliminate variables locally.
                # This is exactly Bayes' theorem —
                #     P(GuiltyParty | Evidence) ∝ P(Evidence | GuiltyParty) · P(GuiltyParty)
                # — computed exactly and efficiently rather than by brute force.
                posterior_gp = inference.query(variables=['GuiltyParty'], evidence=evidence_dict)

                st.subheader("Posterior Probability of Guilt")

                prob_df = pd.DataFrame({
                    'Code': posterior_gp.state_names['GuiltyParty'],
                    'Probability': posterior_gp.values,
                })
                prob_df['Suspect'] = prob_df['Code'].map(SUSPECT_NAMES)
                prob_df = prob_df.sort_values(by='Probability', ascending=False).reset_index(drop=True)
                prob_df['Probability_pct'] = prob_df['Probability'].map('{:.2%}'.format)

                st.dataframe(prob_df[['Suspect', 'Probability_pct']], width='stretch', hide_index=True)

                st.subheader("Probability Distribution")
                chart_data = pd.DataFrame(
                    prob_df['Probability'].values,
                    index=prob_df['Suspect'].values,
                    columns=['Probability']
                )
                st.bar_chart(chart_data)

                # -------------------------------------------------------
                # VERDICT BANNER — a fun, readable wrap-up of the numbers
                # -------------------------------------------------------
                top = prob_df.iloc[0]
                second = prob_df.iloc[1]
                margin = top['Probability'] - second['Probability']

                if top['Probability'] >= 0.60:
                    st.success(
                        f"🚨 **Case Closed!** The evidence overwhelmingly points to "
                        f"**{top['Suspect']}** ({top['Probability_pct']} likely guilty)."
                    )
                    st.balloons()
                elif margin < 0.08:
                    st.info(
                        "🤔 **Too close to call.** The leading suspects are nearly tied — "
                        "gather more clues before making an accusation."
                    )
                else:
                    st.warning(
                        f"🕵️ **Leading suspect: {top['Suspect']}** ({top['Probability_pct']}), "
                        f"but the case isn't airtight yet."
                    )

                st.markdown("---")
                st.markdown("**Interpretation Notes:**")
                st.markdown(
                    "*   Motive, alibis, forensics, and the footage clue are all formally wired "
                    "into the network — every clue you enter genuinely reshapes the posterior above."
                )
                st.markdown(
                    "*   Probabilities reflect the model's belief based *only* on the network's "
                    "structure, its probability tables, and the evidence you entered — they are a "
                    "reasoning aid, not a verdict."
                )

        except ValueError as e:
            st.error(f"An error occurred during inference: {e}")
            st.warning("This can sometimes happen if evidence conflicts strongly. Try removing some evidence.")
        except Exception as e:
            st.error(f"An unexpected error occurred during inference: {e}")
            st.error("Details: " + str(e))

# ============================================================================
# PART 5 - Contributor: Jannat-a-habib-baishakhi
# ============================================================================
    else:
        # No clues submitted yet this run — show the prior distribution so
        # the page never feels empty.
        st.info("Enter clues in the sidebar and click **Solve the Mystery**.")
        try:
            if inference is None:
                st.error("Inference engine not initialized. Cannot show priors.")
            else:
                prior_gp = inference.query(variables=['GuiltyParty'])  # No evidence => the raw priors
                st.subheader("Initial (Prior) Probability of Guilt")

                prior_df = pd.DataFrame({
                    'Code': prior_gp.state_names['GuiltyParty'],
                    'Probability': prior_gp.values,
                })
                prior_df['Suspect'] = prior_df['Code'].map(SUSPECT_NAMES)
                display_df = prior_df[['Suspect', 'Probability']].copy()
                display_df['Probability'] = display_df['Probability'].map('{:.2%}'.format)
                st.dataframe(display_df, width='stretch', hide_index=True)

                chart_data = pd.DataFrame(
                    prior_df['Probability'].values,
                    index=prior_df['Suspect'].values,
                    columns=['Probability']
                )
                st.bar_chart(chart_data)
        except Exception as e:
            st.error(f"An error occurred while calculating prior probabilities: {e}")

else:
    st.warning("The Bayesian Network model could not be built. Please check console/logs for errors. Inference is disabled.")


st.markdown("---")
st.markdown("Built using `pgmpy`, `streamlit`, and `graphviz`. 💎 *Bayes Watch* — because every clue is a witness, and Bayes is the interrogator.")