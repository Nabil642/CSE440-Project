# supports/mystery_solver.py
#
# ============================================================================
#  BAYES WATCH: THE VANISHING SAPPHIRE  —  ENGINE ROOM
# ============================================================================
# This file is the "brain" of the game. It never touches the UI — its only
# job is to (1) define the Bayesian Network that models the heist, and
# (2) turn that network into a picture. main.py handles everything visual.
#
# ---------------------------------------------------------------------------
# THE BIG PICTURE — what kind of Bayesian Network is this?
# ---------------------------------------------------------------------------
# Every node below is either:
#   • GuiltyParty   — the single hidden CAUSE we're trying to infer (whodunit)
#   • everything else — an observable EFFECT / clue that guilt produces
#
# GuiltyParty is the *only* parent of every other node, and none of the
# clue-nodes point at each other. That star-shaped structure is exactly the
# graph used by a Naive Bayes classifier: one class variable, several
# features assumed conditionally independent of each other *given* the
# class. Here "GuiltyParty" is the class, and "did the vault look forced?",
# "does the Butler have an alibi?", etc. are the features.
#
# That conditional-independence assumption is what makes the whole game
# playable in real time: instead of needing one giant probability table
# over every combination of every clue (which would be enormous), we only
# need one small table per clue, each conditioned on GuiltyParty alone.
# pgmpy's VariableElimination then uses the graph structure to combine
# those small tables efficiently when we ask for a posterior — see the
# comments in main.py where inference.query(...) is called for more.
# ---------------------------------------------------------------------------

import pandas as pd
import graphviz
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


# --- Bayesian Network Definition --------------------------------------------

def build_bayesian_network():
    """
    Defines the structure and CPDs of the Bayesian Network for
    "The Case of the Vanishing Sapphire".

    Returns:
        pgmpy.models.DiscreteBayesianNetwork: The configured Bayesian Network model.
    Raises:
        ValueError: If the model structure or CPDs are invalid.
    """

    # ------------------------------------------------------------------
    # 1) STRUCTURE — who points at whom
    # ------------------------------------------------------------------
    # Each tuple is (Parent, Child), i.e. "Parent causally/statistically
    # influences Child". GuiltyParty is the parent of every clue node.
    #
    # NOTE ON THE UPGRADE FROM THE ORIGINAL VERSION:
    # The original manuscript-themed app collected "Motive" answers in the
    # sidebar but never fed them into the network (they were flavor text
    # only). Here MotiveA and MotiveB are wired in as real evidence nodes,
    # so entering a motive clue actually moves the probabilities — the
    # game logic is the same *pattern* (GuiltyParty → clue), just applied
    # to two more clues than before.
    model = DiscreteBayesianNetwork([
        ('GuiltyParty', 'ForcedEntry'),       # Was the vault mechanically forced?
        ('GuiltyParty', 'MotiveA'),           # Does the Heiress have a strong motive?
        ('GuiltyParty', 'MotiveB'),           # Does the Butler have a strong motive?
        ('GuiltyParty', 'AlibiA'),            # Does the Heiress have an alibi?
        ('GuiltyParty', 'AlibiB'),            # Does the Butler have an alibi?
        ('GuiltyParty', 'AlibiC'),            # Does the Velvet Fox have an alibi?
        ('GuiltyParty', 'Fingerprints'),      # Whose prints were on the vault?
        ('GuiltyParty', 'SecurityFootage'),   # Is there usable security footage?
    ])

    # ------------------------------------------------------------------
    # 2) CONDITIONAL PROBABILITY DISTRIBUTIONS (CPDs) — the actual "math"
    # ------------------------------------------------------------------
    # Every CPD below encodes a little piece of detective intuition as a
    # number. Reading a column tells you "IF this suspect were the thief,
    # HOW LIKELY is this clue to look this way?" That's P(Clue | Guilty),
    # which is the opposite direction from what we actually want
    # (P(Guilty | Clue)). Bayes' theorem — via pgmpy's inference engine —
    # is what flips these around once you enter evidence. See main.py.

    # --- P(GuiltyParty) — the prior -------------------------------------
    # Before any clues are entered, all three suspects are equally likely.
    # This is the "innocent until proven otherwise" starting point.
    cpd_gp = TabularCPD(
        variable='GuiltyParty', variable_card=3,
        values=[[1 / 3], [1 / 3], [1 / 3]],
        state_names={'GuiltyParty': ['A', 'B', 'C']}
    )

    # --- P(ForcedEntry | GuiltyParty) -----------------------------------
    # A and B are insiders with legitimate vault access (the Heiress was a
    # guest of honor, the Butler holds the master key) — an insider rarely
    # NEEDS to force the lock. C, the Velvet Fox, is an outside professional
    # with no legitimate access, so forced/bypassed entry is very likely
    # if C is guilty.
    cpd_fe = TabularCPD(
        variable='ForcedEntry', variable_card=2,
        values=[[0.1, 0.1, 0.9],   # ForcedEntry = Yes | GuiltyParty = A, B, C
                [0.9, 0.9, 0.1]],  # ForcedEntry = No  | GuiltyParty = A, B, C
        evidence=['GuiltyParty'], evidence_card=[3],
        state_names={'ForcedEntry': ['Yes', 'No'],
                     'GuiltyParty': ['A', 'B', 'C']}
    )

    # --- P(MotiveA | GuiltyParty)  [NEW NODE] ---------------------------
    # The Heiress has a long-running inheritance feud over the Voss jewels.
    # If she's guilty, that motive is very likely to surface under
    # questioning (0.75). If someone else is guilty, she might still look
    # a little suspicious to investigators (family drama rarely evaporates
    # just because she's innocent), but noticeably less so.
    cpd_ma = TabularCPD(
        variable='MotiveA', variable_card=2,
        values=[[0.75, 0.35, 0.30],   # MotiveA = Yes | GuiltyParty = A, B, C
                [0.25, 0.65, 0.70]],  # MotiveA = No  | GuiltyParty = A, B, C
        evidence=['GuiltyParty'], evidence_card=[3],
        state_names={'MotiveA': ['Yes', 'No'],
                     'GuiltyParty': ['A', 'B', 'C']}
    )

    # --- P(MotiveB | GuiltyParty)  [NEW NODE] ---------------------------
    # The Butler's plausible motive (money troubles, a grudge after being
    # passed over) is strongest if he's actually guilty (0.70), but staff
    # gossip means some baseline suspicion of him exists either way.
    cpd_mb = TabularCPD(
        variable='MotiveB', variable_card=2,
        values=[[0.30, 0.70, 0.25],   # MotiveB = Yes | GuiltyParty = A, B, C
                [0.70, 0.30, 0.75]],  # MotiveB = No  | GuiltyParty = A, B, C
        evidence=['GuiltyParty'], evidence_card=[3],
        state_names={'MotiveB': ['Yes', 'No'],
                     'GuiltyParty': ['A', 'B', 'C']}
    )

    # --- P(AlibiA | GuiltyParty) -----------------------------------------
    # If A is guilty, she probably could NOT produce a solid alibi for the
    # theft window (0.3 chance she has one anyway — alibis can be faked or
    # simply lucky). If someone else is guilty, she's more likely to have
    # a real, checkable alibi (0.8).
    cpd_aa = TabularCPD(
        variable='AlibiA', variable_card=2,
        values=[[0.3, 0.8, 0.8],
                [0.7, 0.2, 0.2]],
        evidence=['GuiltyParty'], evidence_card=[3],
        state_names={'AlibiA': ['Yes', 'No'],
                     'GuiltyParty': ['A', 'B', 'C']}
    )

    # --- P(AlibiB | GuiltyParty) -----------------------------------------
    # Same logic, mirrored for the Butler.
    cpd_ab = TabularCPD(
        variable='AlibiB', variable_card=2,
        values=[[0.8, 0.3, 0.7],
                [0.2, 0.7, 0.3]],
        evidence=['GuiltyParty'], evidence_card=[3],
        state_names={'AlibiB': ['Yes', 'No'],
                     'GuiltyParty': ['A', 'B', 'C']}
    )

    # --- P(AlibiC | GuiltyParty) -----------------------------------------
    # The Velvet Fox is a professional — if guilty, expect a well-prepared
    # cover story to be much rarer than for the two insiders (0.2), since
    # a stranger's movements are easier for investigators to poke holes in.
    cpd_ac = TabularCPD(
        variable='AlibiC', variable_card=2,
        values=[[0.7, 0.7, 0.2],
                [0.3, 0.3, 0.8]],
        evidence=['GuiltyParty'], evidence_card=[3],
        state_names={'AlibiC': ['Yes', 'No'],
                     'GuiltyParty': ['A', 'B', 'C']}
    )

    # --- P(Fingerprints | GuiltyParty) -----------------------------------
    # Whoever did it usually leaves at least a trace of themselves behind,
    # but professionals (C) are more careful (gloves, technique), so "None
    # Found" stays relatively likely even when C is guilty (0.6). When
    # they DO slip up, prints are far more likely to match the true thief
    # than either innocent party.
    cpd_fp = TabularCPD(
        variable='Fingerprints', variable_card=4,
        values=[[0.4, 0.4, 0.6],    # Fingerprints = None
                [0.5, 0.05, 0.1],   # Fingerprints = A
                [0.05, 0.5, 0.1],   # Fingerprints = B
                [0.05, 0.05, 0.2]], # Fingerprints = C
        evidence=['GuiltyParty'], evidence_card=[3],
        state_names={'Fingerprints': ['None', 'A', 'B', 'C'],
                     'GuiltyParty': ['A', 'B', 'C']}
    )

    # --- P(SecurityFootage | GuiltyParty) --------------------------------
    # A guilty insider (A or B) is somewhat more likely to know to dodge or
    # disable the cameras than a random innocent scenario would suggest,
    # and the Velvet Fox — being a professional — is the most camera-shy
    # of all, so useful footage is least likely when C is guilty.
    cpd_sf = TabularCPD(
        variable='SecurityFootage', variable_card=2,
        values=[[0.4, 0.4, 0.2],
                [0.6, 0.6, 0.8]],
        evidence=['GuiltyParty'], evidence_card=[3],
        state_names={'SecurityFootage': ['Yes', 'No'],
                     'GuiltyParty': ['A', 'B', 'C']}
    )

    # ------------------------------------------------------------------
    # 3) ASSEMBLE + VALIDATE
    # ------------------------------------------------------------------
    model.add_cpds(cpd_gp, cpd_fe, cpd_ma, cpd_mb, cpd_aa, cpd_ab, cpd_ac, cpd_fp, cpd_sf)

    # check_model() verifies: every CPD's probabilities sum to 1, every
    # node has exactly one CPD, and the CPDs' variables/evidence match the
    # graph edges we declared above. If we made an arithmetic slip in any
    # table, this catches it immediately instead of failing silently.
    if not model.check_model():
        raise ValueError("Model definition is invalid. Check CPDs and structure.")

    return model


# --- Function to create Graphviz object from pgmpy model --------------------

def create_graphviz_plot(model):
    """
    Creates a graphviz Digraph object representing the BN structure, styled
    to match the "Bayes Watch" midnight-and-gold heist theme.

    Args:
        model (pgmpy.models.DiscreteBayesianNetwork): The Bayesian network model.

    Returns:
        graphviz.Digraph or None: The graphviz object, or None if graphviz is not installed.
    """
    if not graphviz:  # Defensive check, mirrors the import guard in main.py
        print("Warning: Graphviz library not found or not passed correctly.")
        return None

    # Friendlier, multi-line labels for the node boxes in the diagram.
    # (The underlying pgmpy variable names stay short and code-friendly;
    # this dict is purely cosmetic, for the picture the player sees.)
    FRIENDLY_LABELS = {
        'GuiltyParty':     'WHO DID IT?\n(Guilty Party)',
        'ForcedEntry':     'Vault\nForced Open?',
        'MotiveA':         'Motive:\nThe Heiress',
        'MotiveB':         'Motive:\nThe Butler',
        'AlibiA':          'Alibi:\nThe Heiress',
        'AlibiB':          'Alibi:\nThe Butler',
        'AlibiC':          'Alibi:\nThe Velvet Fox',
        'Fingerprints':    'Fingerprints\nFound',
        'SecurityFootage': 'Security\nFootage',
    }

    dot = graphviz.Digraph(
        comment='Bayesian Network Structure - Bayes Watch',
        graph_attr={'rankdir': 'TB', 'bgcolor': 'transparent'}  # Top-to-bottom layout
    )

    # Add nodes — GuiltyParty (the hidden root cause) gets a distinct gold
    # "spotlight" style so it visually reads as the thing every clue-node
    # ultimately points back to.
    for node in model.nodes():
        label = FRIENDLY_LABELS.get(node, node)
        if node == 'GuiltyParty':
            dot.node(
                node, label,
                shape='doublecircle', style='filled',
                fillcolor='#D4AF37', fontcolor='#0B132B',
                fontname='Helvetica-Bold', penwidth='2'
            )
        else:
            dot.node(
                node, label,
                shape='box', style='filled,rounded',
                fillcolor='#1C2541', fontcolor='#F5F5F5',
                color='#D4AF37', fontname='Helvetica'
            )

    # Add edges (all of them fan out from GuiltyParty)
    for parent, child in model.edges():
        dot.edge(parent, child, color='#D4AF37', arrowsize='0.8')

    return dot


# --- Helper: turn a CPD into a readable table for the UI --------------------

def cpd_to_dataframe(cpd):
    """
    Converts a single pgmpy TabularCPD into a tidy pandas DataFrame so the
    app can display "the math behind the magic" to the player.

    Rows    = the variable's own possible outcomes.
    Columns = the parent's (GuiltyParty's) possible outcomes — or a single
              'Prior' column for the one CPD that has no parent.

    This is purely a display helper; it doesn't affect inference at all.
    """
    values = cpd.get_values()  # shape: (variable_card, product_of_evidence_cards)
    evidence_vars = cpd.get_evidence()

    if evidence_vars:
        # Every CPD in this model has exactly one parent (GuiltyParty),
        # so we only need to handle the single-evidence-variable case.
        parent = evidence_vars[0]
        columns = cpd.state_names[parent]
    else:
        columns = ['Prior']

    return pd.DataFrame(values, index=cpd.state_names[cpd.variable], columns=columns)


# --- Quick standalone sanity check ------------------------------------------
# Run this file directly (`python supports/mystery_solver.py`) to sanity-check
# the network outside of Streamlit and print every CPD to the terminal —
# handy while tweaking the probability tables.
if __name__ == "__main__":
    m = build_bayesian_network()
    print(f"Model built OK. Nodes: {list(m.nodes())}")
    for cpd in m.get_cpds():
        print("\n" + "=" * 60)
        print(f"P({cpd.variable} | {', '.join(cpd.get_evidence()) or '—'})")
        print(cpd_to_dataframe(cpd))
