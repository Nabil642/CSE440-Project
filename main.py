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
