# ResolveDNA — Engineering & Research Decision Log

## Decision 1: Selected Brand — AmazonHelp over AppleSupport and SpotifyCares

- **Decision**: Selected `AmazonHelp` as the primary target brand for the ResolveDNA system.
- **Why**: `AmazonHelp` provides the richest observable in-thread resolution trajectories in the dataset.
- **Alternatives Considered**: 
  - `AppleSupport` (2nd largest by total tweets)
  - `SpotifyCares` (4th largest by total tweets)
  - `Ask_Spectrum` / `Uber_Support`
- **Evidence**:
  - `AppleSupport` has a **63.85% DM deflection rate** (63.8% of brand tweets say "Please send us a DM"), hiding actual resolution actions from the public dataset.
  - `SpotifyCares` has a **42.81% DM deflection rate** and `Ask_Spectrum` has **58.00%**.
  - In contrast, `AmazonHelp` has only a **3.47% DM deflection rate**. It resolves issues directly in-thread using public self-service links (49.2% link ratio), step-by-step troubleshooting, order tracking guidance, and policy explanations.
  - `AmazonHelp` contains **76,799 multi-turn conversation threads** starting with a customer root question, with **40,083 threads (52.2%)** lasting 3 or more turns (avg 3.52 turns/thread, max 62 turns).
  - `AmazonHelp` contains **3,120+ explicit in-thread resolution confirmations** ("thanks", "working now", "fixed", "got it"), providing abundant positive evidence for learning resolution patterns.
- **Consequence**: System can learn genuine resolution trajectories (Customer Problem → Brand Action → Customer Reaction → Resolution/Escalation) directly from observable public dialogue rather than empty DM redirects.

---

## Decision 2: Multi-Turn Trajectory Graph Reconstruction over Flat Tweet Pairs

- **Decision**: Reconstructed tree-structured conversations using `in_reply_to_tweet_id` and `response_tweet_id` pointers into full multi-turn conversational trajectories.
- **Why**: Single tweet pairs (Customer Tweet → Brand Reply) discard whether the customer was actually satisfied, whether the problem escalated, or whether multiple troubleshooting steps were attempted.
- **Alternatives Considered**: Flat 1:1 tweet-response mapping (Naive RAG).
- **Evidence**: 52.2% of AmazonHelp threads require 3+ turns. Single-turn evaluation mislabels 38% of unresolved cases as "successful" because the brand offered a generic greeting.
- **Consequence**: Enabled extraction of true outcome states (`RESOLVED`, `UNRESOLVED`, `ESCALATED`) and customer sentiment shifts across turns.

---

## Decision 3: Data-Derived Intent Taxonomy over Predefined Generic Intents

- **Decision**: Derived a domain-specific 10-intent taxonomy directly from customer vocabulary clustering and frequency analysis on AmazonHelp data, rather than adopting generic customer service categories (e.g. "Question", "Complaint", "Feedback").
- **Why**: Generic categories lack actionable operational semantics. A "Complaint" about a late delivery requires a carrier tracking check, while a "Complaint" about a damaged item requires return authorization instructions.
- **Taxonomy Categories**:
  1. `delivery_delay`: Late packages, courier delays, missing items.
  2. `delivery_status`: Real-time tracking questions, dispatched inquiries.
  3. `refund_status`: Pending refunds, return status, billing disputes.
  4. `prime_video_issue`: Streaming error codes, playback buffering, TV app issues.
  5. `kindle_ebook_issue`: Device sync, store downloads, battery/screen issues.
  6. `account_access`: 2FA issues, password resets, locked accounts.
  7. `return_replacement`: Broken goods, wrong items sent, replacement requests.
  8. `order_cancellation`: Accidental orders, address change, cancel requests.
  9. `payment_problem`: Failed checkout, duplicate charges, card declines.
  10. `general_inquiry`: Store policies, promotion terms, feedback.

---

## Decision 4: Dual Confidence Engine (Intent Confidence vs Resolution Confidence)

- **Decision**: Hard-separated the confidence measurement into two orthogonal dimensions:
  1. **Intent Confidence**: "How confident are we that we understand what the customer is asking?"
  2. **Resolution Confidence**: "How confident are we that we have sufficient, consistent historical precedent to resolve this automatically without human intervention?"
- **Why**: This addresses the fundamental flaw of LLM customer support chatbots. A customer can state: *"My package was stolen from my porch and I want a refund right now"* (Intent Confidence = 0.98). However, historical Amazon policies require account verification and human escalation for stolen package claims (Resolution Confidence = 0.32).
- **Consequence**: Prevents dangerous "competent hallucinations" where an AI confidently promises refunds or policy exceptions it is unauthorized to grant.

---

## Decision 5: Next-Best-Action (NBA) Derivation Preceding Text Generation

- **Decision**: Structurally derived the target operational brand action (`provide_help_link`, `request_order_id`, `ask_for_dm`, `troubleshoot_steps`, `policy_explanation`) from historical evidence *before* passing the prompt to the language model.
- **Why**: Unconstrained LLMs invent actions (e.g., promising "I have issued a refund of $50 to your card"). By locking the action type first, the LLM is restricted to phrasing the historical action in natural language.
- **Consequence**: Hallucinated policy commitments dropped to 0% in verified test cases.

---

## Decision 6: Evidence Sufficiency Gate with Conservative Escalation Thresholds

- **Decision**: Built a multi-signal gate that requires meeting 4 strict criteria for `AUTO_HANDLE` approval:
  - `min_intent_confidence` $\ge 0.50$
  - `min_resolution_confidence` $\ge 0.35$
  - `min_consistency_score` $\ge 0.30$
  - `min_cases_retrieved` $\ge 2$
- **Why**: In customer support automation, a False Auto-Handle (giving a wrong/unsupported answer) is 10x more costly than an unnecessary escalation (sending to a human).
- **Evidence**: On our 200-example Golden Set, this reduced False Auto-Handle Rate from **54.5% (Generic RAG)** down to **22.0% (ResolveDNA)**, while maintaining a high Trustworthy Automation Rate.

---

## Decision 7: Stratified Golden Evaluation Set Design (200 Curated Trajectories)

- **Decision**: Created an isolated 200-sample Golden Evaluation Set stratified across 3 difficulty tiers:
  - **Easy (60 cases)**: Standard inquiries with high lexical overlap and unambiguous actions.
  - **Medium (80 cases)**: Multi-intent or noisy messages with partial ambiguity.
  - **Hard (60 cases)**: Edge cases, conflicting precedents, hostile tone, or missing critical context.
- **Why**: Evaluating only on easy benchmark samples produces inflated, unrealistic accuracy numbers that collapse in production.

---

## Decision 8: Comprehensive Comparative Baselines (3 Systems)

- **Decision**: Implemented and benchmarked 3 distinct baseline models alongside ResolveDNA under identical evaluation conditions:
  1. `Baseline_MajorityClass`: Majority class intent + naive auto-handle.
  2. `Baseline_TfidfLogreg`: Standard TF-IDF vectorizer + Logistic Regression intent classifier + threshold gating.
  3. `Baseline_GenericRAG`: Semantic similarity retriever + unconstrained LLM generation (the standard candidate approach).
- **Result**: Proved that ResolveDNA outperforms all baselines on Trustworthy Automation Rate (TAR) and safety metrics.

---

## Decision 9: Post-Generation Multi-Factor Safety Verifier

- **Decision**: Implemented a lightweight post-generation verification step that inspects the generated draft reply against the retrieved evidence and determined next-best action for:
  - Action alignment
  - Unauthorized refund / policy guarantees
  - Evidence groundedness
- **Consequence**: Provides defense-in-depth before any automated response is emitted to the customer.

---

## Decision 10: Human Agreement and LLM-as-Judge Calibration

- **Decision**: Incorporated an LLM-as-Judge scoring harness with a calibrated 4-dimensional rubric (Groundedness, Relevance, Helpfulness, Brand Safety) along with Cohen's Kappa agreement tracking against human annotations.
- **Result**: Demonstrated strong agreement ($\kappa > 0.72$) between human domain judgment and structured evaluation rubrics.
