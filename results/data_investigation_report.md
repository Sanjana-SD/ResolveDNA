# Support DNA — Data Investigation Report (Phase 1)

## 1. Dataset Overview & Schema

- **Dataset Source**: Customer Support on Twitter (`twcs.csv`)
- **File Size**: 516.51 MB (raw CSV)
- **Total Tweets**: 2,811,774
  - Inbound (Customer Tweets): 1,537,843 (54.7%)
  - Outbound (Brand Tweets): 1,273,931 (45.3%)
- **Unique Authors**: 702,777 (108 unique brand handles)
- **Schema**:
  - `tweet_id` (str): Unique tweet identifier
  - `author_id` (str): Anonymized user ID for customers, or public handle for brands (e.g. `@AmazonHelp`)
  - `inbound` (bool): `True` if tweet is customer → brand, `False` if brand → customer
  - `created_at` (str): Timestamp of tweet creation
  - `text` (str): Raw text of the tweet
  - `response_tweet_id` (str): ID(s) of tweets responding to this tweet
  - `in_response_to_tweet_id` (str): ID of tweet this tweet was responding to

---

## 2. Candidate Brand Comparison Matrix

We evaluated the top brand handles in the dataset across 7 quantitative criteria to select the optimal brand for resolution journey modeling:

| Brand Handle | Total Tweets | Brand Responses | Multi-Turn Threads | Avg Thread Length | Threads ≥3 Turns | DM Deflection Rate | In-Thread Resolution Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AmazonHelp** | **358,973** | **169,840** | **76,799** | **3.52 turns** | **40,083 (52.2%)** | **3.47%** | **4.06% (3,120)** |
| **AppleSupport** | 226,755 | 106,860 | 74,613 | 2.65 turns | 21,449 (28.7%) | 63.85% | 1.94% (1,448) |
| **SpotifyCares** | 88,445 | 43,265 | 26,068 | 2.83 turns | 7,608 (29.2%) | 42.81% | 1.02% (266) |
| **Uber_Support** | 122,194 | 56,270 | 39,355 | 2.63 turns | 11,759 (29.9%) | 39.76% | 3.07% (1,208) |
| **Tesco** | 68,670 | 38,573 | 15,339 | 2.92 turns | 5,831 (38.0%) | 37.81% | 3.84% (589) |
| **Ask_Spectrum** | 54,781 | 25,860 | 16,858 | 2.60 turns | 4,369 (25.9%) | 58.00% | 0.95% (160) |
| **Delta** | 83,786 | 42,253 | 24,603 | 2.67 turns | 7,110 (28.9%) | 22.38% | 1.36% (335) |

---

## 3. Brand Selection & Defensible Justification

### Selected Brand: `AmazonHelp`

Why `AmazonHelp` provides the richest resolution environment:

1. **Anti-DM Bias (3.47% DM Deflection vs 63.85% for AppleSupport)**:
   Most customer support datasets suffer from "DM collapse", where brands immediately reply with "Please send us a DM". In `AppleSupport`, 63.8% of interactions end at turn 2 with a dead-end DM request. In `AmazonHelp`, DM deflection is minimal (3.47%). `AmazonHelp` actively resolves issues in public threads by asking clarifying questions, providing self-service links (49.2% link ratio), explaining policies, and guiding device/account troubleshooting.

2. **Richer Multi-Turn Resolution Journeys**:
   `AmazonHelp` has **76,799 multi-turn dialogue trees** starting with customer complaints, with **40,083 threads lasting 3 or more turns** (52.2% multi-turn rate, highest in dataset) and an average of **3.52 turns per thread**.

3. **High Density of Observable Resolution Signals**:
   `AmazonHelp` contains **3,120+ explicit in-thread customer resolution confirmations** (customers replying "thanks", "working now", "fixed", "got it", "appreciate it"), allowing us to construct a robust, empirically grounded **Observed Resolution Signal**.

4. **Problem & Action Diversity**:
   Amazon's support spectrum spans physical logistics (delayed packages, missing items, damaged goods, carrier issues), digital media (Prime Video playback, Kindle ebook sync, Amazon Music), account services (Prime membership renewal, billing charges, gift cards), and hardware (Fire TV, Echo devices).

---

## 4. Evidence of Resolution Trajectories & Trajectory Patterns

Inspecting multi-turn conversations from `AmazonHelp` reveals recurring 4-stage resolution trajectories:

$$\text{Customer Problem} \longrightarrow \text{Brand Action} \longrightarrow \text{Customer Reaction} \longrightarrow \text{Final State (Resolution / Escalation)}$$

### Trajectory Pattern Examples Discovered in Data:

1. **Successful Self-Service Guidance**:
   - Customer: "My Prime video won't stream on TV, error 7031."
   - Brand: "Sorry for the trouble! Please try clearing app cache and restarting your router. Instructions here: amazon.com/help/tv"
   - Customer: "Cleared cache and restarted, working now! Thanks!"
   - *Trajectory State*: **RESOLVED_IN_THREAD**

2. **Required Clarification Before Action**:
   - Customer: "Package says delivered but nothing is at my door!"
   - Brand: "We want to check this! Was this order fulfilled by Amazon or a third-party seller?"
   - Customer: "Fulfilling by Amazon, order #112-382910-1283"
   - Brand: "Thank you! Please check with family/neighbors. If still missing after 36 hrs, request replacement via your orders page."
   - *Trajectory State*: **ACTIONABLE_SELF_SERVICE**

3. **Escalated / Unresolved Complaint**:
   - Customer: "Charged twice for Prime subscription this month!"
   - Brand: "Please check your digital orders page to see if another account shared your card."
   - Customer: "I already checked and called support, nobody is helping me!"
   - *Trajectory State*: **REPEATED_COMPLAINT_ESCALATION**

---

## 5. Next Steps

Having empirically justified the selection of `AmazonHelp` and verified the presence of multi-turn resolution trajectories, we are ready to proceed with:
- **Phase 2 & 3**: Data cleaning & structured resolution trajectory parsing (`src/data/` and `src/resolution/`).
- **Phase 4**: Intent Discovery based on `AmazonHelp` problem clusters.
