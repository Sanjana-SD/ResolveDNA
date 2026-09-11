# Support DNA — AmazonHelp Resolution Trajectory Analysis

- **Total Multi-Turn Trajectories Analyzed**: 76,799
- **Observed Resolution Signals Rate**: 4.1%

## 1. Final Trajectory State Breakdown

| Final Trajectory State | Count | Percentage |
| :--- | :--- | :--- |
| `COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 71,317 | 92.86% |
| `REPEATED_COMPLAINT_ESCALATED` | 1,408 | 1.83% |
| `RESOLVED_IN_THREAD` | 3,107 | 4.05% |
| `DM_DEFLECTED` | 927 | 1.21% |
| `RESOLVED_AFTER_DM` | 40 | 0.05% |

## 2. Brand Action Effectiveness & Resolution Rates

| Brand Action | Frequency | Thread % | Observed Resolution Rate | Escalation Rate |
| :--- | :--- | :--- | :--- | :--- |
| `provide_link` | 46,074 | 59.99% | **4.71%** | 1.99% |
| `general_response` | 37,272 | 48.53% | **3.2%** | 1.37% |
| `apologize` | 27,417 | 35.7% | **5.72%** | 3.11% |
| `clarify_info` | 9,245 | 12.04% | **6.44%** | 3.16% |
| `troubleshoot` | 2,493 | 3.25% | **4.57%** | 2.57% |
| `request_dm` | 984 | 1.28% | **4.07%** | 1.73% |
| `explain_policy` | 895 | 1.17% | **7.37%** | 2.57% |

## 3. Top Recurring Trajectory Patterns

| Trajectory Pattern Signature | Occurrence Count | % of All Threads |
| :--- | :--- | :--- |
| `[general_response] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 20,850 | 27.15% |
| `[provide_link] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 14,247 | 18.55% |
| `[apologize -> provide_link] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 10,788 | 14.05% |
| `[provide_link -> general_response] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 7,306 | 9.51% |
| `[apologize] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 3,467 | 4.51% |
| `[apologize -> provide_link -> clarify_info] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 2,738 | 3.57% |
| `[apologize -> provide_link -> general_response] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 2,092 | 2.72% |
| `[provide_link -> clarify_info] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 1,091 | 1.42% |
| `[apologize -> general_response] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 1,066 | 1.39% |
| `[apologize -> clarify_info] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 990 | 1.29% |
| `[clarify_info] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 805 | 1.05% |
| `[provide_link -> apologize -> general_response] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 799 | 1.04% |
| `[apologize -> provide_link -> clarify_info -> general_response] => COMPLETED_WITHOUT_EXPLICIT_SIGNAL` | 672 | 0.88% |
| `[apologize -> provide_link] => RESOLVED_IN_THREAD` | 646 | 0.84% |
| `[provide_link] => RESOLVED_IN_THREAD` | 570 | 0.74% |