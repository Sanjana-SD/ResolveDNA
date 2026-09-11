import os
import sys
import json
from collections import defaultdict, Counter
from typing import List, Dict, Any

def analyze_support_dna(trajectories_path: str) -> Dict[str, Any]:
    print(f"[Pattern Miner] Loading trajectories from {trajectories_path}...")
    trajectories = []
    with open(trajectories_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
                
    total = len(trajectories)
    print(f"[Pattern Miner] Total trajectories loaded: {total:,}")
    
    brand_action_counts = Counter()
    action_resolution_counts = defaultdict(int)
    action_escalation_counts = defaultdict(int)
    
    final_state_counts = Counter()
    resolution_signal_count = 0
    
    trajectory_patterns = Counter()
    
    for traj in trajectories:
        final_state = traj['final_state']
        final_state_counts[final_state] += 1
        
        is_resolved = traj['observed_resolution_signal']
        if is_resolved:
            resolution_signal_count += 1
            
        actions = traj.get('brand_actions', [])
        for act in actions:
            brand_action_counts[act] += 1
            if is_resolved:
                action_resolution_counts[act] += 1
            if final_state == "REPEATED_COMPLAINT_ESCALATED":
                action_escalation_counts[act] += 1
                
        # Pattern signature: Actions sequence + Final State
        actions_str = " -> ".join(actions) if actions else "no_action"
        pattern_sig = f"[{actions_str}] => {final_state}"
        trajectory_patterns[pattern_sig] += 1
        
    action_stats = []
    for action, count in brand_action_counts.most_common():
        res_c = action_resolution_counts[action]
        esc_c = action_escalation_counts[action]
        action_stats.append({
            'brand_action': action,
            'frequency': count,
            'pct_of_threads': round(count / total * 100, 2),
            'observed_resolution_rate': round(res_c / count * 100, 2),
            'escalation_rate': round(esc_c / count * 100, 2)
        })
        
    top_patterns = []
    for pattern, count in trajectory_patterns.most_common(15):
        top_patterns.append({
            'pattern': pattern,
            'count': count,
            'percentage': round(count / total * 100, 2)
        })
        
    summary = {
        'total_trajectories': total,
        'observed_resolution_signals': resolution_signal_count,
        'overall_observed_resolution_rate': round(resolution_signal_count / total * 100, 2),
        'final_state_breakdown': dict(final_state_counts),
        'brand_action_effectiveness': action_stats,
        'top_trajectory_patterns': top_patterns
    }
    return summary

def generate_markdown_report(summary: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    md = []
    md.append("# Support DNA — AmazonHelp Resolution Trajectory Analysis\n")
    md.append(f"- **Total Multi-Turn Trajectories Analyzed**: {summary['total_trajectories']:,}")
    md.append(f"- **Observed Resolution Signals Rate**: {summary['overall_observed_resolution_rate']}%\n")
    
    md.append("## 1. Final Trajectory State Breakdown\n")
    md.append("| Final Trajectory State | Count | Percentage |")
    md.append("| :--- | :--- | :--- |")
    tot = summary['total_trajectories']
    for state, cnt in summary['final_state_breakdown'].items():
        md.append(f"| `{state}` | {cnt:,} | {cnt/tot*100:.2f}% |")
        
    md.append("\n## 2. Brand Action Effectiveness & Resolution Rates\n")
    md.append("| Brand Action | Frequency | Thread % | Observed Resolution Rate | Escalation Rate |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for act in summary['brand_action_effectiveness']:
        md.append(f"| `{act['brand_action']}` | {act['frequency']:,} | {act['pct_of_threads']}% | **{act['observed_resolution_rate']}%** | {act['escalation_rate']}% |")
        
    md.append("\n## 3. Top Recurring Trajectory Patterns\n")
    md.append("| Trajectory Pattern Signature | Occurrence Count | % of All Threads |")
    md.append("| :--- | :--- | :--- |")
    for pat in summary['top_trajectory_patterns']:
        md.append(f"| `{pat['pattern']}` | {pat['count']:,} | {pat['percentage']}% |")
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
        
    print(f"[Pattern Miner] Saved Support DNA report to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        res = analyze_support_dna(sys.argv[1])
        generate_markdown_report(res, "results/support_dna_analysis.md")
