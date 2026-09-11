import os
import sys
import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

RESOLUTION_SIGNALS = [
    r'\bthank\b', r'\bthanks\b', r'\bworking now\b', r'\bfixed\b',
    r'\bawesome\b', r'\bgreat\b', r'\bresolved\b', r'\bperfect\b',
    r'\bappreciate\b', r'\bsorted\b', r'\bgot it\b', r'\bhelped\b'
]
RES_REGEX = re.compile('|'.join(RESOLUTION_SIGNALS), re.IGNORECASE)

COMPLAINT_SIGNALS = [
    r'\bstill\b', r'\bnot working\b', r'\balready\b', r'\bwaiting\b',
    r'\bno response\b', r'\bunresolved\b', r'\buseless\b', r'\bdisappointed\b',
    r'\bworst\b', r'\bhorrible\b', r'\bnever again\b', r'\bfrustrated\b'
]
COMPLAINT_REGEX = re.compile('|'.join(COMPLAINT_SIGNALS), re.IGNORECASE)

BRAND_ACTION_SIGNALS = {
    'request_dm': [r'\bdm\b', r'\bdirect message\b', r'\binbox\b', r'\bprivate message\b'],
    'provide_link': [r'http://', r'https://', r'amazon\.com/help', r'amzn\.to'],
    'troubleshoot': [r'\brestart\b', r'\bclear cache\b', r'\buninstall\b', r'\breinstall\b', r'\bupdate\b', r'\bsettings\b', r'\bpower cycle\b', r'\bsign out\b'],
    'clarify_info': [r'\border number\b', r'\baccount\b', r'\bwhich device\b', r'\berror code\b', r'\bcarrier\b', r'\bthird party\b', r'\bemail address\b'],
    'explain_policy': [r'\bpolicy\b', r'\brefund time\b', r'\bbusiness days\b', r'\bterms\b', r'\bwindow\b'],
    'apologize': [r'\bsorry\b', r'\bapologize\b', r'\bsincerest apologies\b', r'\bregret\b']
}

def detect_brand_actions(text: str) -> List[str]:
    actions = []
    text_lower = text.lower()
    for action, patterns in BRAND_ACTION_SIGNALS.items():
        if any(re.search(p, text_lower) for p in patterns):
            actions.append(action)
    if not actions:
        actions.append('general_response')
    return actions

def parse_amazon_trajectories(csv_path: str, target_brand: str = "AmazonHelp", max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    print(f"[Parser] Loading dataset from {csv_path}...")
    dtype_dict = {
        'tweet_id': str,
        'author_id': str,
        'inbound': bool,
        'created_at': str,
        'text': str,
        'response_tweet_id': str,
        'in_response_to_tweet_id': str
    }
    
    if max_rows:
        df = pd.read_csv(csv_path, dtype=dtype_dict, nrows=max_rows)
    else:
        df = pd.read_csv(csv_path, dtype=dtype_dict)
        
    print(f"[Parser] Total rows loaded: {len(df):,}")
    
    brand_outbound = df[df['author_id'] == target_brand].copy()
    brand_tweet_ids = set(brand_outbound['tweet_id'].dropna())
    print(f"[Parser] Outbound tweets for {target_brand}: {len(brand_outbound):,}")
    
    inbound_to_brand = df[df['in_response_to_tweet_id'].isin(brand_tweet_ids)]
    brand_in_response = set(brand_outbound['in_response_to_tweet_id'].dropna())
    inbound_brand_responded = df[df['tweet_id'].isin(brand_in_response)]
    
    relevant_ids = set(brand_outbound['tweet_id']).union(
        set(inbound_to_brand['tweet_id'])
    ).union(
        set(inbound_brand_responded['tweet_id'])
    )
    
    sub_df = df[df['tweet_id'].isin(relevant_ids)].copy()
    print(f"[Parser] Relevant brand subset tweets: {len(sub_df):,}")
    
    tweet_dict = sub_df.set_index('tweet_id').to_dict('index')
    
    children_dict = {}
    for _, row in sub_df.iterrows():
        tid = row['tweet_id']
        pid = row['in_response_to_tweet_id']
        if pd.notna(pid):
            pid_str = str(pid)
            if pid_str not in children_dict:
                children_dict[pid_str] = []
            children_dict[pid_str].append(tid)
            
    customer_roots = sub_df[sub_df['inbound'] & sub_df['in_response_to_tweet_id'].isna()]
    print(f"[Parser] Found {len(customer_roots):,} customer root tweets")
    
    trajectories = []
    
    for _, root_row in customer_roots.iterrows():
        root_id = root_row['tweet_id']
        
        curr = root_id
        thread_nodes = [root_id]
        visited = {root_id}
        
        while curr in children_dict and children_dict[curr]:
            next_id = children_dict[curr][0]
            if next_id in visited:
                break
            visited.add(next_id)
            thread_nodes.append(next_id)
            curr = next_id
            
        if len(thread_nodes) >= 2:
            turns = []
            brand_actions = []
            customer_reactions = []
            
            for tid in thread_nodes:
                if tid in tweet_dict:
                    t_info = tweet_dict[tid]
                    role = "CUSTOMER" if t_info['inbound'] else "BRAND"
                    text = str(t_info['text']) if pd.notna(t_info['text']) else ""
                    turns.append({
                        'tweet_id': tid,
                        'author_id': t_info['author_id'],
                        'inbound': bool(t_info['inbound']),
                        'created_at': str(t_info['created_at']),
                        'text': text
                    })
                    
                    if not t_info['inbound']:
                        actions = detect_brand_actions(text)
                        brand_actions.extend(actions)
                    else:
                        if tid != root_id:
                            if RES_REGEX.search(text):
                                customer_reactions.append("confirmed_resolution")
                            elif COMPLAINT_REGEX.search(text):
                                customer_reactions.append("repeated_complaint")
                            else:
                                customer_reactions.append("provided_info")
                                
            customer_problem = turns[0]['text']
            last_turn = turns[-1]
            
            dm_deflected = any('request_dm' in detect_brand_actions(t['text']) for t in turns if not t['inbound'])
            has_explicit_res = bool(last_turn['inbound'] and RES_REGEX.search(last_turn['text']))
            has_escalation = bool(last_turn['inbound'] and COMPLAINT_REGEX.search(last_turn['text']))
            
            if has_explicit_res:
                final_state = "RESOLVED_IN_THREAD" if not dm_deflected else "RESOLVED_AFTER_DM"
                observed_resolution_signal = True
            elif has_escalation:
                final_state = "REPEATED_COMPLAINT_ESCALATED"
                observed_resolution_signal = False
            elif dm_deflected:
                final_state = "DM_DEFLECTED"
                observed_resolution_signal = False
            else:
                final_state = "COMPLETED_WITHOUT_EXPLICIT_SIGNAL"
                observed_resolution_signal = False
                
            trajectories.append({
                'conversation_id': root_id,
                'customer_problem': customer_problem,
                'turns': turns,
                'brand_actions': list(set(brand_actions)),
                'customer_reactions': customer_reactions,
                'final_state': final_state,
                'observed_resolution_signal': observed_resolution_signal,
                'trajectory_length': len(turns),
                'created_at': turns[0]['created_at']
            })
            
    print(f"[Parser] Successfully parsed {len(trajectories):,} resolution trajectories")
    return trajectories

if __name__ == "__main__":
    if len(sys.argv) > 1:
        trajectories = parse_amazon_trajectories(sys.argv[1])
        print(f"Sample trajectory #1: {json.dumps(trajectories[0], indent=2)}")
