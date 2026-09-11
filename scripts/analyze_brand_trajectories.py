import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA_PATH = "data/raw/twcs.csv"

def reconstruct_brand_conversations(brand_name, df):
    print(f"\nReconstructing full conversation trees for: {brand_name}")
    
    # Filter tweets relevant to brand
    # Outbound tweets by brand
    brand_tweets = df[df['author_id'] == brand_name].copy()
    brand_tweet_ids = set(brand_tweets['tweet_id'].dropna())
    
    # Find all root tweets or chained tweets
    # Build adjacency / parent-child mapping
    tweet_dict = df.set_index('tweet_id').to_dict('index')
    
    # We want to find threads involving this brand
    # A thread is a sequence of tweets connected by in_response_to_tweet_id or response_tweet_id
    
    # Collect all customer tweets in response to brand or vice versa
    inbound_to_brand = df[df['in_response_to_tweet_id'].isin(brand_tweet_ids)]
    brand_in_response = set(brand_tweets['in_response_to_tweet_id'].dropna())
    inbound_brand_responded = df[df['tweet_id'].isin(brand_in_response)]
    
    relevant_ids = set(brand_tweets['tweet_id']).union(
        set(inbound_to_brand['tweet_id'])
    ).union(
        set(inbound_brand_responded['tweet_id'])
    )
    
    # Subgraph dataframe
    sub_df = df[df['tweet_id'].isin(relevant_ids)].copy()
    
    # Parent mapping
    parent_map = {}
    for idx, row in sub_df.iterrows():
        tid = row['tweet_id']
        pid = row['in_response_to_tweet_id']
        if pd.notna(pid):
            parent_map[tid] = str(pid)
            
    # Reconstruct chains leading up to or starting from customer questions
    # Root tweets are tweets without in_response_to_tweet_id in our subset or inbound root tweets
    
    # Let's find customer root tweets that got a response from brand
    customer_roots = sub_df[sub_df['inbound'] & sub_df['in_response_to_tweet_id'].isna()]
    
    threads = []
    
    # Map parent_id -> list of child_ids
    children_map = defaultdict(list)
    for tid, pid in parent_map.items():
        children_map[pid].append(tid)
        
    for _, root_row in customer_roots.iterrows():
        root_id = root_row['tweet_id']
        
        # Traverse tree DFS/BFS to build thread turns
        curr_chain = [root_id]
        queue = [root_id]
        visited = set([root_id])
        
        # Follow primary child branch
        curr = root_id
        thread_nodes = [root_id]
        while curr in children_map and children_map[curr]:
            # Take first child for simple linear thread reconstruction
            next_id = children_map[curr][0]
            if next_id in visited:
                break
            visited.add(next_id)
            thread_nodes.append(next_id)
            curr = next_id
            
        if len(thread_nodes) > 1:
            # Reconstruct turn details
            turns = []
            for tid in thread_nodes:
                if tid in tweet_dict:
                    turns.append(tweet_dict[tid])
            threads.append(turns)
            
    print(f"Total reconstructed multi-turn threads starting with customer root: {len(threads):,}")
    
    # Statistics on threads
    thread_lengths = [len(t) for t in threads]
    avg_length = np.mean(thread_lengths) if thread_lengths else 0
    max_length = np.max(thread_lengths) if thread_lengths else 0
    len_2_plus = sum(1 for l in thread_lengths if l >= 2)
    len_3_plus = sum(1 for l in thread_lengths if l >= 3)
    len_4_plus = sum(1 for l in thread_lengths if l >= 4)
    
    # Resolution trajectory analysis
    # Resolution signals in final customer turn:
    res_signals = [r'thank', r'thanks', r'working now', r'fixed', r'awesome', r'great', r'resolved', r'perfect', r'appreciate', r'sorted', r'got it']
    res_pattern = re.compile('|'.join(res_signals), re.IGNORECASE)
    
    repeated_complaint_signals = [r'still', r'not working', r'already', r'waiting', r'no response', r'unresolved', r'useless', r'disappointed', r'worst']
    complaint_pattern = re.compile('|'.join(repeated_complaint_signals), re.IGNORECASE)
    
    resolved_count = 0
    escalated_count = 0
    dm_deflected_count = 0
    in_thread_resolved_count = 0
    
    for t in threads:
        # Check brand actions in thread
        brand_actions = [turn['text'] for turn in t if not turn['inbound']]
        customer_texts = [turn['text'] for turn in t if turn['inbound']]
        
        last_turn = t[-1]
        
        # Check if brand requested DM in any turn
        dm_in_thread = any(re.search(r'dm|direct message|inbox', turn['text'], re.IGNORECASE) for turn in t if not turn['inbound'])
        if dm_in_thread:
            dm_deflected_count += 1
            
        if last_turn['inbound']:
            if res_pattern.search(last_turn['text']):
                resolved_count += 1
                if not dm_in_thread:
                    in_thread_resolved_count += 1
            elif complaint_pattern.search(last_turn['text']):
                escalated_count += 1
                
    print(f"Average Thread Length: {avg_length:.2f} turns (Max: {max_length})")
    print(f"Threads >= 3 turns: {len_3_plus:,} ({len_3_plus/max(1, len(threads))*100:.1f}%)")
    print(f"Threads >= 4 turns: {len_4_plus:,} ({len_4_plus/max(1, len(threads))*100:.1f}%)")
    print(f"Observed Explicit Resolution Signal Rate: {resolved_count/max(1, len(threads))*100:.1f}% ({resolved_count:,} threads)")
    print(f"In-Thread Resolution Rate (No DM required): {in_thread_resolved_count/max(1, len(threads))*100:.1f}% ({in_thread_resolved_count:,} threads)")
    print(f"DM Deflection Rate: {dm_deflected_count/max(1, len(threads))*100:.1f}% ({dm_deflected_count:,} threads)")
    print(f"Repeated Complaint / Frustration Rate: {escalated_count/max(1, len(threads))*100:.1f}% ({escalated_count:,} threads)")
    
    # Print 2 sample multi-turn threads to see real trajectories
    print("\n--- SAMPLE RESOLUTION TRAJECTORIES ---")
    sample_count = 0
    for t in threads:
        if len(t) >= 3:
            sample_count += 1
            print(f"\n[Trajectory Sample #{sample_count} (Length {len(t)})]")
            for turn in t:
                role = "CUSTOMER" if turn['inbound'] else f"BRAND ({turn['author_id']})"
                print(f"  {role}: {turn['text']}")
            if sample_count >= 3:
                break
                
    return {
        'brand': brand_name,
        'total_threads': len(threads),
        'avg_thread_len': round(avg_length, 2),
        'len_3_plus': len_3_plus,
        'explicit_resolution_rate': round(resolved_count/max(1, len(threads))*100, 2),
        'in_thread_resolution_rate': round(in_thread_resolved_count/max(1, len(threads))*100, 2),
        'dm_deflection_rate': round(dm_deflected_count/max(1, len(threads))*100, 2),
        'repeated_complaint_rate': round(escalated_count/max(1, len(threads))*100, 2)
    }

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        sys.exit(1)

    print("Loading TWCS dataset for detailed trajectory analysis...")
    df = pd.read_csv(DATA_PATH, dtype={
        'tweet_id': str,
        'author_id': str,
        'inbound': bool,
        'created_at': str,
        'text': str,
        'response_tweet_id': str,
        'in_response_to_tweet_id': str
    })

    candidate_brands = ["AmazonHelp", "SpotifyCares", "AppleSupport", "Uber_Support", "Delta", "Tesco", "Ask_Spectrum"]
    results = []
    for brand in candidate_brands:
        res = reconstruct_brand_conversations(brand, df)
        results.append(res)

    res_df = pd.DataFrame(results)
    print("\n=== BRAND RESOLUTION TRAJECTORY COMPARISON ===")
    print(res_df.to_string(index=False))
    
    res_df.to_csv("results/brand_trajectory_comparison.csv", index=False)
    print("\nSaved detailed comparison to results/brand_trajectory_comparison.csv")

if __name__ == "__main__":
    main()
