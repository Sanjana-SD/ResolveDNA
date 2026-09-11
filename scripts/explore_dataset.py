import os
import sys
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import json

DATA_PATH = "data/raw/twcs.csv"

def analyze_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        sys.exit(1)

    file_size_bytes = os.path.getsize(DATA_PATH)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"=== TWCS DATASET OVERVIEW ===")
    print(f"File Path: {DATA_PATH}")
    print(f"File Size: {file_size_mb:.2f} MB")

    # Read sample rows to get schema fast
    df_sample = pd.read_csv(DATA_PATH, nrows=1000)
    print("\nColumns:", df_sample.columns.tolist())
    print("\nSample Rows:")
    print(df_sample.head(3))

    print("\nLoading full dataset...")
    df = pd.read_csv(DATA_PATH, dtype={
        'tweet_id': str,
        'author_id': str,
        'inbound': bool,
        'created_at': str,
        'text': str,
        'response_tweet_id': str,
        'in_response_to_tweet_id': str
    })

    total_rows = len(df)
    print(f"Total Rows (Tweets): {total_rows:,}")
    print(f"Inbound (Customer) Tweets: {df['inbound'].sum():,} ({df['inbound'].mean()*100:.1f}%)")
    print(f"Outbound (Brand) Tweets: {(~df['inbound']).sum():,} ({(1-df['inbound'].mean())*100:.1f}%)")
    print(f"Unique Authors: {df['author_id'].nunique():,}")

    outbound_df = df[~df['inbound']]
    brand_counts = outbound_df['author_id'].value_counts()
    print(f"\nTotal Unique Brands: {len(brand_counts):,}")
    print("\nTop 20 Brands by Response Count:")
    print(brand_counts.head(20))

    top_brands = brand_counts.head(15).index.tolist()

    brand_stats = []

    print("\nBuilding conversation threads for candidate brands...")

    for brand in top_brands:
        brand_outbound = df[df['author_id'] == brand]
        brand_out_count = len(brand_outbound)
        
        brand_tweet_ids = set(brand_outbound['tweet_id'].dropna())
        
        inbound_resp_to_brand = df[df['in_response_to_tweet_id'].isin(brand_tweet_ids)]
        
        brand_in_response_to = set(brand_outbound['in_response_to_tweet_id'].dropna())
        inbound_brand_responded = df[df['tweet_id'].isin(brand_in_response_to)]
        
        all_brand_related_tweets = pd.concat([brand_outbound, inbound_resp_to_brand, inbound_brand_responded]).drop_duplicates(subset=['tweet_id'])
        
        inbound_count = all_brand_related_tweets['inbound'].sum()
        total_related = len(all_brand_related_tweets)
        
        has_response_count = all_brand_related_tweets['response_tweet_id'].notna().sum()
        
        brand_texts = brand_outbound['text'].fillna('')
        dm_mentions = brand_texts.str.contains(r'dm|direct message|inbox', case=False, regex=True).sum()
        dm_ratio = dm_mentions / max(1, brand_out_count)
        
        link_mentions = brand_texts.str.contains(r'http|link|site|page|website', case=False, regex=True).sum()
        link_ratio = link_mentions / max(1, brand_out_count)
        
        apology_mentions = brand_texts.str.contains(r'sorry|apologize|regret|hear that', case=False, regex=True).sum()
        apology_ratio = apology_mentions / max(1, brand_out_count)

        brand_stats.append({
            'brand': brand,
            'outbound_responses': brand_out_count,
            'inbound_customer_messages': int(inbound_count),
            'total_tweets': total_related,
            'dm_request_ratio': round(dm_ratio, 3),
            'link_ratio': round(link_ratio, 3),
            'apology_ratio': round(apology_ratio, 3),
        })

    stats_df = pd.DataFrame(brand_stats)
    print("\nCandidate Brands Detailed Comparison:")
    print(stats_df.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    stats_df.to_csv("results/candidate_brands_summary.csv", index=False)
    print("\nSaved summary to results/candidate_brands_summary.csv")

if __name__ == "__main__":
    analyze_dataset()
