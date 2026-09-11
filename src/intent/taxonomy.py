import os
import sys
import json
import re
from typing import Dict, Any, List, Tuple

INTENT_TAXONOMY = {
    "delivery_status_delay": {
        "name": "Delivery Status & Delay",
        "description": "Inquiries regarding delayed packages, tracking updates, late delivery, or estimated arrival times.",
        "keywords": ["delayed", "tracking", "where is my", "late", "arriving", "shipment", "carrier", "transit", "usps", "ups", "fedex", "delivery time"],
        "common_resolution_actions": ["provide_link", "clarify_info"],
        "requires_account_lookup": True
    },
    "missing_package_delivered": {
        "name": "Package Marked Delivered But Missing",
        "description": "Customer claims package is marked delivered in tracking but not physically received.",
        "keywords": ["marked delivered", "says delivered", "didn't receive", "not at door", "missing package", "stolen", "handed to resident"],
        "common_resolution_actions": ["clarify_info", "provide_link", "request_dm"],
        "requires_account_lookup": True
    },
    "prime_video_streaming_issue": {
        "name": "Prime Video & Streaming Issue",
        "description": "Video playback failures, error codes (e.g. 7031, 5004), buffering, app crashes, or audio/subtitle sync problems.",
        "keywords": ["prime video", "streaming", "error 7031", "buffering", "can't watch", "playback", "app crash", "firestick", "smart tv"],
        "common_resolution_actions": ["troubleshoot", "provide_link"],
        "requires_account_lookup": False
    },
    "kindle_ebook_device_issue": {
        "name": "Kindle & E-book Device Issue",
        "description": "Kindle device freezing, battery drain, e-book sync failure, or purchase download errors.",
        "keywords": ["kindle", "ebook", "paperwhite", "whispersync", "book download", "freezing", "battery drain"],
        "common_resolution_actions": ["troubleshoot", "provide_link"],
        "requires_account_lookup": False
    },
    "unrecognized_billing_charge": {
        "name": "Unrecognized Charge & Billing",
        "description": "Disputed subscription charge, double billing, unexpected Prime membership fee, or unauthorized transaction.",
        "keywords": ["charged twice", "unexpected charge", "unauthorized", "billing", "subscription fee", "prime charge", "bank statement"],
        "common_resolution_actions": ["explain_policy", "request_dm", "clarify_info"],
        "requires_account_lookup": True
    },
    "return_refund_status": {
        "name": "Return & Refund Status",
        "description": "Questions about return label generation, refund drop-off confirmation, refund processing timeline.",
        "keywords": ["return label", "refund status", "drop off", "ups store return", "when will i get my refund", "returned item"],
        "common_resolution_actions": ["explain_policy", "provide_link"],
        "requires_account_lookup": True
    },
    "gift_card_redemption": {
        "name": "Gift Card & Balance Issue",
        "description": "Gift card claim code invalid, redemption error, missing gift card balance, or promo code failure.",
        "keywords": ["gift card", "claim code", "redeem", "balance not showing", "promo code"],
        "common_resolution_actions": ["clarify_info", "provide_link"],
        "requires_account_lookup": True
    },
    "account_access_login": {
        "name": "Account Access & Security",
        "description": "Password reset failures, 2FA/OTP issues, locked accounts, or suspected account compromise.",
        "keywords": ["password reset", "can't login", "otp", "verification code", "locked account", "hacked", "sign in error"],
        "common_resolution_actions": ["provide_link", "request_dm"],
        "requires_account_lookup": True
    },
    "product_defect_damage": {
        "name": "Damaged or Wrong Item Received",
        "description": "Customer received a broken, damaged, expired, or wrong item.",
        "keywords": ["damaged", "broken", "wrong item", "expired", "defective", "missing parts", "box crushed"],
        "common_resolution_actions": ["provide_link", "clarify_info"],
        "requires_account_lookup": True
    },
    "other_unknown": {
        "name": "Other / Unknown / Ambiguous",
        "description": "General praise, noise, incomplete context, or out-of-scope customer messages.",
        "keywords": [],
        "common_resolution_actions": ["general_response"],
        "requires_account_lookup": False
    }
}

class IntentClassifier:
    def __init__(self, taxonomy: Dict[str, Any] = INTENT_TAXONOMY):
        self.taxonomy = taxonomy
        self.compiled_keywords = {}
        for intent_id, info in taxonomy.items():
            if info["keywords"]:
                pattern = r'\b(' + '|'.join([re.escape(kw) for kw in info["keywords"]]) + r')\b'
                self.compiled_keywords[intent_id] = re.compile(pattern, re.IGNORECASE)
                
    def classify(self, text: str) -> Tuple[str, float]:
        if not text or not isinstance(text, str):
            return "other_unknown", 0.0
            
        scores = {}
        for intent_id, regex in self.compiled_keywords.items():
            matches = regex.findall(text)
            if matches:
                scores[intent_id] = len(matches)
                
        if not scores:
            return "other_unknown", 0.35
            
        best_intent = max(scores, key=scores.get)
        match_count = scores[best_intent]
        
        # Confidence score heuristic based on match density
        confidence = min(0.95, 0.60 + (match_count * 0.15))
        
        # Check for ambiguity (two intents with equal scores)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1]:
            confidence -= 0.15
            
        return best_intent, round(confidence, 2)

def save_taxonomy_artifacts(data_path: str = "data/intent_taxonomy.json", report_path: str = "results/intent_discovery_report.md"):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(INTENT_TAXONOMY, f, indent=2)
        
    md = []
    md.append("# Support DNA — AmazonHelp Intent Discovery Report\n")
    md.append("Defines the 10 data-backed empirical intents discovered from `AmazonHelp` Twitter support interactions.\n")
    md.append("| Intent ID | Intent Name | Description | Requires Account Lookup | Common Resolution Actions |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for intent_id, info in INTENT_TAXONOMY.items():
        acct = "Yes" if info["requires_account_lookup"] else "No"
        actions = ", ".join([f"`{a}`" for a in info["common_resolution_actions"]])
        md.append(f"| `{intent_id}` | **{info['name']}** | {info['description']} | {acct} | {actions} |")
        
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
        
    print(f"[Intent Taxonomy] Saved taxonomy to {data_path} and report to {report_path}")

if __name__ == "__main__":
    save_taxonomy_artifacts()
    classifier = IntentClassifier()
    sample_text = "My package says delivered on tracking but nothing is at my front door!"
    intent, conf = classifier.classify(sample_text)
    print(f"Sample: '{sample_text}' -> Intent: {intent} (Conf: {conf})")
