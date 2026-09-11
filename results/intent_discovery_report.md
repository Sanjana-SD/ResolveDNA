# Support DNA — AmazonHelp Intent Discovery Report

Defines the 10 data-backed empirical intents discovered from `AmazonHelp` Twitter support interactions.

| Intent ID | Intent Name | Description | Requires Account Lookup | Common Resolution Actions |
| :--- | :--- | :--- | :--- | :--- |
| `delivery_status_delay` | **Delivery Status & Delay** | Inquiries regarding delayed packages, tracking updates, late delivery, or estimated arrival times. | Yes | `provide_link`, `clarify_info` |
| `missing_package_delivered` | **Package Marked Delivered But Missing** | Customer claims package is marked delivered in tracking but not physically received. | Yes | `clarify_info`, `provide_link`, `request_dm` |
| `prime_video_streaming_issue` | **Prime Video & Streaming Issue** | Video playback failures, error codes (e.g. 7031, 5004), buffering, app crashes, or audio/subtitle sync problems. | No | `troubleshoot`, `provide_link` |
| `kindle_ebook_device_issue` | **Kindle & E-book Device Issue** | Kindle device freezing, battery drain, e-book sync failure, or purchase download errors. | No | `troubleshoot`, `provide_link` |
| `unrecognized_billing_charge` | **Unrecognized Charge & Billing** | Disputed subscription charge, double billing, unexpected Prime membership fee, or unauthorized transaction. | Yes | `explain_policy`, `request_dm`, `clarify_info` |
| `return_refund_status` | **Return & Refund Status** | Questions about return label generation, refund drop-off confirmation, refund processing timeline. | Yes | `explain_policy`, `provide_link` |
| `gift_card_redemption` | **Gift Card & Balance Issue** | Gift card claim code invalid, redemption error, missing gift card balance, or promo code failure. | Yes | `clarify_info`, `provide_link` |
| `account_access_login` | **Account Access & Security** | Password reset failures, 2FA/OTP issues, locked accounts, or suspected account compromise. | Yes | `provide_link`, `request_dm` |
| `product_defect_damage` | **Damaged or Wrong Item Received** | Customer received a broken, damaged, expired, or wrong item. | Yes | `provide_link`, `clarify_info` |
| `other_unknown` | **Other / Unknown / Ambiguous** | General praise, noise, incomplete context, or out-of-scope customer messages. | No | `general_response` |