import json
import os
from datetime import datetime

DRIFT_JSON_PATH  = "monitoring/reports/drift_summary.json"
ALERTS_LOG_PATH  = "monitoring/reports/alerts.log"

def check_and_alert():
    """
    Reads drift_summary.json and logs alerts.
    In production this would send Slack / PagerDuty / email.
    For this project it writes to alerts.log.
    """
    if not os.path.exists(DRIFT_JSON_PATH):
        print("No drift summary found. Run drift_detection.py first.")
        return

    with open(DRIFT_JSON_PATH) as f:
        summary = json.load(f)

    alerts = []
    ts     = datetime.now().isoformat()

    # Alert 1 — dataset drift
    if summary["retrain_needed"]:
        alerts.append({
            "timestamp": ts,
            "level"    : "WARNING",
            "message"  : (
                f"Dataset drift detected: "
                f"{summary['drift_share']:.2%} of features drifted "
                f"(threshold {summary['threshold']:.0%}). "
                f"Retraining triggered."
            )
        })

    # Alert 2 — per-feature critical drift
    for feat, info in summary.get("feature_drift", {}).items():
        if info["drift_detected"] and info["drift_score"] > 0.5:
            alerts.append({
                "timestamp": ts,
                "level"    : "CRITICAL",
                "message"  : (
                    f"High drift on feature '{feat}': "
                    f"score={info['drift_score']:.4f}"
                )
            })

    # Write to log
    os.makedirs("monitoring/reports", exist_ok=True)
    with open(ALERTS_LOG_PATH, "a") as f:
        for alert in alerts:
            line = f"[{alert['level']}] {alert['timestamp']} — {alert['message']}\n"
            f.write(line)
            print(line.strip())

    if not alerts:
        print(f"[OK] {ts} — No alerts. Model healthy.")

    return alerts

if __name__ == "__main__":
    check_and_alert()