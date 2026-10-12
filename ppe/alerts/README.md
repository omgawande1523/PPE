# ppe/alerts

Twilio SMS alerts and violation deduplication, moved from the old `ppe_detection_app/`.

- `config.py`: every setting, read from `.env` in the repository root (template: `.env.example`). No credentials live in code.
- `twilio_alert_service.py`, `rate_limiter.py`, `alert_formatter.py`: sending, per-pattern cooldown and daily cap, message text.
- `deduplicator.py`: suppresses repeat alerts for the same violated-item pattern within `ALERT_COOLDOWN_SECONDS`.
- `check_status.py`: troubleshooting. Run `python -m ppe.alerts.check_status`.

Alerts fire only for a person whose status is `violation`, that is, a `no_*` class was detected on them.
An item with no evidence either way is `unknown` and never triggers an alert.

To enable: copy `.env.example` to `.env`, fill in the four Twilio values, set `ENABLE_TWILIO_ALERTS=true`, then run the dashboard (`python -m app.app`).
