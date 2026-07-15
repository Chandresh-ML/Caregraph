"""
Mock backend tools.

In production these would call DTDL's real billing, network-ops, streaming,
and CRM systems. Here they are deterministic, seeded, in-memory stand-ins so
the agent's *logic* can be built and evaluated before real API access exists
-- a normal sequencing decision when the backend team and the agent team
are working in parallel.

Every function returns a plain dict so tool output can be dropped straight
into state and, later, into an LLM prompt without extra marshalling.
"""

from __future__ import annotations
import zlib
from datetime import datetime
from typing import Any, Dict

# --- seeded mock "database" -------------------------------------------------

_USERS: Dict[str, Dict[str, Any]] = {
    "U1001": {
        "name": "Anaya",
        "plan": "Standard",
        "region": "Gurugram-East",
        "duplicate_charge": {
            "is_duplicate": True,
            "item": "World Cup Final PPV",
            "amount_inr": 499,
            "transaction_ids": ["TXN-88213", "TXN-88214"],
        },
    },
    "U1002": {
        "name": "Rohan",
        "plan": "Premium",
        "region": "Bonn-Central",
        "duplicate_charge": {"is_duplicate": False},
    },
    "U1003": {
        "name": "Meera",
        "plan": "Standard",
        "region": "Gurugram-West",
        "duplicate_charge": {"is_duplicate": False},
    },
    "U1004": {
        "name": "Vikram",
        "plan": "Standard",
        "region": "Bonn-Central",
        "duplicate_charge": {"is_duplicate": False},
    },
    "U1005": {
        "name": "Priya",
        "plan": "Premium",
        "region": "Gurugram-East",
        "duplicate_charge": {"is_duplicate": False},
    },
}

_DEFAULT_USER = {
    "name": "Customer",
    "plan": "Standard",
    "region": "Gurugram-East",
    "duplicate_charge": {"is_duplicate": False},
}


def _user(user_id: str) -> Dict[str, Any]:
    return _USERS.get(user_id, _DEFAULT_USER)


# --- tools -------------------------------------------------------------------

def get_user_profile(user_id: str) -> Dict[str, Any]:
    u = _user(user_id)
    return {"user_id": user_id, "name": u["name"], "plan": u["plan"], "region": u["region"]}


def list_demo_users() -> list:
    """Every seeded user, for populating a UI picker. Not itself an
    'agent tool' -- just a convenience for the API/frontend layer."""
    return [{"user_id": uid, **{k: v for k, v in u.items() if k != "duplicate_charge"}} for uid, u in _USERS.items()]


def get_bill(user_id: str) -> Dict[str, Any]:
    u = _user(user_id)
    return {
        "user_id": user_id,
        "plan": u["plan"],
        "current_balance_inr": 1299 if u["plan"] == "Premium" else 799,
        "due_date": "2026-07-20",
    }


def check_payment_status(user_id: str) -> Dict[str, Any]:
    """Looks for duplicate / disputed charges on the account."""
    u = _user(user_id)
    dup = u["duplicate_charge"]
    return {"user_id": user_id, **dup}


def check_network_status(region: str) -> Dict[str, Any]:
    """Simulated network health, worse in regions flagged as congested."""
    congested_regions = {"Gurugram-East", "Gurugram-West"}
    if region in congested_regions:
        return {
            "region": region,
            "status": "congested",
            "reason": "elevated regional load",
            "is_major_outage": False,
            "eta_minutes": 15,
        }
    return {"region": region, "status": "normal", "is_major_outage": False}


def check_stream_status(user_id: str, event: str) -> Dict[str, Any]:
    """Simulated streaming health for a live event, e.g. a World Cup match."""
    u = _user(user_id)
    if u["region"].startswith("Gurugram"):
        return {
            "event": event,
            "status": "degraded",
            "reason": "regional CDN congestion",
            "is_major_outage": False,
            "eta_minutes": 15,
        }
    return {"event": event, "status": "healthy", "is_major_outage": False}


def get_plan_options(user_id: str) -> Dict[str, Any]:
    u = _user(user_id)
    return {
        "user_id": user_id,
        "current_plan": u["plan"],
        "options": [
            {"name": "Premium", "price_inr": 1299, "supports_4k": True},
            {"name": "Ultra", "price_inr": 1799, "supports_4k": True, "extra": "unlimited data rollover"},
        ],
    }


def get_retention_offer(user_id: str) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "offer": "20% off for 3 months",
        "alternative": "1 month free upgrade to Premium",
    }


def raise_ticket(user_id: str, category: str, details: str) -> Dict[str, Any]:
    # zlib.crc32 (unlike Python's built-in hash()) is stable across process
    # runs, which matters for reproducible demos and eval snapshots.
    seed = f"{user_id}|{category}|{details}".encode("utf-8")
    ticket_id = f"DTDL-{zlib.crc32(seed) % 100000:05d}"
    return {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "category": category,
        "details": details,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status": "open",
    }


def apply_service_credit(user_id: str, reason: str) -> Dict[str, Any]:
    return {"user_id": user_id, "credit_applied": True, "reason": reason, "amount_inr": 150}
