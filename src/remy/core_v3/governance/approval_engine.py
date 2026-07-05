"""
Approval Engine for Remy v3 Governance Layer.

Manages approval gates for actions that require human confirmation.
Phase 5: Enhanced with Telegram bridge, batch operations,
auto-approve policies, and approval history.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

log = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ApprovalRequest:
    """A request for human approval."""
    id: str = field(default_factory=lambda: f"approval_{uuid.uuid4().hex[:10]}")
    action: str = ""
    description: str = ""
    mission_id: str = ""
    specialist: str = ""
    risk_category: str = "low"
    estimated_cost_usd: float = 0.0
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0        # 0 = no expiry
    decided_at: float = 0.0
    decided_by: str = ""           # "telegram", "web", "auto"
    context: dict[str, Any] = field(default_factory=dict)
    denial_reason: str = ""


class ApprovalEngine:
    """Manages approval requests and decisions."""

    def __init__(
        self,
        auto_approve_safe: bool = True,
        expiry_sec: float = 3600,
        notify_callback: Callable[[ApprovalRequest], None] | None = None,
        decision_callback: Callable[[ApprovalRequest], None] | None = None,
    ):
        self.auto_approve_safe = auto_approve_safe
        self.expiry_sec = expiry_sec
        self._queue: list[ApprovalRequest] = []
        self._decided: list[ApprovalRequest] = []  # History of decided requests
        self._decided_max = 200
        self._notify = notify_callback  # Called when new approval needed
        self._decision_callback = decision_callback

    def request_approval(
        self,
        action: str,
        description: str = "",
        mission_id: str = "",
        specialist: str = "",
        risk_category: str = "low",
        cost_usd: float = 0.0,
        context: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        """Submit an action for approval."""
        req = ApprovalRequest(
            action=action,
            description=description,
            mission_id=mission_id,
            specialist=specialist,
            risk_category=risk_category,
            estimated_cost_usd=cost_usd,
            expires_at=time.time() + self.expiry_sec if self.expiry_sec else 0,
            context=context or {},
        )

        # Auto-approve safe actions
        if self.auto_approve_safe and risk_category == "safe":
            req.status = ApprovalStatus.AUTO_APPROVED
            req.decided_at = time.time()
            req.decided_by = "auto"
            self._archive(req)
            self._notify_decision(req)
            log.debug("Auto-approved safe action: %s", action)
            return req

        # Auto-approve low-cost actions (< $0.01)
        if cost_usd < 0.01 and risk_category in ("safe", "low"):
            req.status = ApprovalStatus.AUTO_APPROVED
            req.decided_at = time.time()
            req.decided_by = "auto_low_cost"
            self._archive(req)
            self._notify_decision(req)
            return req

        self._queue.append(req)
        log.info("Approval requested: %s (%s)", action, req.id)

        # Notify via callback (e.g., Telegram)
        if self._notify:
            try:
                self._notify(req)
            except Exception as e:
                log.warning("Approval notification failed: %s", e)

        return req

    def approve(self, request_id: str, decided_by: str = "user") -> bool:
        for req in self._queue:
            if req.id == request_id and req.status == ApprovalStatus.PENDING:
                req.status = ApprovalStatus.APPROVED
                req.decided_at = time.time()
                req.decided_by = decided_by
                self._archive(req)
                self._notify_decision(req)
                return True
        return False

    def deny(self, request_id: str, decided_by: str = "user", reason: str = "") -> bool:
        for req in self._queue:
            if req.id == request_id and req.status == ApprovalStatus.PENDING:
                req.status = ApprovalStatus.DENIED
                req.decided_at = time.time()
                req.decided_by = decided_by
                req.denial_reason = reason
                self._archive(req)
                self._notify_decision(req)
                return True
        return False

    def approve_all(self, decided_by: str = "user") -> int:
        """Batch approve all pending requests."""
        count = 0
        for req in list(self._queue):
            if req.status == ApprovalStatus.PENDING:
                req.status = ApprovalStatus.APPROVED
                req.decided_at = time.time()
                req.decided_by = decided_by
                self._archive(req)
                self._notify_decision(req)
                count += 1
        return count

    def approve_by_action(self, action_pattern: str, decided_by: str = "user") -> int:
        """Approve all pending requests matching an action pattern."""
        import fnmatch
        count = 0
        for req in list(self._queue):
            if req.status == ApprovalStatus.PENDING and fnmatch.fnmatch(req.action, action_pattern):
                req.status = ApprovalStatus.APPROVED
                req.decided_at = time.time()
                req.decided_by = decided_by
                self._archive(req)
                self._notify_decision(req)
                count += 1
        return count

    def expire_stale(self):
        """Expire old pending requests."""
        now = time.time()
        for req in list(self._queue):
            if (req.status == ApprovalStatus.PENDING
                    and req.expires_at > 0
                    and now > req.expires_at):
                req.status = ApprovalStatus.EXPIRED
                self._archive(req)
                self._notify_decision(req)

    def pending(self) -> list[ApprovalRequest]:
        self.expire_stale()
        return [r for r in self._queue if r.status == ApprovalStatus.PENDING]

    def is_approved(self, request_id: str) -> bool:
        # Check queue
        for req in self._queue:
            if req.id == request_id:
                return req.status in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED)
        # Check history
        for req in self._decided:
            if req.id == request_id:
                return req.status in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED)
        return False

    def clear_decided(self):
        """Remove decided requests from queue."""
        self._queue = [
            r for r in self._queue
            if r.status == ApprovalStatus.PENDING
        ]

    def set_notify_callback(self, callback: Callable[[ApprovalRequest], None]):
        """Set notification callback (e.g., Telegram sender)."""
        self._notify = callback

    def set_decision_callback(self, callback: Callable[[ApprovalRequest], None]):
        """Set callback invoked when a request is resolved."""
        self._decision_callback = callback

    def _notify_decision(self, req: ApprovalRequest):
        if self._decision_callback:
            try:
                self._decision_callback(req)
            except Exception as e:
                log.warning("Approval decision notification failed: %s", e)

    def recent_decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent approval decisions for observability."""
        return [
            {
                "id": r.id,
                "action": r.action,
                "description": r.description,
                "status": r.status.value,
                "specialist": r.specialist,
                "risk_category": r.risk_category,
                "decided_by": r.decided_by,
                "cost_usd": r.estimated_cost_usd,
                "wait_sec": round(r.decided_at - r.created_at, 1) if r.decided_at else 0,
                "created_at": r.created_at,
                "decided_at": r.decided_at,
                "context": dict(r.context or {}),
                "routing_pressure": bool(
                    (r.context or {}).get("quality_debt") is not None
                    or "routing pressure" in (r.description or "").lower()
                ),
            }
            for r in reversed(self._decided[-limit:])
        ]

    def _archive(self, req: ApprovalRequest):
        """Move to decided history."""
        self._decided.append(req)
        if len(self._decided) > self._decided_max:
            self._decided = self._decided[-self._decided_max:]
        # Remove from queue
        self._queue = [r for r in self._queue if r.id != req.id]

    def summary(self) -> dict[str, Any]:
        """Summary for observability."""
        self.expire_stale()
        return {
            "pending": len(self.pending()),
            "total_decided": len(self._decided),
            "auto_approved": sum(
                1 for r in self._decided
                if r.status == ApprovalStatus.AUTO_APPROVED
            ),
            "human_approved": sum(
                1 for r in self._decided
                if r.status == ApprovalStatus.APPROVED
            ),
            "denied": sum(
                1 for r in self._decided
                if r.status == ApprovalStatus.DENIED
            ),
            "expired": sum(
                1 for r in self._decided
                if r.status == ApprovalStatus.EXPIRED
            ),
        }
