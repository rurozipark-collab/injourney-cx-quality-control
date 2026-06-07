"""
Inspection Manager for InJourney CX Quality Control

Handles saving, loading, and querying of inspection records.
Stores inspections as JSON files in reports/audits/ for simplicity and portability.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid

REPORTS_DIR = Path(__file__).parent.parent / "reports"
AUDITS_DIR = REPORTS_DIR / "audits"
AUDITS_DIR.mkdir(parents=True, exist_ok=True)


def generate_audit_id() -> str:
    """Generate a unique inspection ID."""
    return f"INS-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"


def save_audit(audit_data: Dict[str, Any]) -> str:
    """
    Save a completed inspection.
    Returns the inspection_id (stored internally as audit_id for compatibility).
    """
    audit_id = audit_data.get("audit_id") or generate_audit_id()
    audit_data["audit_id"] = audit_id
    audit_data["saved_at"] = datetime.now().isoformat()
    
    filepath = AUDITS_DIR / f"{audit_id}.json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, ensure_ascii=False, indent=2)
    
    return audit_id


def load_audit(audit_id: str) -> Optional[Dict[str, Any]]:
    """Load a single audit by ID."""
    filepath = AUDITS_DIR / f"{audit_id}.json"
    if not filepath.exists():
        return None
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_all_audits() -> List[Dict[str, Any]]:
    """Return metadata for all saved inspections (newest first)."""
    audits = []
    
    for filepath in sorted(AUDITS_DIR.glob("*.json"), reverse=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            audits.append({
                "audit_id": data.get("audit_id"),
                "airport": data.get("airport"),
                "terminal": data.get("terminal", ""),
                "auditor": data.get("auditor"),
                "audit_date": data.get("audit_date"),
                "overall_score": data.get("overall_score"),
                "pillar_scores": data.get("pillar_scores", {}),
                "saved_at": data.get("saved_at"),
                "total_items": len(data.get("checklist_items", [])),
            })
        except Exception:
            continue
    
    return audits


def get_audit_summary(audit_id: str) -> Optional[Dict[str, Any]]:
    """Get a lightweight summary for dashboard display."""
    audit = load_audit(audit_id)
    if not audit:
        return None
    
    return {
        "audit_id": audit["audit_id"],
        "airport": audit["airport"],
        "overall_score": audit.get("overall_score", 0),
        "pillar_scores": audit.get("pillar_scores", {}),
        "recommendations_count": len(audit.get("recommendations", [])),
    }


def delete_audit(audit_id: str) -> bool:
    """Delete an audit record."""
    filepath = AUDITS_DIR / f"{audit_id}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def delete_all_audits() -> int:
    """Delete all audit records. Returns number of files deleted."""
    count = 0
    for filepath in AUDITS_DIR.glob("*.json"):
        try:
            filepath.unlink()
            count += 1
        except Exception:
            continue
    return count


# =============================================================================
# BACKUP FUNCTIONS
# =============================================================================

BACKUPS_DIR = REPORTS_DIR / "backups"
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)


def create_backup(audit_id: str = None) -> str:
    """
    Create a backup of one or all audits.
    If audit_id is provided, backs up only that file.
    Otherwise, backs up all current audits.
    Returns the backup folder name (timestamp).
    """
    from datetime import datetime
    import shutil

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = BACKUPS_DIR / f"backup_{timestamp}"
    backup_folder.mkdir(parents=True, exist_ok=True)

    if audit_id:
        # Backup single audit
        source = AUDITS_DIR / f"{audit_id}.json"
        if source.exists():
            shutil.copy2(source, backup_folder)
            return str(backup_folder)
        return ""
    else:
        # Backup all audits
        for filepath in AUDITS_DIR.glob("*.json"):
            shutil.copy2(filepath, backup_folder)
        return str(backup_folder)


def delete_audit_with_backup(audit_id: str) -> bool:
    """Delete an audit after creating a backup first."""
    backup_path = create_backup(audit_id)
    if not backup_path:
        return False  # Backup failed, don't delete

    return delete_audit(audit_id)


def duplicate_audit(audit_id: str) -> Optional[str]:
    """
    Duplicate an existing inspection with a new ID and current date.
    Returns the new inspection_id if successful.
    """
    original = load_audit(audit_id)
    if not original:
        return None

    # Create a copy with new ID and updated date
    new_audit = original.copy()
    new_audit["audit_id"] = generate_audit_id()
    new_audit["audit_date"] = datetime.now().strftime("%Y-%m-%d")
    new_audit["saved_at"] = datetime.now().isoformat()
    # Add note that this is a duplicate
    new_audit["comments"] = new_audit.get("comments", {})
    new_audit["comments"]["_note"] = f"Duplikat dari {audit_id}"

    return save_audit(new_audit)


def update_audit(audit_id: str, updated_data: Dict[str, Any]) -> bool:
    """
    Update an existing inspection with new scores, comments, etc.
    Preserves original ID and saved_at is updated.
    Returns True if successful.
    """
    try:
        existing = load_audit(audit_id)
        if not existing:
            return False

        # Merge updates while keeping critical fields
        existing.update(updated_data)
        existing["saved_at"] = datetime.now().isoformat()
        existing["audit_id"] = audit_id

        filepath = AUDITS_DIR / f"{audit_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error updating inspection {audit_id}: {e}")
        return False
