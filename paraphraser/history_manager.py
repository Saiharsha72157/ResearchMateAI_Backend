# Backend/paraphraser/history_manager.py

import os
import json
import uuid
import datetime
from typing import List, Dict, Any, Optional

class HistoryManager:
    """
    Manages paraphrasing history and favorites.
    Currently uses JSON-based local persistence, structurally prepared
    for immediate PostgreSQL/SQLAlchemy migration.
    """
    def __init__(self, filepath: str = "d:\\dummy21\\PDD\\Backend\\history.json"):
        self.filepath = filepath
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load_data(self) -> List[Dict[str, Any]]:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[HistoryManager] Error loading history: {e}")
            return []

    def _save_data(self, data: List[Dict[str, Any]]) -> bool:
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            print(f"[HistoryManager] Error saving history: {e}")
            return False

    def save_history(self, original_text: str, paraphrased_text: str, mode: str, score: float) -> Dict[str, Any]:
        """Saves a new paraphrasing record to local persistent storage."""
        data = self._load_data()
        
        entry = {
            "id": uuid.uuid4().hex,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "original_text": original_text.strip(),
            "paraphrased_text": paraphrased_text.strip(),
            "mode": mode.lower().strip(),
            "score": round(score, 1),
            "favorite": False
        }
        
        data.append(entry)
        self._save_data(data)
        return entry

    def get_history(self) -> List[Dict[str, Any]]:
        """Retrieves all history records sorted by timestamp descending (newest first)."""
        data = self._load_data()
        return sorted(data, key=lambda x: x.get("timestamp", ""), reverse=True)

    def delete_history(self, entry_id: str) -> bool:
        """Deletes a history record by unique identifier."""
        data = self._load_data()
        filtered = [x for x in data if x.get("id") != entry_id]
        
        if len(filtered) < len(data):
            self._save_data(filtered)
            return True
        return False

    def toggle_favorite(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Toggles the favorite state of a history entry by unique identifier."""
        data = self._load_data()
        updated_entry = None
        
        for x in data:
            if x.get("id") == entry_id:
                x["favorite"] = not x.get("favorite", False)
                updated_entry = x
                break
                
        if updated_entry:
            self._save_data(data)
            return updated_entry
        return None

    def favorite_history(self, entry_id: str) -> bool:
        """Sets the favorite status of a history entry to True."""
        data = self._load_data()
        found = False
        for x in data:
            if x.get("id") == entry_id:
                x["favorite"] = True
                found = True
                break
        if found:
            self._save_data(data)
        return found

    def unfavorite_history(self, entry_id: str) -> bool:
        """Sets the favorite status of a history entry to False."""
        data = self._load_data()
        found = False
        for x in data:
            if x.get("id") == entry_id:
                x["favorite"] = False
                found = True
                break
        if found:
            self._save_data(data)
        return found

    def get_favorites(self) -> List[Dict[str, Any]]:
        """Retrieves all starred favorite records sorted by timestamp descending."""
        data = self._load_data()
        favorites = [x for x in data if x.get("favorite", False)]
        return sorted(favorites, key=lambda x: x.get("timestamp", ""), reverse=True)


# =====================================================================
# 📚 FUTURE-READY ARCHITECTURE FOR POSTGRESQL & ENTERPRISE CLOUD MIGRATION
# =====================================================================
"""
The following SQLAlchemy Declarative Models and SQL DDL Schemas outline the 
seamless drop-in migration to a PostgreSQL server when the system moves to production:

--- PostgreSQL SQL DDL Schema ---

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE paraphrase_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    original_text TEXT NOT NULL,
    paraphrased_text TEXT NOT NULL,
    mode VARCHAR(50) NOT NULL,
    score NUMERIC(3, 1) NOT NULL,
    favorite BOOLEAN DEFAULT FALSE,
    cloud_sync_status VARCHAR(50) DEFAULT 'PENDING', -- PENDING, SYNCED, ERROR
    device_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Optimize queries by creating indices on sorting and reference columns
CREATE INDEX idx_paraphrase_history_user ON paraphrase_history(user_id);
CREATE INDEX idx_paraphrase_history_timestamp ON paraphrase_history(timestamp DESC);
CREATE INDEX idx_paraphrase_history_favorite ON paraphrase_history(user_id) WHERE favorite = TRUE;


--- SQLAlchemy Declarative Models ---

from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, Text, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    
    history = relationship("ParaphraseRecord", back_populates="user", cascade="all, delete-orphan")

class ParaphraseRecord(Base):
    __tablename__ = 'paraphrase_history'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=True) # Nullable for guest/local syncs
    original_text = Column(Text, nullable=False)
    paraphrased_text = Column(Text, nullable=False)
    mode = Column(String(50), nullable=False)
    score = Column(Float, nullable=False)
    favorite = Column(Boolean, default=False)
    cloud_sync_status = Column(String(50), default="PENDING") # Supports background offline synchronization queues
    device_id = Column(String(255), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="history")
    
    __table_args__ = (
        Index('idx_user_time', 'user_id', 'timestamp'),
    )

--- Cloud Synchronization & Analytics Ready ---
1. Cloud Sync Queue: The `cloud_sync_status` column ensures that when offline devices make modifications, they are marked as 'PENDING' and synchronized via a service worker / task queue when connectivity resumes.
2. Analytics Models: Aggregate analytics queries can easily compute:
   - Average quality score by user/device over time.
   - Most frequently utilized modes (Standard vs Academic).
   - Paragraph/word volume processed to measure API throttling and billing.
3. PDF/Export Services: The structure isolates individual record rows cleanly, making it ready to be bound to PDF rendering engines (like ReportLab or Weasyprint) to generate elegant scholarly reports.
"""
