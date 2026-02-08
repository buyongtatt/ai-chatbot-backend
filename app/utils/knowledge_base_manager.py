import json
import os
from typing import Dict, Any, Optional, List
from functools import lru_cache

KB_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "knowledge_bases.json")

class KnowledgeBaseManager:
    """Manages knowledge base configurations and URL lookups"""
    
    def __init__(self, config_path: str = KB_CONFIG_PATH):
        self.config_path = config_path
        self._config_cache: Optional[Dict[str, Any]] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load knowledge base configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self._config_cache = json.load(f)
                print(f"[KBManager] Loaded KB config from {self.config_path}")
            else:
                print(f"[KBManager] KB config file not found at {self.config_path}")
                self._config_cache = {"knowledge_bases": []}
        except Exception as e:
            print(f"[KBManager] Error loading KB config: {e}")
            self._config_cache = {"knowledge_bases": []}
    
    def reload_config(self) -> None:
        """Reload configuration from file"""
        self._config_cache = None
        self._load_config()
    
    def get_url_by_area(self, area_name: str) -> Optional[str]:
        """Get URL for a knowledge base area"""
        if not self._config_cache:
            return None
        
        area_name = area_name.strip().lower()
        
        for kb in self._config_cache.get("knowledge_bases", []):
            if kb.get("area_name", "").lower() == area_name:
                return kb.get("url")
        
        return None
    
    def get_area_info(self, area_name: str) -> Optional[Dict[str, Any]]:
        """Get full info for a knowledge base area"""
        if not self._config_cache:
            return None
        
        area_name = area_name.strip().lower()
        
        for kb in self._config_cache.get("knowledge_bases", []):
            if kb.get("area_name", "").lower() == area_name:
                return kb
        
        return None
    
    def get_all_areas(self) -> List[Dict[str, Any]]:
        """Get all available knowledge base areas"""
        if not self._config_cache:
            return []
        
        return self._config_cache.get("knowledge_bases", [])
    
    def list_area_names(self) -> List[str]:
        """Get list of all area names"""
        return [kb.get("area_name") for kb in self.get_all_areas() if kb.get("area_name")]
    
    def validate_area(self, area_name: str) -> tuple[bool, Optional[str]]:
        """Validate if area exists and return URL if valid"""
        url = self.get_url_by_area(area_name)
        if url:
            return True, url
        return False, None
    
    def add_knowledge_base(self, area_name: str, display_name: str, url: str, description: str = "") -> None:
        """Add a new knowledge base area"""
        if not self._config_cache:
            self._config_cache = {"knowledge_bases": []}
        
        # Check if area already exists
        for kb in self._config_cache["knowledge_bases"]:
            if kb.get("area_name", "").lower() == area_name.lower():
                print(f"[KBManager] Area '{area_name}' already exists")
                return
        
        # Add new area
        self._config_cache["knowledge_bases"].append({
            "area_name": area_name.lower(),
            "display_name": display_name,
            "url": url,
            "description": description
        })
        
        self._save_config()
    
    def _save_config(self) -> None:
        """Save configuration back to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self._config_cache, f, indent=2)
            print(f"[KBManager] Saved KB config to {self.config_path}")
        except Exception as e:
            print(f"[KBManager] Error saving KB config: {e}")


# Global instance
kb_manager = KnowledgeBaseManager()
