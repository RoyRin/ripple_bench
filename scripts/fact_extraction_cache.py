#!/usr/bin/env python3
"""
Caching system for fact extraction to avoid redundant API calls
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional


class FactExtractionCache:

    def __init__(self, cache_dir: str = ".fact_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}  # In-memory cache for current session

    def _get_cache_key(self, content: str, topic: str, model: str) -> str:
        """Generate a unique cache key based on content, topic, and model"""
        # Use first 1000 chars of content + topic + model for key
        cache_string = f"{content[:1000]}|{topic}|{model}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def get(self, content: str, topic: str, model: str) -> Optional[str]:
        """Get cached facts if they exist"""
        cache_key = self._get_cache_key(content, topic, model)

        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                facts = data['facts']
                # Add to memory cache
                self.memory_cache[cache_key] = facts
                return facts

        return None

    def set(self, content: str, topic: str, model: str, facts: str):
        """Cache the extracted facts"""
        cache_key = self._get_cache_key(content, topic, model)

        # Save to memory cache
        self.memory_cache[cache_key] = facts

        # Save to disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_data = {
            'topic': topic,
            'model': model,
            'content_preview': content[:200],
            'facts': facts
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        disk_files = len(list(self.cache_dir.glob("*.json")))
        return {
            'memory_entries': len(self.memory_cache),
            'disk_entries': disk_files,
            'cache_dir': str(self.cache_dir)
        }
