import random
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ServicePool:
    def __init__(self, services: List[Dict[str, Any]]):
        if not services:
            raise ValueError("No LLM services provided.")
        self.services = services
        self.cooldowns = {s["name"]: 0 for s in services}
        self.call_count = {s["name"]: 0 for s in services}

    def get_service(self, exclude: List[str] = None) -> Dict[str, Any]:
        """Returns a random available service. Optionally excludes some."""
        now = time.time()
        exclude = exclude or []
        
        available = [
            s for s in self.services
            if self.cooldowns[s["name"]] <= now and s["name"] not in exclude
        ]

        if not available:
            eligible = [s for s in self.services if s["name"] not in exclude]
            if not eligible:
                raise RuntimeError("no_llm_service")
            soonest = min(eligible, key=lambda s: self.cooldowns[s["name"]])
            wait_time = self.cooldowns[soonest["name"]] - now
            if wait_time > 0:
                raise RuntimeError("llm_rate_limited")
            available = [soonest]
        
        service = random.choice(available)
        self.call_count[service["name"]] += 1
        return service

    def get_service_excluding(self, exclude: List[str]) -> Dict[str, Any]:
        """Wrapper for get_service with exclusion list."""
        return self.get_service(exclude=exclude)

    def mark_rate_limited(self, service_name: str, cooldown_seconds: int = 60):
        self.cooldowns[service_name] = time.time() + cooldown_seconds
        logger.warning(f"Service '{service_name}' rate limited. Cooling down for {cooldown_seconds}s.")

    def stats(self) -> Dict[str, int]:
        return self.call_count.copy()

def load_services_from_file(file_path: str = "llm_services.json") -> List[Dict[str, Any]]:
    path = Path(file_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    from document_qa.core.config import settings
    if settings.LLM_BASE_URL and settings.LLM_API_KEY and settings.LLM_MODEL:
        return [{
            "name": "env",
            "base_url": settings.LLM_BASE_URL,
            "api_key": settings.LLM_API_KEY,
            "model": settings.LLM_MODEL,
        }]
    raise FileNotFoundError(f"LLM services file not found: {file_path}")