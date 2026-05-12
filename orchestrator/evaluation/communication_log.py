import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import tiktoken

@dataclass
class EdgeLog:
    wave_number: int
    from_node_ids: List[str]
    to_id: str
    to_agent: str
    raw_context_injected: str
    context_token_count: int
    original_query: str
    resolved_query: str
    agent_output: Optional[str] = None
    quality_score: Optional[float] = None
    
    # Real-time evaluated metrics
    context_relevance: Optional[float] = None
    utilization_score: Optional[float] = None
    dependency_accuracy: bool = True
    low_confidence: bool = False   # True when CRS gate blocked CUGCR

class CommunicationTracker:
    def __init__(self):
        self.logs: List[EdgeLog] =[]
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None

    def _count_tokens(self, text: str) -> int:
        if not text: return 0
        if self.tokenizer: return len(self.tokenizer.encode(text))
        return len(text.split())

    def log_injection(self, wave: int, from_ids: List[str], to_id: str, to_agent: str, context: str, orig_query: str, resolved_query: str, low_confidence: bool = False) -> EdgeLog:
        log = EdgeLog(
            wave_number=wave, from_node_ids=from_ids, to_id=to_id, to_agent=to_agent,
            raw_context_injected=context, context_token_count=self._count_tokens(context),
            original_query=orig_query, resolved_query=resolved_query, low_confidence=low_confidence
        )
        self.logs.append(log)
        return log

    def update_result(self, to_id: str, output: str, quality: float):
        for log in reversed(self.logs):
            if log.to_id == to_id:
                log.agent_output = output
                log.quality_score = quality
                break

    def export_to_dict(self) -> List[Dict]:
        return [asdict(log) for log in self.logs]

    def clear(self):
        self.logs.clear()

comm_tracker = CommunicationTracker()