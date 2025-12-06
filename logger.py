# logger.py
import os
from datetime import datetime
from typing import Optional

class RunLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._buffer: list[str] = []

    def log(self, message: str):
        print(message)
        self._buffer.append(message)

    def clear(self):
        self._buffer.clear()

    def get_content(self) -> str:
        return "\n".join(self._buffer)

    def save(self, case_id: Optional[str] = None) -> str:
        if not self._buffer:
            self.log("[LOGGER] Buffer is empty. Nothing to save.")
            return ""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if case_id:
            filename = f"log_case_{case_id}_{timestamp}.txt"
        else:
            filename = f"log_run_{timestamp}.txt"
            
        path = os.path.join(self.log_dir, filename)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.get_content())
            
        self.log(f"\n[LOGGER] Results saved to -> {path}")
        
        self.clear()
        
        return path