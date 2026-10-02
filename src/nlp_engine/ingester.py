import sys
import time
import json
import logging
import os
from typing import Iterator

from .announcer import announcer

logger = logging.getLogger("nlp_engine.ingester")

class PayloadIngester:
    def __init__(self, mode: str = "pipe", file_path: str = "payloads.jsonl"):
        self.mode = mode
        self.file_path = file_path
        # Keep a small sliding window of payloads for Scene Reasoner
        self.history = []
        self.history_limit = 30

    def run(self):
        logger.info(f"Starting Ingester in {self.mode} mode.")
        if self.mode == "pipe":
            if sys.stdin.isatty():
                logger.error("ERROR: Running in '--mode pipe' but not receiving piped input!")
                logger.error("You must either run it like: 'python run_pipeline.py | python run_nlp.py --mode pipe'")
                logger.error("OR run it in a separate terminal with: 'python run_nlp.py --mode file-tail'")
                sys.exit(1)
            self._ingest_stdin()
        elif self.mode == "file-tail":
            self._ingest_file()
        else:
            logger.error(f"Unknown mode: {self.mode}")

    def _ingest_stdin(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            self._process_line(line)

    def _ingest_file(self):
        # Wait for file to exist
        while not os.path.exists(self.file_path):
            logger.info(f"Waiting for {self.file_path} to be created...")
            time.sleep(1)
            
        with open(self.file_path, "r", encoding="utf-8") as f:
            # Seek to end so we only process new payloads
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                    
                line = line.strip()
                if not line:
                    continue
                self._process_line(line)

    def _process_line(self, line: str):
        try:
            payload = json.loads(line)
            
            # Maintain history window
            self.history.append(payload)
            if len(self.history) > self.history_limit:
                self.history.pop(0)
                
            # Pass to announcer
            announcer.process_payload(payload)
            
        except json.JSONDecodeError:
            logger.warning("Failed to decode JSON payload")
        except Exception as e:
            logger.error(f"Error processing payload: {e}")
