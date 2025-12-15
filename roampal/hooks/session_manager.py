"""
SessionManager - Tracks exchanges for enforced outcome scoring

Stores exchanges in JSONL files so the Stop hook can:
1. Load the previous exchange
2. Inject scoring prompt into UserPromptSubmit
3. Verify record_response() was called before allowing stop
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages conversation sessions for roampal-core.

    Session files store exchanges in JSONL format:
    {"role":"user","content":"...","timestamp":"..."}
    {"role":"assistant","content":"...","doc_id":"...","timestamp":"..."}

    The doc_id links to ChromaDB for outcome scoring.

    Completion Tracking:
    - Stop hook sets assistant_completed=True when assistant finishes
    - UserPromptSubmit checks this flag to decide if scoring is needed
    - This prevents scoring prompts during mid-work interruptions
    """

    def __init__(self, data_path: Path):
        """
        Initialize session manager.

        Args:
            data_path: Root data directory (e.g., %APPDATA%/Roampal/data)
        """
        self.data_path = Path(data_path)
        self.sessions_dir = self.data_path / "mcp_sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of last exchange per session (for fast lookup)
        self._last_exchange_cache: Dict[str, Dict[str, Any]] = {}

        # Completion state file (persists across hook invocations)
        self._state_file = self.sessions_dir / "_completion_state.json"

        logger.info(f"SessionManager initialized: {self.sessions_dir}")

    def _get_session_file(self, conversation_id: str) -> Path:
        """Get path to session file."""
        # Sanitize conversation_id for filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in conversation_id)
        return self.sessions_dir / f"{safe_id}.jsonl"

    async def store_exchange(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        doc_id: str
    ) -> Dict[str, Any]:
        """
        Store a completed exchange.

        Called by Stop hook after Claude responds.

        Args:
            conversation_id: Session identifier
            user_message: What the user said
            assistant_response: What Claude said
            doc_id: ChromaDB document ID for outcome scoring

        Returns:
            Exchange record that was stored
        """
        session_file = self._get_session_file(conversation_id)
        timestamp = datetime.now().isoformat()

        # Create exchange records
        user_record = {
            "role": "user",
            "content": user_message,
            "timestamp": timestamp
        }

        assistant_record = {
            "role": "assistant",
            "content": assistant_response,
            "doc_id": doc_id,
            "timestamp": timestamp,
            "scored": False  # Will be set True when record_response() is called
        }

        # Append to session file
        with open(session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(user_record) + "\n")
            f.write(json.dumps(assistant_record) + "\n")

        # Update cache
        self._last_exchange_cache[conversation_id] = {
            "user": user_message,
            "assistant": assistant_response,
            "doc_id": doc_id,
            "timestamp": timestamp,
            "scored": False
        }

        logger.info(f"Stored exchange for {conversation_id}, doc_id={doc_id}")

        return assistant_record

    async def get_previous_exchange(
        self,
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the previous (unscored) exchange for scoring.

        Called by UserPromptSubmit hook to inject scoring prompt.

        Args:
            conversation_id: Session identifier

        Returns:
            Previous exchange or None if no unscored exchange exists
        """
        # Check cache first
        if conversation_id in self._last_exchange_cache:
            cached = self._last_exchange_cache[conversation_id]
            if not cached.get("scored", False):
                return cached

        # Fall back to reading file
        session_file = self._get_session_file(conversation_id)
        if not session_file.exists():
            return None

        # Read backwards to find last unscored assistant message
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Find last assistant message that hasn't been scored
            last_user = None
            for line in reversed(lines):
                try:
                    record = json.loads(line.strip())
                    if record.get("role") == "user" and last_user is None:
                        # Keep track of the user message that precedes assistant
                        pass
                    if record.get("role") == "assistant":
                        if not record.get("scored", False):
                            # Find the user message before this
                            idx = lines.index(line)
                            if idx > 0:
                                prev_line = lines[idx - 1]
                                user_record = json.loads(prev_line.strip())
                                if user_record.get("role") == "user":
                                    return {
                                        "user": user_record.get("content", ""),
                                        "assistant": record.get("content", ""),
                                        "doc_id": record.get("doc_id"),
                                        "timestamp": record.get("timestamp"),
                                        "scored": False
                                    }
                        break  # Only check the last assistant message
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.error(f"Error reading session file: {e}")

        return None

    async def get_most_recent_unscored_exchange(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent unscored exchange across ALL sessions.

        This handles the MCP/hook session ID mismatch by finding
        the most recent exchange regardless of which session it's in.

        Returns:
            Most recent unscored exchange with conversation_id, or None
        """
        most_recent = None
        most_recent_time = None

        # Scan all session files
        for session_file in self.sessions_dir.glob("*.jsonl"):
            conversation_id = session_file.stem

            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Find last unscored assistant message
                for i in range(len(lines) - 1, -1, -1):
                    try:
                        record = json.loads(lines[i].strip())
                        if record.get("role") == "assistant" and not record.get("scored", False):
                            timestamp = record.get("timestamp", "")

                            # Check if this is more recent
                            if most_recent_time is None or timestamp > most_recent_time:
                                # Find the user message before this
                                if i > 0:
                                    prev_line = lines[i - 1]
                                    user_record = json.loads(prev_line.strip())
                                    if user_record.get("role") == "user":
                                        most_recent = {
                                            "user": user_record.get("content", ""),
                                            "assistant": record.get("content", ""),
                                            "doc_id": record.get("doc_id"),
                                            "timestamp": timestamp,
                                            "scored": False,
                                            "conversation_id": conversation_id
                                        }
                                        most_recent_time = timestamp
                            break  # Only check the last assistant message per file
                    except json.JSONDecodeError:
                        continue

            except Exception as e:
                logger.error(f"Error reading session file {session_file}: {e}")
                continue

        if most_recent:
            logger.info(f"Found most recent unscored exchange in session {most_recent.get('conversation_id')}")

        return most_recent

    async def mark_scored(
        self,
        conversation_id: str,
        doc_id: str,
        outcome: str
    ) -> bool:
        """
        Mark an exchange as scored.

        Called when record_response() MCP tool is invoked.

        Args:
            conversation_id: Session identifier
            doc_id: Document ID that was scored
            outcome: The outcome that was recorded

        Returns:
            True if successfully marked
        """
        # Update cache
        if conversation_id in self._last_exchange_cache:
            if self._last_exchange_cache[conversation_id].get("doc_id") == doc_id:
                self._last_exchange_cache[conversation_id]["scored"] = True
                self._last_exchange_cache[conversation_id]["outcome"] = outcome

        # Update file (rewrite the last assistant record with scored=True)
        session_file = self._get_session_file(conversation_id)
        if not session_file.exists():
            return False

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Find and update the assistant record with this doc_id
            updated = False
            for i in range(len(lines) - 1, -1, -1):
                try:
                    record = json.loads(lines[i].strip())
                    if record.get("role") == "assistant" and record.get("doc_id") == doc_id:
                        record["scored"] = True
                        record["outcome"] = outcome
                        lines[i] = json.dumps(record) + "\n"
                        updated = True
                        break
                except json.JSONDecodeError:
                    continue

            if updated:
                with open(session_file, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                logger.info(f"Marked {doc_id} as scored with outcome={outcome}")
                return True

        except Exception as e:
            logger.error(f"Error marking scored: {e}")

        return False

    async def get_session_history(
        self,
        conversation_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.

        Args:
            conversation_id: Session identifier
            limit: Max messages to return

        Returns:
            List of message records
        """
        session_file = self._get_session_file(conversation_id)
        if not session_file.exists():
            return []

        messages = []
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines[-limit:]:
                try:
                    record = json.loads(line.strip())
                    messages.append(record)
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.error(f"Error reading history: {e}")

        return messages

    def build_scoring_prompt(
        self,
        previous_exchange: Dict[str, Any],
        current_user_message: str,
        surfaced_memories: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build the scoring prompt to inject.

        This is what Claude sees at the start of each turn,
        prompting them to score the previous exchange.

        Args:
            previous_exchange: The exchange to score
            current_user_message: What the user just said (for context)
            surfaced_memories: Optional list of memories that were injected (for selective scoring)

        Returns:
            Formatted prompt to prepend
        """
        user_asked = previous_exchange.get("user", "")
        assistant_said = previous_exchange.get("assistant", "")

        # Build memories section if we have surfaced memories
        memories_section = ""
        if surfaced_memories:
            memory_lines = []
            for mem in surfaced_memories:
                doc_id = mem.get("id", mem.get("doc_id", "unknown"))
                content = mem.get("content") or mem.get("text") or mem.get("metadata", {}).get("content", "")
                # Truncate long content for display
                if len(content) > 100:
                    content = content[:100] + "..."
                memory_lines.append(f"• [{doc_id}] \"{content}\"")

            if memory_lines:
                memories_section = "\nMemories surfaced:\n" + "\n".join(memory_lines) + "\n"

        # Build selective scoring instruction
        scoring_instruction = "Call score_response(outcome=\"worked|failed|partial|unknown\") FIRST, then respond."
        if surfaced_memories:
            scoring_instruction = """Call score_response(outcome="...", related=["doc_ids that were relevant"]) FIRST.
- related is optional: omit to score all, or list only the memories you actually used"""

        return f"""<roampal-score-required>
Score the previous exchange before responding.

Previous:
- User asked: "{user_asked}"
- You answered: "{assistant_said}"
{memories_section}
Current user message: "{current_user_message}"

Based on the user's current message, evaluate if your previous answer helped:
- "worked" = user satisfied, says thanks, moves on to new topic
- "failed" = user corrects you, says no/wrong, repeats question
- "partial" = lukewarm response, "kind of", "I guess"
- "unknown" = no clear signal

{scoring_instruction}

Separately, record_response(key_takeaway="...") is OPTIONAL - only for significant learnings.
</roampal-score-required>

"""

    # ========== Completion State Tracking ==========
    # These methods track whether the assistant has completed a response,
    # so we only inject scoring prompts when the user responds to completed work.

    def _load_completion_state(self) -> Dict[str, Any]:
        """Load completion state from file."""
        if not self._state_file.exists():
            return {}
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error loading completion state: {e}")
            return {}

    def _save_completion_state(self, state: Dict[str, Any]) -> None:
        """Save completion state to file."""
        try:
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Error saving completion state: {e}")

    def set_completed(self, conversation_id: str) -> None:
        """
        Mark that the assistant has completed a response.

        Called by Stop hook when assistant finishes responding.
        This signals that the next user message should trigger scoring.

        Args:
            conversation_id: Session identifier
        """
        state = self._load_completion_state()
        # MERGE with existing state to preserve first_message_seen and other flags
        if conversation_id not in state:
            state[conversation_id] = {}
        state[conversation_id]["completed"] = True
        state[conversation_id]["timestamp"] = datetime.now().isoformat()
        state[conversation_id]["scoring_required"] = False  # Will be set by check_and_clear_completed
        self._save_completion_state(state)
        logger.info(f"Marked conversation {conversation_id} as completed")

    def set_scoring_required(self, conversation_id: str, required: bool) -> None:
        """
        Track that scoring was required this turn.

        Called by get-context endpoint when it injects a scoring prompt.
        The Stop hook uses this to decide whether to block.

        Args:
            conversation_id: Session identifier
            required: Whether scoring prompt was injected
        """
        state = self._load_completion_state()
        if conversation_id not in state:
            state[conversation_id] = {}
        state[conversation_id]["scoring_required"] = required
        # Reset scored_this_turn when starting a new turn that requires scoring
        if required:
            state[conversation_id]["scored_this_turn"] = False
        self._save_completion_state(state)
        logger.info(f"Set scoring_required={required} for {conversation_id}")

    def was_scoring_required(self, conversation_id: str) -> bool:
        """
        Check if scoring was required this turn.

        Called by Stop hook to decide whether to block.

        Args:
            conversation_id: Session identifier

        Returns:
            True if scoring prompt was injected this turn
        """
        state = self._load_completion_state()
        return state.get(conversation_id, {}).get("scoring_required", False)

    def set_scored_this_turn(self, conversation_id: str, scored: bool = True) -> None:
        """
        Track that scoring was completed this turn.

        Called by record_response endpoint when LLM calls record_response MCP tool.
        The Stop hook uses this to decide whether to block.

        Args:
            conversation_id: Session identifier
            scored: Whether scoring was completed (default True)
        """
        state = self._load_completion_state()
        if conversation_id not in state:
            state[conversation_id] = {}
        state[conversation_id]["scored_this_turn"] = scored
        self._save_completion_state(state)
        logger.info(f"Set scored_this_turn={scored} for {conversation_id}")

    def was_scored_this_turn(self, conversation_id: str) -> bool:
        """
        Check if scoring was completed this turn.

        Called by Stop hook to decide whether to block.

        Args:
            conversation_id: Session identifier

        Returns:
            True if record_response was called this turn
        """
        state = self._load_completion_state()
        return state.get(conversation_id, {}).get("scored_this_turn", False)

    def check_and_clear_completed(self, conversation_id: str) -> bool:
        """
        Check if assistant completed and clear the flag.

        Called by UserPromptSubmit hook to decide if scoring is needed.
        Returns True and clears the flag if assistant had completed.
        Returns False if assistant was mid-work (no scoring needed).

        Args:
            conversation_id: Session identifier

        Returns:
            True if assistant had completed (scoring should happen)
            False if assistant was mid-work (no scoring needed)
        """
        state = self._load_completion_state()

        if conversation_id in state and state[conversation_id].get("completed"):
            # Clear only the completed flag, preserve other state (first_message_seen, etc.)
            state[conversation_id]["completed"] = False
            self._save_completion_state(state)
            logger.info(f"Conversation {conversation_id} was completed - scoring needed")
            return True

        logger.info(f"Conversation {conversation_id} not completed - skip scoring")
        return False

    def is_completed(self, conversation_id: str) -> bool:
        """
        Check if assistant completed without clearing flag.

        Useful for checking state without side effects.

        Args:
            conversation_id: Session identifier

        Returns:
            True if assistant had completed
        """
        state = self._load_completion_state()
        return conversation_id in state and state[conversation_id].get("completed", False)

    # ========== Cold Start / First Message Tracking ==========
    # Track which sessions have had their first message, so we can dump
    # the full user profile on cold start.

    def is_first_message(self, conversation_id: str) -> bool:
        """
        Check if this is the first message in a session.

        Called by get-context to decide if cold start user profile dump is needed.

        Args:
            conversation_id: Session identifier

        Returns:
            True if this is the first message (cold start)
        """
        state = self._load_completion_state()
        return not state.get(conversation_id, {}).get("first_message_seen", False)

    def mark_first_message_seen(self, conversation_id: str) -> None:
        """
        Mark that the first message has been seen for this session.

        Called by get-context after injecting cold start profile.

        Args:
            conversation_id: Session identifier
        """
        state = self._load_completion_state()
        if conversation_id not in state:
            state[conversation_id] = {}
        state[conversation_id]["first_message_seen"] = True
        state[conversation_id]["first_message_timestamp"] = datetime.now().isoformat()
        self._save_completion_state(state)
        logger.info(f"Marked first message seen for {conversation_id}")
