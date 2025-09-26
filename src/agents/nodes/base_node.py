"""
Base node class for LangGraph agent nodes.
Provides common functionality and interface for all agent nodes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import time
from contextlib import contextmanager

from ..state import AgentState, NodeResult, NodeConfig


class BaseAgentNode(ABC):
    """
    Abstract base class for all agent nodes.
    Provides common functionality like logging, error handling, and state management.
    """

    def __init__(self, config: NodeConfig):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for this node."""
        logger = logging.getLogger(f"excel_agent.{self.config.node_name}")
        if self.config.enable_logging and not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.config.node_name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @abstractmethod
    async def execute(self, state: AgentState) -> NodeResult:
        """
        Execute the node's main functionality.
        Must be implemented by subclasses.
        """
        pass

    @contextmanager
    def _execution_context(self, state: AgentState):
        """Context manager for node execution with timing and error handling."""
        start_time = time.time()
        self.logger.info(f"Starting execution of {self.config.node_name}")
        state.add_message(f"Starting {self.config.node_name}")

        try:
            yield
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error in {self.config.node_name}: {str(e)}"
            self.logger.error(error_msg)
            state.set_error(error_msg)
            raise
        else:
            execution_time = time.time() - start_time
            self.logger.info(f"Completed {self.config.node_name} in {execution_time:.2f}s")
            state.add_message(f"Completed {self.config.node_name} in {execution_time:.2f}s")

    async def execute_with_retry(self, state: AgentState) -> NodeResult:
        """
        Execute the node with retry logic.
        """
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                with self._execution_context(state):
                    result = await self.execute(state)
                    return result
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {self.config.node_name}, retrying..."
                    )
                    state.add_message(f"Retrying {self.config.node_name} (attempt {attempt + 2})")
                else:
                    self.logger.error(
                        f"All {self.config.max_retries + 1} attempts failed for {self.config.node_name}"
                    )

        # If we get here, all retries failed
        error_msg = f"Node {self.config.node_name} failed after {self.config.max_retries + 1} attempts: {str(last_error)}"
        return NodeResult.error_result(state, error_msg)

    def validate_input_state(self, state: AgentState, required_fields: list) -> bool:
        """
        Validate that the state contains required fields.
        """
        missing_fields = []
        for field in required_fields:
            if not hasattr(state, field) or getattr(state, field) is None:
                missing_fields.append(field)

        if missing_fields:
            error_msg = f"{self.config.node_name} missing required fields: {missing_fields}"
            self.logger.error(error_msg)
            return False

        return True

    def update_state_safely(self, state: AgentState, updates: Dict[str, Any]) -> AgentState:
        """
        Safely update state with new values.
        """
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
                self.logger.debug(f"Updated state.{key}")
            else:
                self.logger.warning(f"Attempted to set unknown state field: {key}")

        return state

    def get_node_name(self) -> str:
        """Get the name of this node."""
        return self.config.node_name

    def log_state_summary(self, state: AgentState):
        """Log a summary of the current state."""
        summary = state.get_context_summary()
        self.logger.info(f"State summary: {summary}")