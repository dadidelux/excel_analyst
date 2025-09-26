"""
State management for the Excel Agent LangGraph workflow.
Defines the shared state structure that flows between agent nodes.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from pydantic import BaseModel


class WorkflowStatus(Enum):
    """Status of the workflow execution."""
    INITIALIZED = "initialized"
    LOADING_DATA = "loading_data"
    DATA_LOADED = "data_loaded"
    PARSING_QUERY = "parsing_query"
    QUERY_PARSED = "query_parsed"
    ANALYZING_DATA = "analyzing_data"
    ANALYSIS_COMPLETE = "analysis_complete"
    GENERATING_CHARTS = "generating_charts"
    CHARTS_GENERATED = "charts_generated"
    EXPORTING_EXCEL = "exporting_excel"
    COMPLETED = "completed"
    ERROR = "error"


class AgentState(BaseModel):
    """
    Central state object that flows through the LangGraph workflow.
    Contains all data and metadata needed by different agent nodes.
    """
    # Input parameters
    csv_file_path: Optional[str] = None
    user_query: Optional[str] = None
    output_directory: str = "outputs"

    # Data processing state (stored as serializable formats)
    raw_data: Optional[Dict[str, Any]] = None  # Store as dict with 'data' and 'columns'
    processed_data: Optional[Dict[str, Any]] = None
    data_summary: Optional[Dict[str, Any]] = None

    # Query processing state
    parsed_query: Optional[Dict[str, Any]] = None
    query_confidence: Optional[float] = None
    query_suggestions: List[str] = field(default_factory=list)

    # Analysis state
    analysis_results: Optional[Dict[str, Any]] = None
    analysis_type: Optional[str] = None
    analysis_metadata: Optional[Dict[str, Any]] = None

    # Visualization state
    charts: List[Dict[str, Any]] = field(default_factory=list)
    chart_paths: List[str] = field(default_factory=list)

    # Export state
    excel_output_path: Optional[str] = None
    export_metadata: Optional[Dict[str, Any]] = None

    # Workflow control
    status: WorkflowStatus = WorkflowStatus.INITIALIZED
    error_message: Optional[str] = None
    messages: List[str] = field(default_factory=list)
    next_action: Optional[str] = None

    # Conversation memory
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def set_dataframe(self, df: pd.DataFrame, attr_name: str):
        """Set a DataFrame as serializable data."""
        import json

        # Convert DataFrame to records with Python native types
        records = []
        for record in df.to_dict('records'):
            clean_record = {}
            for key, value in record.items():
                # Convert numpy types to Python types
                if hasattr(value, 'item'):  # numpy scalar
                    clean_record[key] = value.item()
                elif pd.isna(value):
                    clean_record[key] = None
                else:
                    clean_record[key] = value
            records.append(clean_record)

        serializable_data = {
            'data': records,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            'shape': tuple(int(x) for x in df.shape)  # Convert numpy.int64 to int
        }
        setattr(self, attr_name, serializable_data)

    def get_dataframe(self, attr_name: str) -> Optional[pd.DataFrame]:
        """Get a DataFrame from serializable data."""
        data_dict = getattr(self, attr_name)
        if data_dict is None:
            return None

        df = pd.DataFrame(data_dict['data'])

        # Restore data types
        for col, dtype_str in data_dict.get('dtypes', {}).items():
            if col in df.columns:
                try:
                    if 'int' in dtype_str:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    elif 'float' in dtype_str:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif 'datetime' in dtype_str:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass  # Keep as object if conversion fails

        return df

    def add_message(self, message: str, level: str = "info"):
        """Add a message to the workflow log."""
        self.messages.append(f"[{level.upper()}] {message}")

    def set_error(self, error_message: str):
        """Set error state and message."""
        self.status = WorkflowStatus.ERROR
        self.error_message = error_message
        self.add_message(error_message, "error")

    def add_conversation_turn(self, user_input: str, agent_response: str):
        """Add a conversation turn to memory."""
        self.conversation_history.append({
            "user": user_input,
            "agent": agent_response,
            "timestamp": pd.Timestamp.now().isoformat()
        })

    def get_context_summary(self) -> str:
        """Get a summary of current context for LLM."""
        summary = []

        if self.raw_data is not None:
            shape = self.raw_data.get('shape', (0, 0))
            summary.append(f"Data loaded: {shape[0]} rows, {shape[1]} columns")

        if self.parsed_query:
            summary.append(f"Query type: {self.parsed_query.get('intent', 'unknown')}")

        if self.analysis_results:
            summary.append(f"Analysis completed: {self.analysis_type}")

        if self.charts:
            summary.append(f"Charts generated: {len(self.charts)}")

        return "; ".join(summary) if summary else "No context available"


@dataclass
class NodeResult:
    """Result returned by agent nodes."""
    success: bool
    updated_state: AgentState
    next_node: Optional[str] = None
    error_message: Optional[str] = None

    @classmethod
    def success_result(cls, state: AgentState, next_node: Optional[str] = None):
        """Create a successful result."""
        return cls(success=True, updated_state=state, next_node=next_node)

    @classmethod
    def error_result(cls, state: AgentState, error_message: str):
        """Create an error result."""
        state.set_error(error_message)
        return cls(success=False, updated_state=state, error_message=error_message)


class NodeConfig(BaseModel):
    """Configuration for agent nodes."""
    node_name: str
    max_retries: int = 2
    timeout_seconds: int = 30
    enable_logging: bool = True

    def __init__(self, **data):
        super().__init__(**data)


# Workflow routing conditions
def should_continue_to_analysis(state: AgentState) -> bool:
    """Check if workflow should continue to analysis."""
    return (
        state.status == WorkflowStatus.QUERY_PARSED and
        state.parsed_query is not None and
        state.query_confidence and state.query_confidence > 0.5
    )


def should_generate_charts(state: AgentState) -> bool:
    """Check if workflow should generate charts."""
    return (
        state.status == WorkflowStatus.ANALYSIS_COMPLETE and
        state.analysis_results is not None
    )


def should_export_excel(state: AgentState) -> bool:
    """Check if workflow should export to Excel."""
    return (
        state.status == WorkflowStatus.CHARTS_GENERATED and
        len(state.charts) > 0
    )


def needs_query_clarification(state: AgentState) -> bool:
    """Check if query needs clarification."""
    return (
        state.parsed_query is not None and
        state.query_confidence is not None and
        state.query_confidence < 0.5
    )