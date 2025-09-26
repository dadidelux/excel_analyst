"""
CSV Ingestion Node for the Excel Agent LangGraph workflow.
Handles loading and processing CSV files with smart detection.
"""

import os
import sys
from typing import Optional

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.csv_loader import CSVLoader
from .base_node import BaseAgentNode
from ..state import AgentState, NodeResult, NodeConfig, WorkflowStatus


class CSVIngestionNode(BaseAgentNode):
    """
    Agent node responsible for loading and processing CSV files.
    Uses the CSVLoader component to handle encoding detection, validation, and cleaning.
    """

    def __init__(self, config: Optional[NodeConfig] = None):
        if config is None:
            config = NodeConfig(node_name="csv_ingestion")
        super().__init__(config)
        self.csv_loader = CSVLoader()

    async def execute(self, state: AgentState) -> NodeResult:
        """
        Execute CSV loading and processing.
        """
        # Validate input
        if not self.validate_input_state(state, ["csv_file_path"]):
            return NodeResult.error_result(state, "CSV file path is required")

        # Check if file exists
        if not os.path.exists(state.csv_file_path):
            return NodeResult.error_result(state, f"CSV file not found: {state.csv_file_path}")

        try:
            # Update status
            state.status = WorkflowStatus.LOADING_DATA
            state.add_message(f"Loading CSV file: {state.csv_file_path}")

            # Load the CSV file
            data = self.csv_loader.load_csv(state.csv_file_path)

            if data is None or data.empty:
                return NodeResult.error_result(state, "Failed to load CSV: Empty or invalid data")

            # Store data using serializable format
            state.set_dataframe(data, "raw_data")
            state.set_dataframe(data, "processed_data")  # Initially same as raw

            # Update other state fields
            updates = {
                "data_summary": {
                    "file_path": state.csv_file_path,
                    "rows": len(data),
                    "columns": len(data.columns),
                    "column_names": list(data.columns),
                    "data_types": {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()},
                    "memory_usage": data.memory_usage(deep=True).sum(),
                    "encoding": "utf-8",  # Default encoding
                    "delimiter": ",",
                    "has_header": True
                },
                "status": WorkflowStatus.DATA_LOADED
            }

            state = self.update_state_safely(state, updates)

            # Generate data preview for logging
            preview = self.csv_loader.get_preview(data)
            state.add_message(f"Data loaded successfully: {len(data)} rows, {len(data.columns)} columns")

            # Log column information
            columns_info = ", ".join([f"{col}({str(dtype)})" for col, dtype in
                                    data.dtypes.to_dict().items()])
            state.add_message(f"Columns: {columns_info}")

            # Check for data quality issues
            quality_issues = self._check_data_quality(data)
            if quality_issues:
                state.add_message(f"Data quality notes: {'; '.join(quality_issues)}")

            self.log_state_summary(state)

            return NodeResult.success_result(state, next_node="query_processing")

        except Exception as e:
            error_msg = f"Unexpected error during CSV ingestion: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return NodeResult.error_result(state, error_msg)

    def _check_data_quality(self, data) -> list:
        """
        Check for common data quality issues.
        """
        issues = []

        # Check for missing values
        missing_counts = data.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0]
        if len(columns_with_missing) > 0:
            total_missing = missing_counts.sum()
            issues.append(f"{total_missing} missing values across {len(columns_with_missing)} columns")

        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"{duplicate_count} duplicate rows found")

        # Check for constant columns
        constant_columns = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_columns.append(col)
        if constant_columns:
            issues.append(f"Constant columns: {', '.join(constant_columns)}")

        # Check for very high cardinality columns (potential IDs)
        high_cardinality_columns = []
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].nunique() / len(data) > 0.9:
                high_cardinality_columns.append(col)
        if high_cardinality_columns:
            issues.append(f"High cardinality columns: {', '.join(high_cardinality_columns)}")

        return issues

    def get_data_summary(self, state: AgentState) -> dict:
        """
        Get a comprehensive summary of the loaded data.
        """
        if state.raw_data is None:
            return {"error": "No data loaded"}

        data = state.raw_data

        summary = {
            "basic_info": {
                "rows": len(data),
                "columns": len(data.columns),
                "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024
            },
            "column_types": {
                "numeric": len(data.select_dtypes(include=['number']).columns),
                "text": len(data.select_dtypes(include=['object']).columns),
                "datetime": len(data.select_dtypes(include=['datetime']).columns),
                "boolean": len(data.select_dtypes(include=['bool']).columns)
            },
            "data_quality": {
                "missing_values": data.isnull().sum().sum(),
                "duplicate_rows": data.duplicated().sum(),
                "complete_rows": len(data.dropna())
            }
        }

        # Add column-specific information
        summary["columns"] = {}
        for col in data.columns:
            col_info = {
                "type": str(data[col].dtype),
                "non_null_count": data[col].count(),
                "unique_values": data[col].nunique()
            }

            if data[col].dtype in ['int64', 'float64']:
                col_info.update({
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "mean": data[col].mean()
                })
            elif data[col].dtype == 'object':
                col_info["sample_values"] = data[col].dropna().head(3).tolist()

            summary["columns"][col] = col_info

        return summary