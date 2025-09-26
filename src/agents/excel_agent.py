"""
Main Excel Agent interface that wraps the LangGraph workflow.
Provides a simple API for executing CSV analysis with natural language queries.
"""

import os
import sys
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .workflow import ExcelAgentWorkflow
from .state import AgentState, WorkflowStatus


class ExcelAgent:
    """
    Main interface for the Excel Agent with LangGraph workflow.
    Provides a simple API for analyzing CSV files and generating Excel reports.
    """

    def __init__(self, output_directory: str = "outputs"):
        """
        Initialize the Excel Agent.

        Args:
            output_directory: Directory for output files
        """
        self.output_directory = output_directory
        self.workflow = ExcelAgentWorkflow()
        self.last_execution_state: Optional[AgentState] = None

        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)

    async def analyze_csv(self, csv_file_path: str, user_query: str) -> Dict[str, Any]:
        """
        Analyze a CSV file with a natural language query.

        Args:
            csv_file_path: Path to the CSV file
            user_query: Natural language query about the data

        Returns:
            Dictionary with analysis results and metadata
        """
        # Validate inputs
        if not os.path.exists(csv_file_path):
            return {
                "success": False,
                "error": f"CSV file not found: {csv_file_path}",
                "output_path": None
            }

        if not user_query.strip():
            return {
                "success": False,
                "error": "User query cannot be empty",
                "output_path": None
            }

        try:
            # Execute the workflow
            result_state = await self.workflow.execute_workflow(
                csv_file_path=csv_file_path,
                user_query=user_query,
                output_directory=self.output_directory
            )

            self.last_execution_state = result_state

            # Format results
            return self._format_results(result_state)

        except Exception as e:
            return {
                "success": False,
                "error": f"Workflow execution failed: {str(e)}",
                "output_path": None
            }

    def analyze_csv_sync(self, csv_file_path: str, user_query: str) -> Dict[str, Any]:
        """
        Synchronous version of analyze_csv.
        """
        try:
            # Create event loop if it doesn't exist
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, create a new loop
                    import threading
                    result = None
                    exception = None

                    def run_in_thread():
                        nonlocal result, exception
                        try:
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            result = new_loop.run_until_complete(
                                self.analyze_csv(csv_file_path, user_query)
                            )
                        except Exception as e:
                            exception = e
                        finally:
                            new_loop.close()

                    thread = threading.Thread(target=run_in_thread)
                    thread.start()
                    thread.join()

                    if exception:
                        raise exception
                    return result
                else:
                    return loop.run_until_complete(self.analyze_csv(csv_file_path, user_query))
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(self.analyze_csv(csv_file_path, user_query))

        except Exception as e:
            return {
                "success": False,
                "error": f"Synchronous execution failed: {str(e)}",
                "output_path": None
            }

    def _format_results(self, state: AgentState) -> Dict[str, Any]:
        """
        Format the workflow results into a user-friendly dictionary.
        """
        if state.status == WorkflowStatus.ERROR:
            return {
                "success": False,
                "error": state.error_message or "Unknown error occurred",
                "output_path": None,
                "messages": state.messages
            }

        elif state.status == WorkflowStatus.COMPLETED:
            return {
                "success": True,
                "output_path": state.excel_output_path,
                "query": state.user_query,
                "analysis_type": state.analysis_type,
                "confidence": state.query_confidence,
                "data_summary": self._format_data_summary(state),
                "analysis_summary": self._format_analysis_summary(state),
                "charts_generated": len(state.charts) if state.charts else 0,
                "export_metadata": state.export_metadata,
                "messages": state.messages,
                "processing_time": self._calculate_processing_time(state)
            }

        else:
            return {
                "success": False,
                "error": f"Workflow incomplete (status: {state.status.value})",
                "output_path": None,
                "messages": state.messages
            }

    def _format_data_summary(self, state: AgentState) -> Dict[str, Any]:
        """Format data summary for user display."""
        if not state.data_summary:
            return {}

        return {
            "file_path": state.data_summary.get("file_path"),
            "rows": state.data_summary.get("rows"),
            "columns": state.data_summary.get("columns"),
            "column_names": state.data_summary.get("column_names", [])[:10],  # Limit for display
            "data_types": {
                col: str(dtype) for col, dtype in state.data_summary.get("data_types", {}).items()
            },
            "memory_usage_mb": round(state.data_summary.get("memory_usage", 0) / 1024 / 1024, 2)
        }

    def _format_analysis_summary(self, state: AgentState) -> Dict[str, Any]:
        """Format analysis summary for user display."""
        if not state.analysis_results:
            return {}

        summary = {
            "analysis_type": state.analysis_type,
            "success": state.analysis_results.get("success", False)
        }

        results = state.analysis_results.get("results", {})

        # Add type-specific summaries
        if state.analysis_type == "correlation" and "correlation_matrix" in results:
            matrix = results["correlation_matrix"]
            summary["correlation_info"] = {
                "variables": list(matrix.columns),
                "strongest_correlation": self._find_strongest_correlation(matrix)
            }

        elif state.analysis_type == "trend_analysis" and "trend_metrics" in results:
            trends = results["trend_metrics"]
            summary["trend_info"] = {
                "variables_analyzed": list(trends.keys()),
                "trends_found": len([t for t in trends.values() if abs(t.get("slope", 0)) > 0.01])
            }

        elif state.analysis_type == "summary" and "statistics" in results:
            stats = results["statistics"]
            summary["summary_info"] = {
                "variables_analyzed": list(stats.keys()),
                "numeric_variables": len(stats)
            }

        return summary

    def _find_strongest_correlation(self, correlation_matrix) -> Dict[str, Any]:
        """Find the strongest correlation in the matrix."""
        try:
            import numpy as np

            # Get upper triangle to avoid duplicates
            upper_tri = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )

            # Find maximum absolute correlation
            max_corr = 0
            max_pair = None

            for col1 in upper_tri.columns:
                for col2 in upper_tri.columns:
                    if not np.isnan(upper_tri.loc[col1, col2]):
                        corr_value = abs(upper_tri.loc[col1, col2])
                        if corr_value > max_corr:
                            max_corr = corr_value
                            max_pair = (col1, col2, upper_tri.loc[col1, col2])

            if max_pair:
                return {
                    "variables": [max_pair[0], max_pair[1]],
                    "correlation": round(max_pair[2], 3),
                    "strength": "strong" if abs(max_pair[2]) > 0.7 else "moderate" if abs(max_pair[2]) > 0.3 else "weak"
                }

        except Exception:
            pass

        return {"error": "Could not calculate strongest correlation"}

    def _calculate_processing_time(self, state: AgentState) -> float:
        """Calculate total processing time."""
        # Simple estimation based on workflow completion
        if state.export_metadata and "export_time" in state.export_metadata:
            return 5.0  # Rough estimate
        return 0.0

    def get_last_execution_summary(self) -> Optional[str]:
        """
        Get a human-readable summary of the last execution.
        """
        if not self.last_execution_state:
            return None

        state = self.last_execution_state
        summary_lines = []

        summary_lines.append("=== Excel Agent Execution Summary ===")
        summary_lines.append(f"Query: {state.user_query}")
        summary_lines.append(f"Status: {state.status.value}")

        if state.data_summary:
            summary_lines.append(f"Data: {state.data_summary['rows']} rows, {state.data_summary['columns']} columns")

        if state.analysis_type:
            summary_lines.append(f"Analysis: {state.analysis_type}")

        if state.query_confidence:
            summary_lines.append(f"Query Confidence: {state.query_confidence:.1%}")

        if state.charts:
            summary_lines.append(f"Charts Generated: {len(state.charts)}")

        if state.excel_output_path:
            summary_lines.append(f"Output: {state.excel_output_path}")

        if state.error_message:
            summary_lines.append(f"Error: {state.error_message}")

        summary_lines.append(f"Messages: {len(state.messages)} workflow steps")

        return "\n".join(summary_lines)

    async def validate_setup(self) -> Dict[str, Any]:
        """
        Validate that the agent is properly set up.
        """
        validation = await self.workflow.validate_workflow()

        # Add agent-specific validation
        validation["output_directory"] = {
            "path": self.output_directory,
            "exists": os.path.exists(self.output_directory),
            "writable": os.access(self.output_directory, os.W_OK) if os.path.exists(self.output_directory) else False
        }

        return validation

    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the workflow configuration.
        """
        return self.workflow.get_workflow_summary()

    def suggest_queries(self, csv_file_path: str) -> List[str]:
        """
        Suggest possible queries based on CSV file analysis.
        """
        if not os.path.exists(csv_file_path):
            return ["File not found"]

        try:
            # Quick data preview to generate suggestions
            import pandas as pd
            data = pd.read_csv(csv_file_path, nrows=100)  # Sample first 100 rows

            suggestions = []
            numeric_cols = list(data.select_dtypes(include=['number']).columns)
            categorical_cols = list(data.select_dtypes(include=['object']).columns)

            if numeric_cols:
                suggestions.append(f"Show summary statistics for {numeric_cols[0]}")
                if len(numeric_cols) > 1:
                    suggestions.append(f"Analyze correlation between {numeric_cols[0]} and {numeric_cols[1]}")

            if categorical_cols and numeric_cols:
                suggestions.append(f"Compare {numeric_cols[0]} across different {categorical_cols[0]}")

            if len(numeric_cols) >= 2:
                suggestions.append(f"Show trends in {numeric_cols[0]} and {numeric_cols[1]}")

            if not suggestions:
                suggestions.append("Analyze this data")

            return suggestions[:5]  # Limit to 5 suggestions

        except Exception as e:
            return [f"Error reading file: {str(e)}"]