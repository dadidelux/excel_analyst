"""
Analysis Node for the Excel Agent LangGraph workflow.
Handles data analysis execution based on parsed queries.
"""

import os
import sys
from typing import Optional, Dict, Any

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.data_analyzer import DataAnalyzer
from .base_node import BaseAgentNode
from ..state import AgentState, NodeResult, NodeConfig, WorkflowStatus


class AnalysisNode(BaseAgentNode):
    """
    Agent node responsible for executing data analysis based on parsed queries.
    Uses the DataAnalyzer component to perform statistical analysis and computations.
    """

    def __init__(self, config: Optional[NodeConfig] = None):
        if config is None:
            config = NodeConfig(node_name="analysis")
        super().__init__(config)
        self.data_analyzer = DataAnalyzer()

    async def execute(self, state: AgentState) -> NodeResult:
        """
        Execute data analysis based on the parsed query.
        """
        # Validate input
        required_fields = ["parsed_query", "analysis_type"]
        if not self.validate_input_state(state, required_fields):
            return NodeResult.error_result(
                state, "Parsed query and analysis type are required"
            )

        # Get the processed data as DataFrame
        processed_data = state.get_dataframe("processed_data")
        if processed_data is None:
            return NodeResult.error_result(
                state, "No processed data available"
            )

        try:
            # Update status
            state.status = WorkflowStatus.ANALYZING_DATA
            analysis_type = state.analysis_type
            state.add_message(f"Executing {analysis_type} analysis")

            # Prepare analysis parameters
            analysis_params = self._prepare_analysis_params(state)

            # Create a QueryIntent from the parsed query for the DataAnalyzer
            from src.components.query_parser import QueryIntent, QueryType

            # Map analysis type to QueryType
            query_type_mapping = {
                "summary": "summary_stats",
                "summary_stats": "summary_stats",
                "correlation": "correlation",
                "comparison": "comparison",
                "trend_analysis": "trends",
                "aggregation": "aggregation",
                "distribution": "distribution",
                "outlier_detection": "outliers",
                "grouping": "aggregation"
            }

            query_type_str = query_type_mapping.get(analysis_type, analysis_type)

            # Create QueryIntent
            try:
                query_type_enum = QueryType(query_type_str)
            except ValueError:
                # Fallback to a basic type
                query_type_enum = QueryType.SUMMARY_STATS

            intent = QueryIntent(
                query_type=query_type_enum,
                primary_columns=analysis_params.get("columns", []),
                secondary_columns=[],
                aggregation_type=None,
                comparison_type=None,
                filters=analysis_params.get("filters"),
                time_column=analysis_params.get("time_column"),
                group_by_columns=analysis_params.get("group_by", []),
                confidence=0.8
            )

            # Set data and execute analysis
            self.data_analyzer.set_data(processed_data)
            analysis_result = self.data_analyzer.analyze(intent)

            # Convert AnalysisResult to dict format
            analysis_result_dict = {
                "success": True,
                "results": analysis_result.data,
                "insights": analysis_result.insights,
                "visualizations": analysis_result.visualizations,
                "execution_time": 0.1  # Placeholder
            }

            if not analysis_result_dict.get("success", False):
                return NodeResult.error_result(
                    state, f"Analysis failed: {analysis_result_dict.get('error', 'Unknown error')}"
                )

            # Update state with analysis results
            updates = {
                "analysis_results": analysis_result_dict,
                "analysis_metadata": {
                    "analysis_type": analysis_type,
                    "columns_analyzed": analysis_params.get("columns", []),
                    "parameters": analysis_params,
                    "execution_time": analysis_result_dict.get("execution_time"),
                    "data_points": len(processed_data)
                },
                "status": WorkflowStatus.ANALYSIS_COMPLETE
            }

            state = self.update_state_safely(state, updates)

            # Add detailed logging about results
            self._log_analysis_results(state, analysis_result_dict)

            self.log_state_summary(state)

            return NodeResult.success_result(state, next_node="visualization")

        except Exception as e:
            error_msg = f"Unexpected error during analysis: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return NodeResult.error_result(state, error_msg)


    def _prepare_analysis_params(self, state: AgentState) -> Dict[str, Any]:
        """
        Prepare parameters for the analysis based on the parsed query.
        """
        parsed_query = state.parsed_query
        params = {}

        # Extract columns
        if parsed_query.get("columns"):
            params["columns"] = parsed_query["columns"]

        # Extract grouping information
        if parsed_query.get("group_by"):
            params["group_by"] = parsed_query["group_by"]

        # Extract aggregation method
        if parsed_query.get("aggregation"):
            params["method"] = parsed_query["aggregation"]

        # Extract filters
        if parsed_query.get("filters"):
            params["filters"] = parsed_query["filters"]

        # Analysis-specific parameters
        analysis_type = state.analysis_type

        if analysis_type == "correlation":
            params["method"] = parsed_query.get("correlation_method", "pearson")

        elif analysis_type == "trend_analysis":
            params["time_column"] = parsed_query.get("time_column")

        elif analysis_type == "comparison":
            params["comparison_method"] = parsed_query.get("comparison_method", "statistical")

        elif analysis_type == "outlier_detection":
            params["method"] = parsed_query.get("outlier_method", "iqr")
            params["threshold"] = parsed_query.get("threshold", 1.5)

        elif analysis_type == "distribution":
            params["bins"] = parsed_query.get("bins", 30)

        return params

    def _log_analysis_results(self, state: AgentState, analysis_result: Dict[str, Any]):
        """
        Log key findings from the analysis results.
        """
        results = analysis_result.get("results", {})
        analysis_type = state.analysis_type

        if analysis_type == "summary":
            if "statistics" in results:
                stats = results["statistics"]
                state.add_message(f"Summary complete: {len(stats)} columns analyzed")

        elif analysis_type == "correlation":
            if "correlation_matrix" in results:
                # Find strongest correlations
                matrix = results["correlation_matrix"]
                strong_correlations = []
                for i in range(len(matrix.columns)):
                    for j in range(i + 1, len(matrix.columns)):
                        corr_value = matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            strong_correlations.append(
                                f"{matrix.columns[i]} - {matrix.columns[j]}: {corr_value:.3f}"
                            )
                if strong_correlations:
                    state.add_message(f"Strong correlations found: {'; '.join(strong_correlations[:3])}")

        elif analysis_type == "trend_analysis":
            if "trend_metrics" in results:
                metrics = results["trend_metrics"]
                for column, trend_data in metrics.items():
                    if "slope" in trend_data:
                        direction = "increasing" if trend_data["slope"] > 0 else "decreasing"
                        state.add_message(f"{column} trend: {direction} (slope: {trend_data['slope']:.4f})")

        elif analysis_type == "comparison":
            if "comparison_results" in results:
                comp_results = results["comparison_results"]
                for comparison in comp_results:
                    if "p_value" in comparison:
                        significance = "significant" if comparison["p_value"] < 0.05 else "not significant"
                        state.add_message(f"Comparison {significance} (p={comparison['p_value']:.4f})")

        elif analysis_type == "aggregation":
            if "aggregated_data" in results:
                agg_data = results["aggregated_data"]
                state.add_message(f"Aggregation complete: {len(agg_data)} groups created")

        elif analysis_type == "outlier_detection":
            if "outliers" in results:
                outlier_count = len(results["outliers"])
                total_points = len(state.processed_data)
                percentage = (outlier_count / total_points) * 100
                state.add_message(f"Outliers detected: {outlier_count} ({percentage:.1f}% of data)")

    def get_analysis_summary(self, state: AgentState) -> str:
        """
        Generate a human-readable summary of the analysis results.
        """
        if not state.analysis_results:
            return "No analysis results available."

        results = state.analysis_results.get("results", {})
        analysis_type = state.analysis_type
        metadata = state.analysis_metadata or {}

        summary = f"Analysis Type: {analysis_type.replace('_', ' ').title()}\n"
        summary += f"Data Points: {metadata.get('data_points', 'unknown')}\n"

        if metadata.get("columns_analyzed"):
            summary += f"Columns: {', '.join(metadata['columns_analyzed'])}\n"

        # Add specific results based on analysis type
        if analysis_type == "summary" and "statistics" in results:
            stats = results["statistics"]
            summary += f"Statistics calculated for {len(stats)} columns\n"

        elif analysis_type == "correlation" and "correlation_matrix" in results:
            matrix = results["correlation_matrix"]
            summary += f"Correlation matrix: {matrix.shape[0]}x{matrix.shape[1]}\n"

        elif analysis_type == "trend_analysis" and "trend_metrics" in results:
            trends = results["trend_metrics"]
            summary += f"Trend analysis for {len(trends)} columns\n"

        elif analysis_type == "aggregation" and "aggregated_data" in results:
            agg_data = results["aggregated_data"]
            summary += f"Aggregated into {len(agg_data)} groups\n"

        if state.analysis_results.get("execution_time"):
            summary += f"Execution time: {state.analysis_results['execution_time']:.3f}s\n"

        return summary

    def validate_analysis_requirements(self, state: AgentState) -> Dict[str, Any]:
        """
        Validate that the data and query are suitable for the requested analysis.
        """
        data = state.get_dataframe("processed_data")
        parsed_query = state.parsed_query
        analysis_type = state.analysis_type

        validation = {"valid": True, "warnings": [], "errors": []}

        # Check data availability
        if data is None:
            validation["errors"].append("No data available for analysis")
            validation["valid"] = False
            return validation

        # Check data size
        if len(data) < 2:
            validation["errors"].append("Insufficient data points for analysis")
            validation["valid"] = False

        # Check analysis-specific requirements
        if analysis_type == "correlation":
            numeric_cols = data.select_dtypes(include=['number']).columns
            query_cols = parsed_query.get("columns", [])
            non_numeric = [col for col in query_cols if col not in numeric_cols]
            if non_numeric:
                validation["errors"].append(f"Correlation requires numeric columns: {non_numeric}")
                validation["valid"] = False

        elif analysis_type == "trend_analysis":
            if "time_column" not in parsed_query:
                validation["warnings"].append("No time column specified for trend analysis")

        elif analysis_type == "grouping":
            if "group_by" not in parsed_query:
                validation["errors"].append("Grouping analysis requires group_by column")
                validation["valid"] = False

        return validation