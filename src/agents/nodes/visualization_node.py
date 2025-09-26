"""
Visualization Node for the Excel Agent LangGraph workflow.
Handles chart generation based on analysis results.
"""

import os
import sys
from typing import Optional, List, Dict, Any

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.chart_generator import ChartGenerator
from .base_node import BaseAgentNode
from ..state import AgentState, NodeResult, NodeConfig, WorkflowStatus


class VisualizationNode(BaseAgentNode):
    """
    Agent node responsible for generating visualizations based on analysis results.
    Uses the ChartGenerator component to create appropriate charts for the data.
    """

    def __init__(self, config: Optional[NodeConfig] = None):
        if config is None:
            config = NodeConfig(node_name="visualization")
        super().__init__(config)
        self.chart_generator = ChartGenerator()

    async def execute(self, state: AgentState) -> NodeResult:
        """
        Execute visualization generation based on analysis results.
        """
        # Validate input
        required_fields = ["analysis_results", "analysis_type"]
        if not self.validate_input_state(state, required_fields):
            return NodeResult.error_result(
                state, "Analysis results and analysis type are required"
            )

        # Get the processed data as DataFrame
        processed_data = state.get_dataframe("processed_data")
        if processed_data is None:
            return NodeResult.error_result(
                state, "No processed data available"
            )

        try:
            # Update status
            state.status = WorkflowStatus.GENERATING_CHARTS
            state.add_message(f"Generating visualizations for {state.analysis_type} analysis")

            # Determine appropriate chart types
            chart_specs = self._determine_chart_types(state, processed_data)

            if not chart_specs:
                state.add_message("No suitable visualizations for this analysis type")
                # Still continue to export, just without charts
                updates = {
                    "status": WorkflowStatus.CHARTS_GENERATED,
                    "charts": [],
                    "chart_paths": []
                }
                state = self.update_state_safely(state, updates)
                return NodeResult.success_result(state, next_node="excel_export")

            # Generate charts
            generated_charts = []
            chart_paths = []

            for chart_spec in chart_specs:
                try:
                    chart_result = self._generate_single_chart(state, chart_spec)
                    if chart_result["success"]:
                        generated_charts.append(chart_result["chart_data"])
                        chart_paths.append(chart_result["file_path"])
                        state.add_message(f"Generated {chart_spec['type']} chart: {chart_spec['title']}")
                    else:
                        state.add_message(f"Failed to generate {chart_spec['type']} chart: {chart_result['error']}")
                except Exception as e:
                    self.logger.warning(f"Error generating chart {chart_spec['type']}: {str(e)}")
                    state.add_message(f"Error generating {chart_spec['type']} chart")

            # Update state with generated charts
            updates = {
                "charts": generated_charts,
                "chart_paths": chart_paths,
                "status": WorkflowStatus.CHARTS_GENERATED
            }

            state = self.update_state_safely(state, updates)

            state.add_message(f"Generated {len(generated_charts)} charts successfully")

            self.log_state_summary(state)

            return NodeResult.success_result(state, next_node="excel_export")

        except Exception as e:
            error_msg = f"Unexpected error during visualization generation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return NodeResult.error_result(state, error_msg)

    def _determine_chart_types(self, state: AgentState, data) -> List[Dict[str, Any]]:
        """
        Determine appropriate chart types based on analysis type and results.
        """
        analysis_type = state.analysis_type
        analysis_results = state.analysis_results
        parsed_query = state.parsed_query

        chart_specs = []

        if analysis_type == "summary":
            # Generate histograms for numeric columns
            columns = parsed_query.get("columns", [])
            numeric_columns = [col for col in columns if col in data.select_dtypes(include=['number']).columns]

            for col in numeric_columns[:3]:  # Limit to 3 charts
                chart_specs.append({
                    "type": "histogram",
                    "title": f"Distribution of {col}",
                    "data": data[col],
                    "column": col
                })

        elif analysis_type == "correlation":
            # Generate correlation heatmap
            if "correlation_matrix" in analysis_results.get("results", {}):
                chart_specs.append({
                    "type": "heatmap",
                    "title": "Correlation Matrix",
                    "data": analysis_results["results"]["correlation_matrix"]
                })

            # Generate scatter plots for strong correlations
            columns = parsed_query.get("columns", [])
            if len(columns) >= 2:
                chart_specs.append({
                    "type": "scatter",
                    "title": f"Scatter Plot: {columns[0]} vs {columns[1]}",
                    "data": data,
                    "x_column": columns[0],
                    "y_column": columns[1]
                })

        elif analysis_type == "comparison":
            columns = parsed_query.get("columns", [])
            if len(columns) >= 2:
                # Box plot for comparison
                chart_specs.append({
                    "type": "box",
                    "title": f"Comparison: {' vs '.join(columns)}",
                    "data": data,
                    "columns": columns
                })

                # Bar chart if there's a categorical column
                categorical_cols = [col for col in columns if col in data.select_dtypes(include=['object']).columns]
                numeric_cols = [col for col in columns if col in data.select_dtypes(include=['number']).columns]

                if categorical_cols and numeric_cols:
                    chart_specs.append({
                        "type": "bar",
                        "title": f"{numeric_cols[0]} by {categorical_cols[0]}",
                        "data": data,
                        "x_column": categorical_cols[0],
                        "y_column": numeric_cols[0]
                    })

        elif analysis_type == "trend_analysis":
            if "trend_metrics" in analysis_results.get("results", {}):
                trend_data = analysis_results["results"]["trend_metrics"]
                time_column = parsed_query.get("time_column")

                for column, metrics in trend_data.items():
                    if column != time_column:  # Don't plot time against itself
                        chart_specs.append({
                            "type": "line",
                            "title": f"Trend: {column} over time",
                            "data": data,
                            "x_column": time_column,
                            "y_column": column
                        })

        elif analysis_type == "aggregation":
            if "aggregated_data" in analysis_results.get("results", {}):
                agg_data = analysis_results["results"]["aggregated_data"]
                group_by = parsed_query.get("group_by")

                # Bar chart for aggregated data
                if group_by and len(agg_data.columns) > 1:
                    value_col = [col for col in agg_data.columns if col != group_by][0]
                    chart_specs.append({
                        "type": "bar",
                        "title": f"Aggregated {value_col} by {group_by}",
                        "data": agg_data,
                        "x_column": group_by,
                        "y_column": value_col
                    })

        elif analysis_type == "distribution":
            columns = parsed_query.get("columns", [])
            for col in columns:
                if col in data.select_dtypes(include=['number']).columns:
                    chart_specs.append({
                        "type": "histogram",
                        "title": f"Distribution of {col}",
                        "data": data[col],
                        "column": col
                    })

        elif analysis_type == "outlier_detection":
            if "outliers" in analysis_results.get("results", {}):
                columns = parsed_query.get("columns", [])
                for col in columns:
                    if col in data.select_dtypes(include=['number']).columns:
                        chart_specs.append({
                            "type": "box",
                            "title": f"Outlier Detection: {col}",
                            "data": data[col],
                            "column": col
                        })

        return chart_specs

    def _generate_single_chart(self, state: AgentState, chart_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a single chart based on the specification.
        """
        chart_type = chart_spec["type"]
        title = chart_spec["title"]
        data = chart_spec["data"]

        # Create output directory if it doesn't exist
        output_dir = os.path.join(state.output_directory, "charts")
        os.makedirs(output_dir, exist_ok=True)

        try:
            if chart_type == "histogram":
                result = self.chart_generator.create_histogram(
                    data=data,
                    column=chart_spec.get("column"),
                    title=title,
                    output_dir=output_dir
                )

            elif chart_type == "scatter":
                result = self.chart_generator.create_scatter_plot(
                    data=data,
                    x_column=chart_spec["x_column"],
                    y_column=chart_spec["y_column"],
                    title=title,
                    output_dir=output_dir
                )

            elif chart_type == "line":
                result = self.chart_generator.create_line_chart(
                    data=data,
                    x_column=chart_spec["x_column"],
                    y_column=chart_spec["y_column"],
                    title=title,
                    output_dir=output_dir
                )

            elif chart_type == "bar":
                result = self.chart_generator.create_bar_chart(
                    data=data,
                    x_column=chart_spec["x_column"],
                    y_column=chart_spec["y_column"],
                    title=title,
                    output_dir=output_dir
                )

            elif chart_type == "box":
                if "columns" in chart_spec:
                    result = self.chart_generator.create_box_plot(
                        data=data,
                        columns=chart_spec["columns"],
                        title=title,
                        output_dir=output_dir
                    )
                else:
                    result = self.chart_generator.create_box_plot(
                        data=data,
                        columns=[chart_spec["column"]],
                        title=title,
                        output_dir=output_dir
                    )

            elif chart_type == "heatmap":
                result = self.chart_generator.create_heatmap(
                    data=data,
                    title=title,
                    output_dir=output_dir
                )

            else:
                return {
                    "success": False,
                    "error": f"Unsupported chart type: {chart_type}"
                }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Error generating {chart_type} chart: {str(e)}"
            }

    def get_chart_summary(self, state: AgentState) -> str:
        """
        Generate a summary of the generated charts.
        """
        if not state.charts:
            return "No charts generated."

        summary = f"Generated {len(state.charts)} charts:\n"

        for i, chart in enumerate(state.charts, 1):
            chart_type = chart.get("chart_type", "unknown")
            title = chart.get("title", f"Chart {i}")
            summary += f"{i}. {chart_type.title()}: {title}\n"

        return summary