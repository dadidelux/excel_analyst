"""
Query Processing Node for the Excel Agent LangGraph workflow.
Handles natural language query parsing and intent recognition.
"""

import os
import sys
from typing import Optional

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.query_parser import QueryParser
from .base_node import BaseAgentNode
from ..state import AgentState, NodeResult, NodeConfig, WorkflowStatus


class QueryProcessingNode(BaseAgentNode):
    """
    Agent node responsible for processing natural language queries.
    Uses the QueryParser component to understand user intent and extract parameters.
    """

    def __init__(self, config: Optional[NodeConfig] = None):
        if config is None:
            config = NodeConfig(node_name="query_processing")
        super().__init__(config)
        self.query_parser = QueryParser()

    async def execute(self, state: AgentState) -> NodeResult:
        """
        Execute query processing and intent recognition.
        """
        # Validate input
        if not self.validate_input_state(state, ["user_query"]):
            return NodeResult.error_result(
                state, "User query is required"
            )

        # Get the processed data as DataFrame
        processed_data = state.get_dataframe("processed_data")
        if processed_data is None:
            return NodeResult.error_result(
                state, "No processed data available"
            )

        try:
            # Update status
            state.status = WorkflowStatus.PARSING_QUERY
            state.add_message(f"Parsing query: '{state.user_query}'")

            # Parse the query
            parsed_intent = self.query_parser.parse_query(state.user_query)

            # Convert QueryIntent to dict format expected by the workflow
            parsed_query = {
                "intent": parsed_intent.query_type.value if hasattr(parsed_intent.query_type, 'value') else str(parsed_intent.query_type),
                "columns": parsed_intent.primary_columns + (parsed_intent.secondary_columns or []),
                "group_by": parsed_intent.group_by_columns,
                "aggregation": parsed_intent.aggregation_type.value if parsed_intent.aggregation_type and hasattr(parsed_intent.aggregation_type, 'value') else str(parsed_intent.aggregation_type) if parsed_intent.aggregation_type else None,
                "filters": parsed_intent.filters,
                "time_column": parsed_intent.time_column,
                "correlation_method": 'pearson',  # Default
                "comparison_method": parsed_intent.comparison_type.value if parsed_intent.comparison_type and hasattr(parsed_intent.comparison_type, 'value') else 'statistical'
            }

            # Calculate confidence based on intent completeness
            confidence = self._calculate_confidence(parsed_intent, processed_data)

            # Update state with parsed query information
            updates = {
                "parsed_query": parsed_query,
                "query_confidence": confidence,
                "analysis_type": parsed_query.get("intent"),
                "status": WorkflowStatus.QUERY_PARSED
            }

            state = self.update_state_safely(state, updates)

            # Add detailed logging
            state.add_message(f"Query intent: {parsed_query.get('intent', 'unknown')}")
            state.add_message(f"Confidence: {confidence:.2f}")

            if parsed_query.get("columns"):
                state.add_message(f"Target columns: {', '.join(parsed_query['columns'])}")

            # Generate suggestions if confidence is low
            if confidence < 0.7:
                suggestions = self._generate_query_suggestions(processed_data)
                state.query_suggestions = suggestions
                state.add_message(f"Low confidence query. Suggestions: {'; '.join(suggestions[:3])}")

            # Check if we have enough confidence to proceed
            if confidence < 0.3:
                return NodeResult.error_result(
                    state,
                    f"Query confidence too low ({confidence:.2f}). Please rephrase your query or try one of the suggestions."
                )

            # Validate that required columns exist in data
            validation_result = self._validate_query_against_data(parsed_query, processed_data)
            if not validation_result["valid"]:
                # Try to suggest corrections
                corrected_query = self._attempt_query_correction(parsed_query, processed_data)
                if corrected_query:
                    state.parsed_query = corrected_query
                    state.add_message("Auto-corrected column names in query")
                else:
                    return NodeResult.error_result(
                        state, f"Query validation failed: {validation_result['error']}"
                    )

            self.log_state_summary(state)

            # Determine next node based on confidence
            if confidence >= 0.7:
                return NodeResult.success_result(state, next_node="analysis")
            else:
                # Might want to route to a clarification node in the future
                return NodeResult.success_result(state, next_node="analysis")

        except Exception as e:
            error_msg = f"Unexpected error during query processing: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return NodeResult.error_result(state, error_msg)

    def _generate_query_suggestions(self, data) -> list:
        """
        Generate helpful query suggestions based on the data.
        """
        suggestions = []
        columns = list(data.columns)
        numeric_columns = list(data.select_dtypes(include=['number']).columns)
        categorical_columns = list(data.select_dtypes(include=['object']).columns)

        # Basic analysis suggestions
        if numeric_columns:
            suggestions.append(f"Show summary statistics for {numeric_columns[0]}")
            if len(numeric_columns) > 1:
                suggestions.append(f"Compare {numeric_columns[0]} and {numeric_columns[1]}")

        if categorical_columns and numeric_columns:
            suggestions.append(f"Group {numeric_columns[0]} by {categorical_columns[0]}")

        # Correlation suggestions
        if len(numeric_columns) >= 2:
            suggestions.append(f"Show correlation between {numeric_columns[0]} and {numeric_columns[1]}")

        # Trend analysis if date columns exist
        date_columns = list(data.select_dtypes(include=['datetime']).columns)
        if date_columns and numeric_columns:
            suggestions.append(f"Show trends in {numeric_columns[0]} over time")

        # Top/bottom analysis
        if categorical_columns and numeric_columns:
            suggestions.append(f"Show top {categorical_columns[0]} by {numeric_columns[0]}")

        return suggestions

    def _validate_query_against_data(self, parsed_query: dict, data) -> dict:
        """
        Validate that the parsed query can be executed against the data.
        """
        columns = list(data.columns)
        query_columns = parsed_query.get("columns", [])

        # Check if all referenced columns exist
        missing_columns = [col for col in query_columns if col not in columns]
        if missing_columns:
            return {
                "valid": False,
                "error": f"Columns not found in data: {missing_columns}",
                "missing_columns": missing_columns
            }

        # Check if the analysis type is appropriate for the data types
        intent = parsed_query.get("intent")
        if intent in ["correlation", "trend_analysis"] and len(query_columns) < 2:
            return {
                "valid": False,
                "error": f"{intent} requires at least 2 columns"
            }

        # Check for numeric operations on non-numeric columns
        numeric_intents = ["correlation", "aggregation", "comparison"]
        if intent in numeric_intents:
            numeric_columns = list(data.select_dtypes(include=['number']).columns)
            non_numeric_query_cols = [col for col in query_columns if col not in numeric_columns]
            if non_numeric_query_cols and intent == "correlation":
                return {
                    "valid": False,
                    "error": f"Correlation requires numeric columns. Non-numeric: {non_numeric_query_cols}"
                }

        return {"valid": True}

    def _attempt_query_correction(self, parsed_query: dict, data) -> Optional[dict]:
        """
        Attempt to automatically correct column names in the query.
        """
        columns = list(data.columns)
        query_columns = parsed_query.get("columns", [])
        corrected_columns = []

        for query_col in query_columns:
            if query_col not in columns:
                # Try fuzzy matching
                best_match = None
                best_score = 0

                for data_col in columns:
                    # Simple similarity scoring
                    score = self._calculate_similarity(query_col.lower(), data_col.lower())
                    if score > best_score and score > 0.6:  # Threshold for correction
                        best_score = score
                        best_match = data_col

                if best_match:
                    corrected_columns.append(best_match)
                    self.logger.info(f"Auto-corrected '{query_col}' to '{best_match}'")
                else:
                    return None  # Can't correct this column
            else:
                corrected_columns.append(query_col)

        # Return corrected query
        corrected_query = parsed_query.copy()
        corrected_query["columns"] = corrected_columns
        return corrected_query

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate simple similarity score between two strings.
        """
        # Simple Jaccard similarity on character level
        set1 = set(str1)
        set2 = set(str2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0

    def get_query_explanation(self, state: AgentState) -> str:
        """
        Generate a human-readable explanation of the parsed query.
        """
        if not state.parsed_query:
            return "No query parsed yet."

        query = state.parsed_query
        intent = query.get("intent", "unknown")
        columns = query.get("columns", [])
        confidence = state.query_confidence or 0

        explanation = f"Intent: {intent.replace('_', ' ').title()}\n"
        explanation += f"Confidence: {confidence:.1%}\n"

        if columns:
            explanation += f"Columns: {', '.join(columns)}\n"

        if query.get("filters"):
            explanation += f"Filters: {query['filters']}\n"

        if query.get("aggregation"):
            explanation += f"Aggregation: {query['aggregation']}\n"

        return explanation

    def _calculate_confidence(self, parsed_intent, data) -> float:
        """
        Calculate confidence score for the parsed query.
        """
        confidence = 0.5  # Base confidence

        # Add confidence based on intent clarity
        if parsed_intent.query_type and str(parsed_intent.query_type) != "unknown":
            confidence += 0.3

        # Add confidence based on column matching
        all_columns = parsed_intent.primary_columns + (parsed_intent.secondary_columns or [])
        if all_columns:
            data_columns = list(data.columns)
            matching_columns = [col for col in all_columns if col in data_columns]
            if matching_columns:
                confidence += 0.2 * (len(matching_columns) / len(all_columns))

        # Reduce confidence for complex queries without specific details
        query_type_str = str(parsed_intent.query_type).lower()
        if query_type_str in ["correlation", "trend_analysis"] and not all_columns:
            confidence -= 0.2

        return min(1.0, max(0.1, confidence))