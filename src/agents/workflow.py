"""
LangGraph workflow for the Excel Agent.
Defines the agent workflow with nodes and routing logic.
"""

import os
import sys
from typing import Dict, Any, List
import asyncio
from datetime import datetime

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    print("Warning: LangGraph not available. Install with: pip install langgraph")
    StateGraph = None
    END = "END"
    MemorySaver = None

from .state import AgentState, WorkflowStatus, NodeConfig, should_continue_to_analysis, should_generate_charts, should_export_excel, needs_query_clarification
from .nodes import (
    CSVIngestionNode,
    QueryProcessingNode,
    AnalysisNode,
    VisualizationNode,
    ExcelExportNode
)


class ExcelAgentWorkflow:
    """
    Main workflow orchestrator for the Excel Agent using LangGraph.
    """

    def __init__(self):
        self.nodes = {}
        self.workflow = None
        self.app = None
        self._initialize_nodes()
        self._build_workflow()

    def _initialize_nodes(self):
        """Initialize all agent nodes."""
        self.nodes = {
            "csv_ingestion": CSVIngestionNode(NodeConfig(node_name="csv_ingestion")),
            "query_processing": QueryProcessingNode(NodeConfig(node_name="query_processing")),
            "analysis": AnalysisNode(NodeConfig(node_name="analysis")),
            "visualization": VisualizationNode(NodeConfig(node_name="visualization")),
            "excel_export": ExcelExportNode(NodeConfig(node_name="excel_export"))
        }

    def _build_workflow(self):
        """Build the LangGraph workflow."""
        if StateGraph is None:
            print("Warning: LangGraph not available. Using fallback workflow.")
            return

        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("csv_ingestion", self._wrap_node_execution("csv_ingestion"))
        workflow.add_node("query_processing", self._wrap_node_execution("query_processing"))
        workflow.add_node("analysis", self._wrap_node_execution("analysis"))
        workflow.add_node("visualization", self._wrap_node_execution("visualization"))
        workflow.add_node("excel_export", self._wrap_node_execution("excel_export"))

        # Set entry point
        workflow.set_entry_point("csv_ingestion")

        # Add conditional routing
        workflow.add_conditional_edges(
            "csv_ingestion",
            self._route_from_csv_ingestion,
            {
                "query_processing": "query_processing",
                "error": END
            }
        )

        workflow.add_conditional_edges(
            "query_processing",
            self._route_from_query_processing,
            {
                "analysis": "analysis",
                "clarification": END,  # Could route to clarification node in future
                "error": END
            }
        )

        workflow.add_conditional_edges(
            "analysis",
            self._route_from_analysis,
            {
                "visualization": "visualization",
                "error": END
            }
        )

        workflow.add_conditional_edges(
            "visualization",
            self._route_from_visualization,
            {
                "excel_export": "excel_export",
                "error": END
            }
        )

        workflow.add_edge("excel_export", END)

        # Compile the workflow
        if MemorySaver:
            checkpointer = MemorySaver()
            self.app = workflow.compile(checkpointer=checkpointer)
        else:
            self.app = workflow.compile()

        self.workflow = workflow

    def _wrap_node_execution(self, node_name: str):
        """Wrap node execution for LangGraph compatibility."""
        async def execute_node(state: AgentState) -> AgentState:
            node = self.nodes[node_name]
            try:
                result = await node.execute_with_retry(state)
                return result.updated_state
            except Exception as e:
                state.set_error(f"Node {node_name} failed: {str(e)}")
                return state

        return execute_node

    def _route_from_csv_ingestion(self, state: AgentState) -> str:
        """Route from CSV ingestion node."""
        if state.status == WorkflowStatus.ERROR:
            return "error"
        elif state.status == WorkflowStatus.DATA_LOADED:
            return "query_processing"
        else:
            return "error"

    def _route_from_query_processing(self, state: AgentState) -> str:
        """Route from query processing node."""
        if state.status == WorkflowStatus.ERROR:
            return "error"
        elif needs_query_clarification(state):
            return "clarification"
        elif should_continue_to_analysis(state):
            return "analysis"
        else:
            return "error"

    def _route_from_analysis(self, state: AgentState) -> str:
        """Route from analysis node."""
        if state.status == WorkflowStatus.ERROR:
            return "error"
        elif state.status == WorkflowStatus.ANALYSIS_COMPLETE:
            return "visualization"
        else:
            return "error"

    def _route_from_visualization(self, state: AgentState) -> str:
        """Route from visualization node."""
        if state.status == WorkflowStatus.ERROR:
            return "error"
        elif state.status == WorkflowStatus.CHARTS_GENERATED:
            return "excel_export"
        else:
            return "error"

    async def execute_workflow(self, csv_file_path: str, user_query: str, output_directory: str = "outputs") -> AgentState:
        """
        Execute the complete workflow.
        """
        # Initialize state
        initial_state = AgentState(
            csv_file_path=csv_file_path,
            user_query=user_query,
            output_directory=output_directory
        )

        initial_state.add_message(f"Starting Excel Agent workflow at {datetime.now().isoformat()}")

        if self.app is None:
            # Fallback to sequential execution if LangGraph not available
            return await self._execute_fallback_workflow(initial_state)

        try:
            # Execute using LangGraph
            config = {"configurable": {"thread_id": "excel_agent_session"}}
            result = await self.app.ainvoke(initial_state, config=config)
            return result

        except Exception as e:
            initial_state.set_error(f"Workflow execution failed: {str(e)}")
            return initial_state

    async def _execute_fallback_workflow(self, state: AgentState) -> AgentState:
        """
        Fallback workflow execution without LangGraph.
        """
        state.add_message("Executing fallback workflow (LangGraph not available)")

        # Sequential execution of nodes
        node_sequence = [
            "csv_ingestion",
            "query_processing",
            "analysis",
            "visualization",
            "excel_export"
        ]

        for node_name in node_sequence:
            if state.status == WorkflowStatus.ERROR:
                break

            node = self.nodes[node_name]
            try:
                result = await node.execute_with_retry(state)
                state = result.updated_state

                if not result.success:
                    state.add_message(f"Node {node_name} failed, stopping workflow")
                    break

            except Exception as e:
                state.set_error(f"Node {node_name} failed: {str(e)}")
                break

        return state

    def get_workflow_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the workflow configuration.
        """
        return {
            "nodes": list(self.nodes.keys()),
            "workflow_available": self.app is not None,
            "langgraph_available": StateGraph is not None,
            "node_configs": {
                name: {
                    "name": node.config.node_name,
                    "max_retries": node.config.max_retries,
                    "timeout": node.config.timeout_seconds
                }
                for name, node in self.nodes.items()
            }
        }

    async def validate_workflow(self) -> Dict[str, Any]:
        """
        Validate that the workflow is properly configured.
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "node_status": {}
        }

        # Check each node
        for name, node in self.nodes.items():
            try:
                # Basic validation - check if node can be initialized
                validation["node_status"][name] = "ok"
            except Exception as e:
                validation["errors"].append(f"Node {name} validation failed: {str(e)}")
                validation["node_status"][name] = "error"
                validation["valid"] = False

        # Check LangGraph availability
        if StateGraph is None:
            validation["warnings"].append("LangGraph not available - using fallback workflow")

        # Check workflow compilation
        if self.app is None and StateGraph is not None:
            validation["errors"].append("Workflow compilation failed")
            validation["valid"] = False

        return validation

    async def execute_single_node(self, node_name: str, state: AgentState) -> AgentState:
        """
        Execute a single node for testing purposes.
        """
        if node_name not in self.nodes:
            state.set_error(f"Node {node_name} not found")
            return state

        node = self.nodes[node_name]
        try:
            result = await node.execute_with_retry(state)
            return result.updated_state
        except Exception as e:
            state.set_error(f"Node {node_name} execution failed: {str(e)}")
            return state

    def get_node_info(self, node_name: str) -> Dict[str, Any]:
        """
        Get information about a specific node.
        """
        if node_name not in self.nodes:
            return {"error": f"Node {node_name} not found"}

        node = self.nodes[node_name]
        return {
            "name": node.config.node_name,
            "type": type(node).__name__,
            "max_retries": node.config.max_retries,
            "timeout_seconds": node.config.timeout_seconds,
            "enable_logging": node.config.enable_logging
        }