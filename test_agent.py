"""
Test script for the LangGraph Excel Agent workflow.
Demonstrates the agent-based approach for CSV analysis.
"""

import os
import sys
import asyncio
import pandas as pd
from datetime import datetime

# Add the project root to the path
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents import ExcelAgent


def create_sample_data():
    """Create sample CSV data for testing."""
    print("Creating sample data...")

    # Create sample sales data
    data = {
        'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'Store_Type': ['Store'] * 60 + ['Non_Store'] * 40,
        'Category': ['Electronics', 'Clothing', 'Books', 'Home'] * 25,
        'Sales': [1000 + i * 10 + (i % 7) * 50 for i in range(100)],
        'Units_Sold': [10 + i + (i % 5) * 2 for i in range(100)],
        'Customer_Rating': [3.5 + (i % 10) * 0.2 for i in range(100)]
    }

    df = pd.DataFrame(data)

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    # Save to CSV
    csv_path = "outputs/sample_sales_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Sample data saved to: {csv_path}")

    return csv_path


async def test_agent_workflow():
    """Test the agent workflow with various queries."""
    print("=== Testing Excel Agent LangGraph Workflow ===\n")

    # Create sample data
    csv_path = create_sample_data()

    # Initialize the agent
    agent = ExcelAgent(output_directory="outputs/agent_tests")

    # Validate setup
    print("Validating agent setup...")
    validation = await agent.validate_setup()
    print(f"Setup valid: {validation['valid']}")
    if validation.get('warnings'):
        print(f"Warnings: {validation['warnings']}")
    if validation.get('errors'):
        print(f"Errors: {validation['errors']}")
    print()

    # Test queries
    test_queries = [
        "Compare store vs non-store sales performance",
        "Show correlation between sales and customer rating",
        "Analyze sales trends over time",
        "What are the top performing categories?",
        "Show summary statistics for all numeric columns"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"=== Test {i}: {query} ===")

        try:
            # Execute the query
            result = await agent.analyze_csv(csv_path, query)

            if result["success"]:
                print("[SUCCESS] Analysis completed successfully!")
                print(f"Output file: {result['output_path']}")
                print(f"Analysis type: {result['analysis_type']}")
                print(f"Query confidence: {result['confidence']:.1%}")
                print(f"Charts generated: {result['charts_generated']}")

                if result.get('data_summary'):
                    data_summary = result['data_summary']
                    print(f"Data: {data_summary['rows']} rows, {data_summary['columns']} columns")

                # Show last few messages
                if result.get('messages'):
                    print("Recent workflow steps:")
                    for msg in result['messages'][-3:]:
                        print(f"  - {msg}")

            else:
                print("[FAILED] Analysis failed!")
                print(f"Error: {result['error']}")
                if result.get('messages'):
                    print("Workflow messages:")
                    for msg in result['messages'][-3:]:
                        print(f"  - {msg}")

        except Exception as e:
            print(f"[ERROR] Test failed with exception: {str(e)}")

        print("-" * 60 + "\n")

    # Show execution summary
    summary = agent.get_last_execution_summary()
    if summary:
        print("Last Execution Summary:")
        print(summary)
        print()

    # Show workflow info
    workflow_info = agent.get_workflow_info()
    print("Workflow Information:")
    print(f"Nodes: {workflow_info['nodes']}")
    print(f"LangGraph available: {workflow_info['langgraph_available']}")
    print(f"Workflow compiled: {workflow_info['workflow_available']}")


def test_synchronous_agent():
    """Test the synchronous version of the agent."""
    print("=== Testing Synchronous Agent Interface ===\n")

    # Create sample data
    csv_path = create_sample_data()

    # Initialize the agent
    agent = ExcelAgent(output_directory="outputs/sync_tests")

    # Test a simple query
    query = "Show summary statistics for sales data"
    print(f"Query: {query}")

    try:
        result = agent.analyze_csv_sync(csv_path, query)

        if result["success"]:
            print("[SUCCESS] Synchronous analysis completed!")
            print(f"Output: {result['output_path']}")
            print(f"Analysis type: {result['analysis_type']}")
        else:
            print("[FAILED] Synchronous analysis failed!")
            print(f"Error: {result['error']}")

    except Exception as e:
        print(f"[ERROR] Synchronous test failed: {str(e)}")


def test_query_suggestions():
    """Test query suggestion functionality."""
    print("=== Testing Query Suggestions ===\n")

    # Create sample data
    csv_path = create_sample_data()

    # Initialize the agent
    agent = ExcelAgent()

    # Get suggestions
    suggestions = agent.suggest_queries(csv_path)

    print("Suggested queries for the sample data:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")


async def main():
    """Main test function."""
    print(f"Excel Agent Test Started: {datetime.now()}\n")

    try:
        # Test async workflow
        await test_agent_workflow()

        print("\n" + "="*80 + "\n")

        # Test sync interface
        test_synchronous_agent()

        print("\n" + "="*80 + "\n")

        # Test suggestions
        test_query_suggestions()

    except Exception as e:
        print(f"Main test failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print(f"\nExcel Agent Test Completed: {datetime.now()}")


if __name__ == "__main__":
    asyncio.run(main())