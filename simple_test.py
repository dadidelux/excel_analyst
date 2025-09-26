"""
Simple test for the agent workflow.
"""

import os
import sys
import asyncio
import pandas as pd

# Add the project root to the path
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents import ExcelAgent


def create_simple_data():
    """Create simple CSV data for testing."""
    print("Creating simple test data...")

    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Score': [85, 90, 95]
    }

    df = pd.DataFrame(data)
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/simple_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"Simple data saved to: {csv_path}")
    return csv_path


async def test_simple_query():
    """Test a simple query."""
    print("=== Simple Agent Test ===\n")

    csv_path = create_simple_data()
    agent = ExcelAgent(output_directory="outputs/simple_test")

    query = "Show summary statistics for the data"
    print(f"Query: {query}")

    try:
        result = await agent.analyze_csv(csv_path, query)

        if result["success"]:
            print("[SUCCESS] Analysis completed!")
            print(f"Output: {result['output_path']}")
            print(f"Analysis type: {result['analysis_type']}")
            print(f"Confidence: {result['confidence']:.1%}")
        else:
            print("[FAILED] Analysis failed!")
            print(f"Error: {result['error']}")

    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple_query())