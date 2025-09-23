"""
Test file for CSV loader functionality
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from components.csv_loader import CSVLoader
from utils.exceptions import CSVLoadError


def create_sample_csv():
    """Create a sample CSV file for testing"""
    data = {
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'Store Sales': ['1,234.50', '2,345.60', '3,456.70', '4,567.80', '5,678.90'],
        'Non-Store Sales': [1500, 2600, 3700, 4800, 5900],
        'Category': ['Electronics', 'Clothing', 'Electronics', 'Home', 'Clothing'],
        'Region': ['North', 'South', 'East', 'West', 'North'],
        ' Extra Spaces ': ['   value1   ', '   value2   ', '   value3   ', '   value4   ', '   value5   '],
        'Empty Column': [None, None, None, None, None],
        'Mixed Data': ['123', 'abc', '456.78', 'def', '789']
    }

    df = pd.DataFrame(data)
    csv_path = Path(__file__).parent.parent / "data" / "sample_data.csv"
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def test_csv_loader():
    """Test the CSV loader functionality"""
    print("=== CSV Loader Test ===\n")

    # Create sample data
    csv_path = create_sample_csv()
    print(f"Created sample CSV at: {csv_path}")

    # Initialize loader
    loader = CSVLoader()

    # Test complete workflow
    try:
        df, validation, preview = loader.load_and_process(str(csv_path))

        print(f"\n--- Data Shape ---")
        print(f"Shape: {df.shape}")

        print(f"\n--- Columns ---")
        print(f"Columns: {df.columns.tolist()}")

        print(f"\n--- Data Types ---")
        print(df.dtypes)

        print(f"\n--- Validation Results ---")
        print(f"Valid: {validation['is_valid']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
        if validation['suggestions']:
            print(f"Suggestions: {validation['suggestions']}")

        print(f"\n--- Preview ---")
        print(f"Memory usage: {preview['memory_usage']}")
        print(f"Null counts: {preview['null_counts']}")

        print(f"\n--- Sample Data ---")
        print(df.head())

        print(f"\n--- Metadata ---")
        if loader.metadata:
            print(f"Filename: {loader.metadata.filename}")
            print(f"Encoding: {loader.metadata.encoding}")
            print(f"Delimiter: {loader.metadata.delimiter}")
            print(f"Memory usage: {loader.metadata.memory_usage:.2f} MB")

        print("\n✅ CSV Loader test completed successfully!")

    except CSVLoadError as e:
        print(f"❌ CSV Loading failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    test_csv_loader()