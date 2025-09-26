"""
Excel Agent Demo - Complete Functionality Showcase

This demo script creates sample data and demonstrates all Excel Agent capabilities.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.excel_agent import ExcelAgent


def create_sample_data():
    """Create comprehensive sample dataset for demonstration"""
    print("📊 Creating sample dataset...")

    # Generate 365 days of data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    np.random.seed(42)  # For reproducible results

    # Create realistic business data
    data = {
        'Date': dates,
        'Store_Sales': [5000 + i*10 + np.random.normal(0, 500) + 1000*np.sin(i/30) for i in range(365)],
        'Online_Sales': [3000 + i*8 + np.random.normal(0, 300) + 800*np.sin(i/20 + 1) for i in range(365)],
        'Customer_Count': [200 + i*2 + np.random.normal(0, 20) + 50*np.sin(i/7) for i in range(365)],
        'Marketing_Spend': [1000 + np.random.normal(0, 200) + 500*np.sin(i/14) for i in range(365)],
        'Product_Returns': [50 + np.random.normal(0, 15) + 20*np.random.random() for i in range(365)],
    }

    # Add categorical data
    categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']
    regions = ['North', 'South', 'East', 'West', 'Central']

    data['Category'] = [categories[i % len(categories)] for i in range(365)]
    data['Region'] = [regions[i % len(regions)] for i in range(365)]

    # Add seasonal patterns
    data['Season'] = ['Winter' if m in [12, 1, 2] else
                     'Spring' if m in [3, 4, 5] else
                     'Summer' if m in [6, 7, 8] else
                     'Fall' for m in pd.to_datetime(data['Date']).dt.month]

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure positive values
    df['Store_Sales'] = df['Store_Sales'].abs()
    df['Online_Sales'] = df['Online_Sales'].abs()
    df['Customer_Count'] = df['Customer_Count'].abs()
    df['Marketing_Spend'] = df['Marketing_Spend'].abs()
    df['Product_Returns'] = df['Product_Returns'].abs()

    # Save to CSV
    csv_path = Path("data/sample_business_data.csv")
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"✅ Sample data created: {csv_path}")
    print(f"   📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   📅 Date range: {df['Date'].min()} to {df['Date'].max()}")

    return str(csv_path)


def demo_workflow():
    """Demonstrate complete Excel Agent workflow"""
    print("\n" + "="*60)
    print("🚀 EXCEL AGENT MVP - COMPLETE DEMO")
    print("="*60)

    # Create sample data
    csv_file = create_sample_data()

    # Initialize Excel Agent
    print("\n🔧 Initializing Excel Agent...")
    agent = ExcelAgent()

    # Demo queries to test different analysis types
    demo_queries = [
        {
            'query': 'Compare store sales vs online sales performance',
            'description': 'Comparison Analysis'
        },
        {
            'query': 'Show correlation between marketing spend and total sales',
            'description': 'Correlation Analysis'
        },
        {
            'query': 'What are the trends in sales over time?',
            'description': 'Trend Analysis'
        },
        {
            'query': 'Top 5 regions by average sales',
            'description': 'Top-N Analysis'
        },
        {
            'query': 'Summary statistics for all sales metrics',
            'description': 'Summary Statistics'
        }
    ]

    print(f"\n📋 Running {len(demo_queries)} different analysis types...")

    # Demo 1: Data Preview
    print(f"\n📊 Demo 1: Data Preview")
    print("-" * 30)
    preview = agent.get_data_preview(csv_file)
    if preview['success']:
        print(f"✅ Data loaded successfully")
        print(f"   📏 Shape: {preview['shape'][0]} rows × {preview['shape'][1]} columns")
        print(f"   📋 Columns: {', '.join(preview['columns'])}")
        print(f"   💾 Memory: {preview['metadata']['memory_usage']}")

    # Demo 2: Query Suggestions
    print(f"\n💡 Demo 2: Query Suggestions")
    print("-" * 30)
    suggestions = agent.suggest_queries(csv_file)
    print("Suggested queries:")
    for i, suggestion in enumerate(suggestions[:5], 1):
        print(f"   {i}. {suggestion}")

    # Demo 3: Individual Query Processing
    print(f"\n🔍 Demo 3: Individual Query Analysis")
    print("-" * 40)

    for i, demo in enumerate(demo_queries[:3], 1):  # Run first 3 for individual demo
        print(f"\n📈 Analysis {i}: {demo['description']}")
        print(f"Query: '{demo['query']}'")

        result = agent.process_file_and_query(
            csv_file_path=csv_file,
            query=demo['query'],
            excel_filename=f"demo_analysis_{i}.xlsx"
        )

        print(f"Result: {'✅ Success' if result.success else '❌ Failed'}")
        if result.success and result.excel_result:
            print(f"   📄 Excel: {result.excel_result.file_path}")
            print(f"   📊 Sheets: {len(result.excel_result.sheets_created)}")
            print(f"   📈 Charts: {result.excel_result.charts_embedded}")

        # Show key insights
        if result.analysis_results and result.analysis_results[0].insights:
            print("   🔍 Key Insights:")
            for insight in result.analysis_results[0].insights[:2]:
                print(f"      • {insight}")

    # Demo 4: Multiple Query Analysis
    print(f"\n🔄 Demo 4: Multi-Query Analysis")
    print("-" * 35)

    all_queries = [demo['query'] for demo in demo_queries]
    multi_result = agent.analyze_multiple_queries(
        csv_file_path=csv_file,
        queries=all_queries,
        excel_filename="comprehensive_analysis_report.xlsx"
    )

    print(f"Multi-analysis result: {'✅ Success' if multi_result.success else '❌ Failed'}")
    if multi_result.success:
        print(f"   📊 Analyses completed: {len(multi_result.analysis_results)}")
        print(f"   📈 Charts generated: {len([c for c in multi_result.charts if c.success])}")
        if multi_result.excel_result:
            print(f"   📄 Comprehensive report: {multi_result.excel_result.file_path}")
            print(f"   📁 File size: {multi_result.excel_result.file_size_mb:.2f} MB")

    # Demo 5: System Status
    print(f"\n⚙️  Demo 5: System Status")
    print("-" * 25)
    status = agent.get_workflow_status()
    print("Component Status:")
    for component, state in status.items():
        if component != 'history_count':
            print(f"   {component}: {state}")
    print(f"   Analysis history: {status['history_count']} items")

    # Final Summary
    print(f"\n" + "="*60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Output files created in: {agent.output_dir}")
    print(f"📊 Multiple Excel reports generated with:")
    print(f"   • Professional multi-sheet layout")
    print(f"   • Embedded charts and visualizations")
    print(f"   • Formatted analysis results")
    print(f"   • Executive summaries")
    print(f"   • Raw data preservation")
    print(f"   • Comprehensive metadata")

    print(f"\n🔧 Key Features Demonstrated:")
    print(f"   ✅ Smart CSV loading with encoding detection")
    print(f"   ✅ Natural language query understanding")
    print(f"   ✅ 5+ different analysis types")
    print(f"   ✅ Professional chart generation")
    print(f"   ✅ Excel export with embedded visualizations")
    print(f"   ✅ Multi-query batch processing")
    print(f"   ✅ Comprehensive error handling")

    print(f"\n📋 Ready for LangGraph agent integration!")


if __name__ == "__main__":
    try:
        demo_workflow()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()