"""
Excel Agent - Command Line Interface

Simple CLI for the Excel Agent MVP that demonstrates the complete workflow.
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.excel_agent import ExcelAgent
from src.components.excel_exporter import ExcelExportConfig


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Excel Agent - CSV Analysis to Excel Reports")

    parser.add_argument("csv_file", help="Path to CSV file to analyze")
    parser.add_argument("query", help="Natural language query for analysis")
    parser.add_argument("--output", "-o", help="Output Excel filename")
    parser.add_argument("--preview", "-p", action="store_true", help="Show data preview only")
    parser.add_argument("--suggest", "-s", action="store_true", help="Show suggested queries")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Initialize Excel Agent
    agent = ExcelAgent()

    try:
        if args.preview:
            print("📊 Getting data preview...")
            preview = agent.get_data_preview(args.csv_file)

            if preview['success']:
                print(f"\n✅ File loaded successfully:")
                print(f"   📁 Shape: {preview['shape'][0]} rows × {preview['shape'][1]} columns")
                print(f"   📋 Columns: {', '.join(preview['columns'][:5])}{'...' if len(preview['columns']) > 5 else ''}")
                print(f"   💾 Memory: {preview['metadata']['memory_usage']}")
                print(f"   🔤 Encoding: {preview['metadata']['encoding']}")
            else:
                print(f"❌ Error: {preview['error']}")
            return

        if args.suggest:
            print("💡 Generating suggested queries...")
            suggestions = agent.suggest_queries(args.csv_file)

            print(f"\n📝 Suggested queries for your data:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
            return

        # Run full analysis workflow
        print("🚀 Starting Excel Agent analysis...")
        print(f"📁 File: {args.csv_file}")
        print(f"❓ Query: {args.query}")
        print()

        # Process file and query with progress bar
        with tqdm(total=5, desc="Progress", unit="step") as pbar:
            pbar.set_description("Loading CSV")
            pbar.update(1)

            result = agent.process_file_and_query(
                csv_file_path=args.csv_file,
                query=args.query,
                excel_filename=args.output,
                progress_callback=lambda step: (pbar.set_description(step), pbar.update(1))
            )

        # Display results
        print("📋 Workflow Results:")
        print("=" * 50)

        # Print messages
        for message in result.messages:
            print(f"   {message}")

        # Print warnings
        if result.warnings:
            print("\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   {warning}")

        # Print errors
        if result.errors:
            print("\n❌ Errors:")
            for error in result.errors:
                print(f"   {error}")

        print("\n" + "=" * 50)

        if result.success:
            print("🎉 Analysis completed successfully!")
            if result.excel_result:
                print(f"📄 Excel report: {result.excel_result.file_path}")
                print(f"📊 Sheets: {', '.join(result.excel_result.sheets_created)}")
        else:
            print("❌ Analysis failed. Check errors above.")
            sys.exit(1)

        # Verbose output
        if args.verbose:
            print(f"\n🔍 Detailed Results:")
            print(f"   CSV Loaded: {result.csv_loaded}")
            print(f"   Query Parsed: {result.query_parsed}")
            print(f"   Analysis Completed: {result.analysis_completed}")
            print(f"   Charts Generated: {result.charts_generated}")
            print(f"   Excel Exported: {result.excel_exported}")

            if result.query_intent:
                print(f"   Query Type: {result.query_intent.query_type.value}")
                print(f"   Confidence: {result.query_intent.confidence:.2f}")

            if result.analysis_results:
                for analysis in result.analysis_results:
                    print(f"   Analysis: {analysis.analysis_type}")
                    print(f"   Insights: {len(analysis.insights)}")

    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()