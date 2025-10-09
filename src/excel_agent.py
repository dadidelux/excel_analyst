"""
Excel Agent - Main Integration Module

Orchestrates the complete workflow from CSV loading to Excel report generation.
This is the primary interface for the Excel Agent MVP.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import traceback

from .components.csv_loader import CSVLoader, CSVMetadata
from .components.query_parser import QueryParser, QueryIntent, QueryValidationResult
from .components.data_analyzer import DataAnalyzer, AnalysisResult
from .components.chart_generator import ChartGenerator, ChartOutput, ChartConfig
from .components.excel_exporter import ExcelExporter, ExcelExportResult, ExcelExportConfig
from .utils.logger import get_logger
from .utils.exceptions import ExcelAgentError

logger = get_logger(__name__)


@dataclass
class AgentWorkflowResult:
    """Complete workflow result"""
    success: bool
    csv_loaded: bool = False
    query_parsed: bool = False
    analysis_completed: bool = False
    charts_generated: bool = False
    excel_exported: bool = False

    # Data
    raw_data: Optional[pd.DataFrame] = None
    metadata: Optional[CSVMetadata] = None
    query_intent: Optional[QueryIntent] = None
    analysis_results: List[AnalysisResult] = None
    charts: List[ChartOutput] = None
    excel_result: Optional[ExcelExportResult] = None

    # Messages
    messages: List[str] = None
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.analysis_results is None:
            self.analysis_results = []
        if self.charts is None:
            self.charts = []
        if self.messages is None:
            self.messages = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class ExcelAgent:
    """
    Main Excel Agent class that orchestrates the complete analysis workflow
    """

    def __init__(self, output_dir: str = "outputs"):
        """Initialize the Excel Agent with all components"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all components
        self.csv_loader = CSVLoader()
        self.query_parser = QueryParser()
        self.data_analyzer = DataAnalyzer()
        self.chart_generator = ChartGenerator(str(self.output_dir / "charts"))
        self.excel_exporter = ExcelExporter(str(self.output_dir))

        logger.info("Excel Agent initialized successfully")

    def process_file_and_query(
        self,
        csv_file_path: str,
        query: str,
        excel_filename: Optional[str] = None,
        export_config: Optional[ExcelExportConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> AgentWorkflowResult:
        """
        Complete workflow: Load CSV, parse query, analyze data, generate charts, export to Excel

        Args:
            csv_file_path: Path to CSV file
            query: Natural language query
            excel_filename: Optional output Excel filename
            export_config: Optional Excel export configuration
            progress_callback: Optional callback function for progress updates

        Returns:
            AgentWorkflowResult with complete workflow results
        """
        result = AgentWorkflowResult(success=False)

        try:
            logger.info(f"Starting Excel Agent workflow for file: {csv_file_path}")
            logger.info(f"Query: '{query}'")

            # Step 1: Load and validate CSV
            result.messages.append("Loading CSV file...")
            try:
                df, validation, preview = self.csv_loader.load_and_process(csv_file_path)
                result.raw_data = df
                result.metadata = self.csv_loader.metadata
                result.csv_loaded = True
                result.messages.append(f"✅ CSV loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

                # Add validation warnings
                if validation['warnings']:
                    result.warnings.extend(validation['warnings'])
                if validation['suggestions']:
                    result.messages.extend([f"💡 {s}" for s in validation['suggestions']])

            except Exception as e:
                result.errors.append(f"CSV loading failed: {str(e)}")
                logger.error(f"CSV loading failed: {e}")
                return result

            # Step 2: Parse and validate query
            if progress_callback:
                progress_callback("Parsing query")
            result.messages.append("Parsing natural language query...")
            try:
                # Set data context for parser
                self.query_parser.set_data_context(df)

                # Parse query
                intent = self.query_parser.parse_query(query)
                result.query_intent = intent
                result.query_parsed = True

                result.messages.append(f"✅ Query parsed: {intent.query_type.value}")
                result.messages.append(f"📊 Confidence: {intent.confidence:.2f}")

                # Validate query against data
                query_validation = self.query_parser.validate_query_intent(intent)
                if not query_validation.is_valid:
                    result.warnings.extend(query_validation.errors)
                    result.messages.append("⚠️ Query validation issues found")

                if intent.clarifications_needed:
                    result.messages.append("🤔 Clarifications that could improve results:")
                    result.messages.extend([f"  • {c}" for c in intent.clarifications_needed])

            except Exception as e:
                result.errors.append(f"Query parsing failed: {str(e)}")
                logger.error(f"Query parsing failed: {e}")
                return result

            # Step 3: Perform data analysis
            if progress_callback:
                progress_callback("Running analysis")
            result.messages.append("Performing data analysis...")
            try:
                # Set data for analyzer
                self.data_analyzer.set_data(df)

                # Perform analysis
                analysis_result = self.data_analyzer.analyze(intent)
                result.analysis_results = [analysis_result]
                result.analysis_completed = analysis_result.success

                if analysis_result.success:
                    result.messages.append(f"✅ Analysis completed: {analysis_result.analysis_type}")
                    result.messages.append(f"📈 Insights generated: {len(analysis_result.insights)}")
                else:
                    result.errors.append(f"Analysis failed: {analysis_result.error_message}")
                    return result

            except Exception as e:
                result.errors.append(f"Data analysis failed: {str(e)}")
                logger.error(f"Data analysis failed: {e}")
                return result

            # Step 4: Generate charts
            if progress_callback:
                progress_callback("Generating charts")
            result.messages.append("Generating visualizations...")
            try:
                charts = self.chart_generator.generate_charts_from_analysis(analysis_result)
                result.charts = charts
                successful_charts = [c for c in charts if c.success]
                result.charts_generated = len(successful_charts) > 0

                result.messages.append(f"✅ Charts generated: {len(successful_charts)}/{len(charts)}")

                # Report chart generation failures
                failed_charts = [c for c in charts if not c.success]
                if failed_charts:
                    result.warnings.extend([f"Chart generation failed: {c.title}" for c in failed_charts])

            except Exception as e:
                result.errors.append(f"Chart generation failed: {str(e)}")
                logger.error(f"Chart generation failed: {e}")
                # Continue without charts

            # Step 5: Export to Excel
            if progress_callback:
                progress_callback("Exporting to Excel")
            result.messages.append("Exporting to Excel...")
            try:
                if not export_config:
                    export_config = ExcelExportConfig()

                excel_result = self.excel_exporter.export_analysis_to_excel(
                    analysis_results=result.analysis_results,
                    charts=result.charts,
                    raw_data=df,
                    metadata=result.metadata,
                    config=export_config,
                    filename=excel_filename
                )

                result.excel_result = excel_result
                result.excel_exported = excel_result.success

                if excel_result.success:
                    result.messages.append(f"✅ Excel report exported: {excel_result.file_path}")
                    result.messages.append(f"📄 Sheets created: {len(excel_result.sheets_created)}")
                    result.messages.append(f"📊 Charts embedded: {excel_result.charts_embedded}")
                    result.messages.append(f"📁 File size: {excel_result.file_size_mb:.2f} MB")
                else:
                    result.errors.append(f"Excel export failed: {excel_result.error_message}")
                    return result

            except Exception as e:
                result.errors.append(f"Excel export failed: {str(e)}")
                logger.error(f"Excel export failed: {e}")
                return result

            # Mark overall success
            result.success = (
                result.csv_loaded and
                result.query_parsed and
                result.analysis_completed and
                result.excel_exported
            )

            if result.success:
                result.messages.append("🎉 Complete workflow finished successfully!")
                logger.info("Excel Agent workflow completed successfully")
            else:
                result.messages.append("⚠️ Workflow completed with some issues")

            return result

        except Exception as e:
            result.errors.append(f"Unexpected workflow error: {str(e)}")
            logger.error(f"Workflow error: {e}")
            logger.error(traceback.format_exc())
            return result

    def analyze_multiple_queries(
        self,
        csv_file_path: str,
        queries: List[str],
        excel_filename: Optional[str] = None
    ) -> AgentWorkflowResult:
        """
        Process multiple queries against the same dataset

        Args:
            csv_file_path: Path to CSV file
            queries: List of natural language queries
            excel_filename: Optional output Excel filename

        Returns:
            AgentWorkflowResult with all analyses
        """
        result = AgentWorkflowResult(success=False)

        try:
            # Load CSV once
            result.messages.append("Loading CSV file...")
            df, validation, preview = self.csv_loader.load_and_process(csv_file_path)
            result.raw_data = df
            result.metadata = self.csv_loader.metadata
            result.csv_loaded = True

            # Set data context
            self.query_parser.set_data_context(df)
            self.data_analyzer.set_data(df)

            # Process each query
            all_analyses = []
            all_charts = []

            for i, query in enumerate(queries):
                result.messages.append(f"Processing query {i+1}/{len(queries)}: '{query}'")

                try:
                    # Parse query
                    intent = self.query_parser.parse_query(query)

                    # Analyze data
                    analysis_result = self.data_analyzer.analyze(intent)
                    all_analyses.append(analysis_result)

                    # Generate charts
                    charts = self.chart_generator.generate_charts_from_analysis(analysis_result)
                    all_charts.extend(charts)

                    if analysis_result.success:
                        result.messages.append(f"  ✅ Query {i+1} completed successfully")
                    else:
                        result.warnings.append(f"  ⚠️ Query {i+1} failed: {analysis_result.error_message}")

                except Exception as e:
                    result.warnings.append(f"  ❌ Query {i+1} failed: {str(e)}")

            result.analysis_results = all_analyses
            result.charts = all_charts
            result.analysis_completed = len(all_analyses) > 0
            result.charts_generated = len([c for c in all_charts if c.success]) > 0

            # Export to Excel
            if all_analyses:
                excel_result = self.excel_exporter.export_analysis_to_excel(
                    analysis_results=all_analyses,
                    charts=all_charts,
                    raw_data=df,
                    metadata=result.metadata,
                    filename=excel_filename
                )

                result.excel_result = excel_result
                result.excel_exported = excel_result.success

                if excel_result.success:
                    result.messages.append(f"✅ Multi-analysis Excel report exported: {excel_result.file_path}")

            result.success = result.csv_loaded and result.analysis_completed and result.excel_exported
            return result

        except Exception as e:
            result.errors.append(f"Multi-query workflow error: {str(e)}")
            logger.error(f"Multi-query workflow error: {e}")
            return result

    def get_data_preview(self, csv_file_path: str) -> Dict[str, Any]:
        """
        Get a preview of the CSV data without running full analysis

        Args:
            csv_file_path: Path to CSV file

        Returns:
            Dictionary with data preview information
        """
        try:
            df, validation, preview = self.csv_loader.load_and_process(csv_file_path)

            return {
                'success': True,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'preview': preview,
                'validation': validation,
                'metadata': {
                    'filename': self.csv_loader.metadata.filename,
                    'encoding': self.csv_loader.metadata.encoding,
                    'delimiter': self.csv_loader.metadata.delimiter,
                    'memory_usage': f"{self.csv_loader.metadata.memory_usage:.2f} MB"
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def suggest_queries(self, csv_file_path: str) -> List[str]:
        """
        Suggest relevant queries based on the dataset structure

        Args:
            csv_file_path: Path to CSV file

        Returns:
            List of suggested queries
        """
        try:
            df, _, _ = self.csv_loader.load_and_process(csv_file_path)
            self.query_parser.set_data_context(df)

            suggestions = []
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

            # Suggest based on column types
            if len(numeric_cols) >= 2:
                suggestions.append(f"Compare {numeric_cols[0]} vs {numeric_cols[1]}")
                suggestions.append(f"Show correlation between {numeric_cols[0]} and {numeric_cols[1]}")

            if numeric_cols and categorical_cols:
                suggestions.append(f"Show {numeric_cols[0]} by {categorical_cols[0]}")
                suggestions.append(f"Top 10 {categorical_cols[0]} by {numeric_cols[0]}")

            if date_cols and numeric_cols:
                suggestions.append(f"Show trends in {numeric_cols[0]} over time")

            if numeric_cols:
                suggestions.append(f"Summary statistics for {numeric_cols[0]}")

            if categorical_cols:
                suggestions.append(f"Distribution of {categorical_cols[0]}")

            # Generic suggestions
            suggestions.extend([
                "Overview of the dataset",
                "Identify outliers and patterns",
                "Key insights and recommendations"
            ])

            return suggestions[:8]  # Return top 8 suggestions

        except Exception as e:
            logger.error(f"Failed to generate query suggestions: {e}")
            return [
                "Compare columns in the dataset",
                "Show correlation between variables",
                "Analyze trends over time",
                "Summary statistics",
                "Top performers analysis"
            ]

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow component status"""
        return {
            'csv_loader': 'Ready',
            'query_parser': 'Ready',
            'data_analyzer': 'Ready',
            'chart_generator': 'Ready',
            'excel_exporter': 'Ready',
            'output_directory': str(self.output_dir),
            'history_count': len(self.data_analyzer.analysis_history)
        }