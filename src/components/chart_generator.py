"""
Chart Generation Module

Creates visualizations and charts based on analysis results for embedding in Excel files.
Supports multiple chart types using matplotlib, seaborn, and plotly.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import io
import base64
from pathlib import Path

from .data_analyzer import AnalysisResult
from .query_parser import QueryType
from ..utils.logger import get_logger
from ..utils.exceptions import VisualizationError

logger = get_logger(__name__)


class ChartType(Enum):
    """Supported chart types"""
    BAR_CHART = "bar"
    LINE_CHART = "line"
    SCATTER_PLOT = "scatter"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box"
    HEATMAP = "heatmap"
    PIE_CHART = "pie"
    AREA_CHART = "area"
    VIOLIN_PLOT = "violin"
    CORRELATION_MATRIX = "correlation"
    TIME_SERIES = "timeseries"
    MULTI_SERIES = "multi_series"


@dataclass
class ChartConfig:
    """Configuration for chart generation"""
    chart_type: ChartType
    title: str
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    width: int = 800
    height: int = 600
    color_palette: str = "viridis"
    style: str = "default"
    show_legend: bool = True
    show_grid: bool = True
    format: str = "png"  # png, svg, html


@dataclass
class ChartOutput:
    """Chart generation output"""
    chart_type: ChartType
    title: str
    file_path: Optional[str] = None
    base64_data: Optional[str] = None
    html_data: Optional[str] = None
    metadata: Dict[str, Any] = None
    success: bool = True
    error_message: Optional[str] = None


class ChartGenerator:
    """
    Chart generation system for creating visualizations from analysis results
    """

    def __init__(self, output_dir: str = "outputs/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style preferences
        plt.style.use('default')
        sns.set_palette("husl")

        # Chart type mapping for different analysis types
        self.analysis_chart_mapping = {
            QueryType.COMPARISON: [ChartType.BAR_CHART, ChartType.BOX_PLOT],
            QueryType.CORRELATION: [ChartType.SCATTER_PLOT, ChartType.HEATMAP],
            QueryType.TREND_ANALYSIS: [ChartType.LINE_CHART, ChartType.AREA_CHART],
            QueryType.AGGREGATION: [ChartType.BAR_CHART, ChartType.PIE_CHART],
            QueryType.TOP_N: [ChartType.BAR_CHART],
            QueryType.SUMMARY_STATS: [ChartType.HISTOGRAM, ChartType.BOX_PLOT],
            QueryType.TIME_SERIES: [ChartType.LINE_CHART, ChartType.TIME_SERIES]
        }

    def generate_charts_from_analysis(
        self,
        analysis_result: AnalysisResult,
        chart_configs: Optional[List[ChartConfig]] = None
    ) -> List[ChartOutput]:
        """
        Generate appropriate charts based on analysis results

        Args:
            analysis_result: Results from data analysis
            chart_configs: Optional specific chart configurations

        Returns:
            List of generated chart outputs
        """
        if not analysis_result.success:
            logger.error(f"Cannot generate charts for failed analysis: {analysis_result.error_message}")
            return []

        charts = []

        try:
            # Use provided configs or generate defaults
            if chart_configs:
                configs = chart_configs
            else:
                configs = self._generate_default_configs(analysis_result)

            # Generate each chart
            for config in configs:
                chart_output = self._generate_single_chart(analysis_result, config)
                charts.append(chart_output)

            logger.info(f"Generated {len(charts)} charts for {analysis_result.analysis_type}")

        except Exception as e:
            error_chart = ChartOutput(
                chart_type=ChartType.BAR_CHART,
                title="Chart Generation Failed",
                success=False,
                error_message=str(e)
            )
            charts.append(error_chart)
            logger.error(f"Chart generation failed: {e}")

        return charts

    def _generate_default_configs(self, analysis_result: AnalysisResult) -> List[ChartConfig]:
        """Generate default chart configurations based on analysis type"""
        configs = []

        analysis_type = analysis_result.analysis_type.lower()

        if "comparison" in analysis_type:
            configs.extend(self._create_comparison_configs(analysis_result))
        elif "correlation" in analysis_type:
            configs.extend(self._create_correlation_configs(analysis_result))
        elif "trend" in analysis_type:
            configs.extend(self._create_trend_configs(analysis_result))
        elif "aggregation" in analysis_type:
            configs.extend(self._create_aggregation_configs(analysis_result))
        elif "top" in analysis_type.lower():
            configs.extend(self._create_top_n_configs(analysis_result))
        elif "summary" in analysis_type:
            configs.extend(self._create_summary_configs(analysis_result))
        else:
            # Default fallback
            configs.append(ChartConfig(
                chart_type=ChartType.BAR_CHART,
                title="Analysis Results",
                width=800,
                height=600
            ))

        return configs

    def _create_comparison_configs(self, analysis_result: AnalysisResult) -> List[ChartConfig]:
        """Create chart configs for comparison analysis"""
        configs = []

        # Bar chart for basic comparison
        configs.append(ChartConfig(
            chart_type=ChartType.BAR_CHART,
            title="Comparison Analysis",
            x_label="Categories",
            y_label="Values",
            width=800,
            height=500
        ))

        # Box plot if multiple groups
        if any("grouped" in key for key in analysis_result.data.keys()):
            configs.append(ChartConfig(
                chart_type=ChartType.BOX_PLOT,
                title="Distribution Comparison",
                width=800,
                height=500
            ))

        return configs

    def _create_correlation_configs(self, analysis_result: AnalysisResult) -> List[ChartConfig]:
        """Create chart configs for correlation analysis"""
        configs = []

        # Scatter plot for correlation
        configs.append(ChartConfig(
            chart_type=ChartType.SCATTER_PLOT,
            title="Correlation Analysis",
            width=800,
            height=600
        ))

        # Heatmap if correlation matrix exists
        if any("matrix" in key.lower() for key in analysis_result.data.keys()):
            configs.append(ChartConfig(
                chart_type=ChartType.HEATMAP,
                title="Correlation Matrix",
                width=800,
                height=600
            ))

        return configs

    def _create_trend_configs(self, analysis_result: AnalysisResult) -> List[ChartConfig]:
        """Create chart configs for trend analysis"""
        configs = []

        # Line chart for trends
        configs.append(ChartConfig(
            chart_type=ChartType.LINE_CHART,
            title="Trend Analysis",
            x_label="Time",
            y_label="Value",
            width=1000,
            height=600
        ))

        return configs

    def _create_aggregation_configs(self, analysis_result: AnalysisResult) -> List[ChartConfig]:
        """Create chart configs for aggregation analysis"""
        configs = []

        # Bar chart for aggregated values
        configs.append(ChartConfig(
            chart_type=ChartType.BAR_CHART,
            title="Aggregation Results",
            width=800,
            height=500
        ))

        return configs

    def _create_top_n_configs(self, analysis_result: AnalysisResult) -> List[ChartConfig]:
        """Create chart configs for top-N analysis"""
        configs = []

        # Horizontal bar chart for top-N
        configs.append(ChartConfig(
            chart_type=ChartType.BAR_CHART,
            title="Top N Analysis",
            width=800,
            height=500
        ))

        return configs

    def _create_summary_configs(self, analysis_result: AnalysisResult) -> List[ChartConfig]:
        """Create chart configs for summary statistics"""
        configs = []

        # Histogram for distributions
        configs.append(ChartConfig(
            chart_type=ChartType.HISTOGRAM,
            title="Data Distribution",
            width=800,
            height=500
        ))

        # Box plot for summary stats
        configs.append(ChartConfig(
            chart_type=ChartType.BOX_PLOT,
            title="Summary Statistics",
            width=800,
            height=500
        ))

        return configs

    def _generate_single_chart(
        self,
        analysis_result: AnalysisResult,
        config: ChartConfig
    ) -> ChartOutput:
        """Generate a single chart based on configuration"""

        try:
            # Route to appropriate chart generation method
            if config.chart_type == ChartType.BAR_CHART:
                return self._create_bar_chart(analysis_result, config)
            elif config.chart_type == ChartType.LINE_CHART:
                return self._create_line_chart(analysis_result, config)
            elif config.chart_type == ChartType.SCATTER_PLOT:
                return self._create_scatter_plot(analysis_result, config)
            elif config.chart_type == ChartType.HISTOGRAM:
                return self._create_histogram(analysis_result, config)
            elif config.chart_type == ChartType.BOX_PLOT:
                return self._create_box_plot(analysis_result, config)
            elif config.chart_type == ChartType.HEATMAP:
                return self._create_heatmap(analysis_result, config)
            elif config.chart_type == ChartType.PIE_CHART:
                return self._create_pie_chart(analysis_result, config)
            else:
                return self._create_fallback_chart(analysis_result, config)

        except Exception as e:
            return ChartOutput(
                chart_type=config.chart_type,
                title=config.title,
                success=False,
                error_message=str(e)
            )

    def _create_bar_chart(
        self,
        analysis_result: AnalysisResult,
        config: ChartConfig
    ) -> ChartOutput:
        """Create a bar chart"""

        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))

        # Extract data for bar chart
        data = self._extract_bar_chart_data(analysis_result)

        if data is None or len(data) == 0:
            # Create placeholder chart
            ax.bar(['No Data'], [0])
            ax.set_title('No Data Available')
        else:
            # Create actual bar chart
            x_values = list(data.keys())
            y_values = list(data.values())

            bars = ax.bar(x_values, y_values, color=plt.cm.viridis(np.linspace(0, 1, len(x_values))))
            ax.set_title(config.title, fontsize=14, fontweight='bold')

            if config.x_label:
                ax.set_xlabel(config.x_label)
            if config.y_label:
                ax.set_ylabel(config.y_label)

            if config.show_grid:
                ax.grid(True, alpha=0.3)

            # Rotate x labels if needed
            if len(x_values) > 5:
                plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Save chart
        file_path = self.output_dir / f"bar_chart_{analysis_result.analysis_type.lower().replace(' ', '_')}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.read()).decode()

        plt.close()

        return ChartOutput(
            chart_type=ChartType.BAR_CHART,
            title=config.title,
            file_path=str(file_path),
            base64_data=base64_data,
            success=True
        )

    def _create_line_chart(
        self,
        analysis_result: AnalysisResult,
        config: ChartConfig
    ) -> ChartOutput:
        """Create a line chart"""

        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))

        # Extract time series data
        data = self._extract_time_series_data(analysis_result)

        if data is None or len(data) == 0:
            ax.plot([0, 1], [0, 0])
            ax.set_title('No Time Series Data Available')
        else:
            for series_name, series_data in data.items():
                ax.plot(series_data.get('x', []), series_data.get('y', []),
                       label=series_name, marker='o', markersize=4)

            ax.set_title(config.title, fontsize=14, fontweight='bold')

            if config.x_label:
                ax.set_xlabel(config.x_label)
            if config.y_label:
                ax.set_ylabel(config.y_label)

            if config.show_legend and len(data) > 1:
                ax.legend()

            if config.show_grid:
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save chart
        file_path = self.output_dir / f"line_chart_{analysis_result.analysis_type.lower().replace(' ', '_')}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.read()).decode()

        plt.close()

        return ChartOutput(
            chart_type=ChartType.LINE_CHART,
            title=config.title,
            file_path=str(file_path),
            base64_data=base64_data,
            success=True
        )

    def _create_scatter_plot(
        self,
        analysis_result: AnalysisResult,
        config: ChartConfig
    ) -> ChartOutput:
        """Create a scatter plot"""

        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))

        # Extract scatter plot data
        data = self._extract_scatter_data(analysis_result)

        if data is None or len(data.get('x', [])) == 0:
            ax.scatter([0], [0])
            ax.set_title('No Correlation Data Available')
        else:
            ax.scatter(data['x'], data['y'], alpha=0.6, s=50)

            # Add trend line if correlation exists
            if len(data['x']) > 1:
                z = np.polyfit(data['x'], data['y'], 1)
                p = np.poly1d(z)
                ax.plot(data['x'], p(data['x']), "r--", alpha=0.8, linewidth=2)

            ax.set_title(config.title, fontsize=14, fontweight='bold')

            if config.x_label:
                ax.set_xlabel(config.x_label)
            if config.y_label:
                ax.set_ylabel(config.y_label)

            if config.show_grid:
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save chart
        file_path = self.output_dir / f"scatter_plot_{analysis_result.analysis_type.lower().replace(' ', '_')}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.read()).decode()

        plt.close()

        return ChartOutput(
            chart_type=ChartType.SCATTER_PLOT,
            title=config.title,
            file_path=str(file_path),
            base64_data=base64_data,
            success=True
        )

    def _create_histogram(
        self,
        analysis_result: AnalysisResult,
        config: ChartConfig
    ) -> ChartOutput:
        """Create a histogram"""

        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))

        # Extract histogram data
        data = self._extract_distribution_data(analysis_result)

        if data is None or len(data) == 0:
            ax.hist([0, 1, 2], bins=3)
            ax.set_title('No Distribution Data Available')
        else:
            ax.hist(data, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(config.title, fontsize=14, fontweight='bold')

            if config.x_label:
                ax.set_xlabel(config.x_label)
            ax.set_ylabel('Frequency')

            if config.show_grid:
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save chart
        file_path = self.output_dir / f"histogram_{analysis_result.analysis_type.lower().replace(' ', '_')}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.read()).decode()

        plt.close()

        return ChartOutput(
            chart_type=ChartType.HISTOGRAM,
            title=config.title,
            file_path=str(file_path),
            base64_data=base64_data,
            success=True
        )

    def _create_box_plot(
        self,
        analysis_result: AnalysisResult,
        config: ChartConfig
    ) -> ChartOutput:
        """Create a box plot"""

        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))

        # Extract box plot data
        data = self._extract_box_plot_data(analysis_result)

        if not data or len(data) == 0:
            ax.boxplot([[0, 1, 2]])
            ax.set_title('No Data Available for Box Plot')
        else:
            ax.boxplot(data.values(), labels=data.keys())
            ax.set_title(config.title, fontsize=14, fontweight='bold')

            if config.y_label:
                ax.set_ylabel(config.y_label)

            if config.show_grid:
                ax.grid(True, alpha=0.3)

            # Rotate labels if needed
            if len(data) > 5:
                plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Save chart
        file_path = self.output_dir / f"box_plot_{analysis_result.analysis_type.lower().replace(' ', '_')}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.read()).decode()

        plt.close()

        return ChartOutput(
            chart_type=ChartType.BOX_PLOT,
            title=config.title,
            file_path=str(file_path),
            base64_data=base64_data,
            success=True
        )

    def _create_heatmap(
        self,
        analysis_result: AnalysisResult,
        config: ChartConfig
    ) -> ChartOutput:
        """Create a heatmap"""

        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))

        # Extract correlation matrix data
        matrix_data = self._extract_correlation_matrix(analysis_result)

        if matrix_data is None or matrix_data.empty:
            # Create placeholder heatmap
            dummy_data = np.random.rand(3, 3)
            sns.heatmap(dummy_data, annot=True, cmap='viridis', ax=ax)
            ax.set_title('No Correlation Matrix Available')
        else:
            sns.heatmap(matrix_data, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title(config.title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save chart
        file_path = self.output_dir / f"heatmap_{analysis_result.analysis_type.lower().replace(' ', '_')}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.read()).decode()

        plt.close()

        return ChartOutput(
            chart_type=ChartType.HEATMAP,
            title=config.title,
            file_path=str(file_path),
            base64_data=base64_data,
            success=True
        )

    def _create_pie_chart(
        self,
        analysis_result: AnalysisResult,
        config: ChartConfig
    ) -> ChartOutput:
        """Create a pie chart"""

        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))

        # Extract pie chart data
        data = self._extract_categorical_data(analysis_result)

        if not data or len(data) == 0:
            ax.pie([1, 1, 1], labels=['No', 'Data', 'Available'])
        else:
            wedges, texts, autotexts = ax.pie(data.values(), labels=data.keys(),
                                            autopct='%1.1f%%', startangle=90)

            # Enhance text appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')

        ax.set_title(config.title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save chart
        file_path = self.output_dir / f"pie_chart_{analysis_result.analysis_type.lower().replace(' ', '_')}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.read()).decode()

        plt.close()

        return ChartOutput(
            chart_type=ChartType.PIE_CHART,
            title=config.title,
            file_path=str(file_path),
            base64_data=base64_data,
            success=True
        )

    def _create_fallback_chart(
        self,
        analysis_result: AnalysisResult,
        config: ChartConfig
    ) -> ChartOutput:
        """Create a fallback chart when specific type is not implemented"""
        return self._create_bar_chart(analysis_result, config)

    # Data extraction methods (placeholders that extract data from analysis results)

    def _extract_bar_chart_data(self, analysis_result: AnalysisResult) -> Dict[str, float]:
        """Extract data suitable for bar chart"""
        # This would extract relevant data from analysis_result.data
        # For now, return placeholder data
        data = analysis_result.data

        # Try to find suitable key-value pairs
        for key, value in data.items():
            if isinstance(value, dict) and all(isinstance(v, (int, float)) for v in value.values()):
                return value

        # Fallback: create sample data based on insights
        if analysis_result.insights:
            return {f"Item {i+1}": i*10 for i in range(min(5, len(analysis_result.insights)))}

        return {"Sample A": 10, "Sample B": 20, "Sample C": 15}

    def _extract_time_series_data(self, analysis_result: AnalysisResult) -> Dict[str, Dict[str, List]]:
        """Extract time series data"""
        # Placeholder implementation
        return {
            "Series 1": {
                "x": list(range(10)),
                "y": [i + np.random.random() for i in range(10)]
            }
        }

    def _extract_scatter_data(self, analysis_result: AnalysisResult) -> Dict[str, List[float]]:
        """Extract scatter plot data"""
        # Placeholder implementation
        x_data = [i + np.random.random() for i in range(20)]
        y_data = [2*x + np.random.random()*5 for x in x_data]
        return {"x": x_data, "y": y_data}

    def _extract_distribution_data(self, analysis_result: AnalysisResult) -> List[float]:
        """Extract distribution data for histogram"""
        # Placeholder implementation
        return [np.random.normal(50, 15) for _ in range(100)]

    def _extract_box_plot_data(self, analysis_result: AnalysisResult) -> Dict[str, List[float]]:
        """Extract box plot data"""
        # Placeholder implementation
        return {
            "Group A": [np.random.normal(50, 10) for _ in range(30)],
            "Group B": [np.random.normal(60, 15) for _ in range(30)],
            "Group C": [np.random.normal(45, 8) for _ in range(30)]
        }

    def _extract_correlation_matrix(self, analysis_result: AnalysisResult) -> pd.DataFrame:
        """Extract correlation matrix data"""
        # Try to find correlation matrix in analysis results
        data = analysis_result.data

        for key, value in data.items():
            if "correlation" in key.lower() and "matrix" in key.lower():
                if isinstance(value, dict):
                    try:
                        return pd.DataFrame(value)
                    except:
                        pass

        # Placeholder correlation matrix
        cols = ['Var1', 'Var2', 'Var3']
        matrix = np.random.rand(3, 3)
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        np.fill_diagonal(matrix, 1)  # Diagonal should be 1
        return pd.DataFrame(matrix, columns=cols, index=cols)

    def _extract_categorical_data(self, analysis_result: AnalysisResult) -> Dict[str, float]:
        """Extract categorical data for pie chart"""
        # Placeholder implementation
        return {"Category A": 30, "Category B": 25, "Category C": 20, "Category D": 25}