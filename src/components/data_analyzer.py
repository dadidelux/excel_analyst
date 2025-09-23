"""
Data Analysis Engine

Performs various analytical operations on datasets based on parsed query intents.
Handles comparisons, correlations, trends, aggregations, and statistical analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

from .query_parser import QueryIntent, QueryType, AggregationType, ComparisonType
from ..utils.logger import get_logger
from ..utils.exceptions import AnalysisError

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """Container for analysis results"""
    analysis_type: str
    summary: str
    data: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    insights: List[str]
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class DataAnalyzer:
    """
    Core data analysis engine that performs various analytical operations
    """

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.analysis_history: List[AnalysisResult] = []

    def set_data(self, df: pd.DataFrame):
        """Set the dataset for analysis"""
        self.df = df.copy()
        logger.info(f"Data set for analysis: {df.shape[0]} rows, {df.shape[1]} columns")

    def analyze(self, intent: QueryIntent) -> AnalysisResult:
        """
        Perform analysis based on query intent

        Args:
            intent: Parsed query intent

        Returns:
            AnalysisResult with findings and visualizations
        """
        if self.df is None:
            raise AnalysisError("No data provided for analysis")

        logger.info(f"Starting analysis: {intent.query_type.value}")

        try:
            # Route to appropriate analysis method
            if intent.query_type == QueryType.COMPARISON:
                result = self._analyze_comparison(intent)
            elif intent.query_type == QueryType.CORRELATION:
                result = self._analyze_correlation(intent)
            elif intent.query_type == QueryType.TREND_ANALYSIS:
                result = self._analyze_trends(intent)
            elif intent.query_type == QueryType.AGGREGATION:
                result = self._analyze_aggregation(intent)
            elif intent.query_type == QueryType.TOP_N:
                result = self._analyze_top_n(intent)
            elif intent.query_type == QueryType.SUMMARY_STATS:
                result = self._analyze_summary_stats(intent)
            elif intent.query_type == QueryType.TIME_SERIES:
                result = self._analyze_time_series(intent)
            else:
                result = self._analyze_custom(intent)

            # Store in history
            self.analysis_history.append(result)
            logger.info(f"Analysis completed successfully: {result.analysis_type}")

            return result

        except Exception as e:
            error_result = AnalysisResult(
                analysis_type=intent.query_type.value,
                summary=f"Analysis failed: {str(e)}",
                data={},
                visualizations=[],
                insights=[],
                metadata={},
                success=False,
                error_message=str(e)
            )
            logger.error(f"Analysis failed: {e}")
            return error_result

    def _analyze_comparison(self, intent: QueryIntent) -> AnalysisResult:
        """Perform comparison analysis between columns or groups"""
        primary_cols = intent.primary_columns
        secondary_cols = intent.secondary_columns or []
        group_by = intent.group_by_columns or []

        if not primary_cols:
            raise AnalysisError("No primary columns specified for comparison")

        results_data = {}
        insights = []
        visualizations = []

        if len(primary_cols) >= 2:
            # Compare multiple columns
            col1, col2 = primary_cols[0], primary_cols[1]

            # Basic statistics
            stats1 = self._get_column_stats(col1)
            stats2 = self._get_column_stats(col2)

            results_data[col1] = stats1
            results_data[col2] = stats2

            # Calculate differences
            if stats1['type'] == 'numeric' and stats2['type'] == 'numeric':
                diff_mean = stats1['mean'] - stats2['mean']
                diff_median = stats1['median'] - stats2['median']

                results_data['differences'] = {
                    'mean_difference': diff_mean,
                    'median_difference': diff_median,
                    'mean_ratio': stats1['mean'] / stats2['mean'] if stats2['mean'] != 0 else None
                }

                insights.append(f"{col1} has a mean of {stats1['mean']:.2f} vs {col2} with {stats2['mean']:.2f}")
                if abs(diff_mean) > 0.1 * max(stats1['mean'], stats2['mean']):
                    insights.append(f"Significant difference in means: {diff_mean:.2f}")

            # Group-by comparison if specified
            if group_by:
                grouped_comparison = self._compare_by_groups(primary_cols, group_by[0])
                results_data['grouped_comparison'] = grouped_comparison

        elif primary_cols and secondary_cols:
            # Compare primary vs secondary
            comparison_data = self._compare_columns(primary_cols[0], secondary_cols[0], group_by)
            results_data['comparison'] = comparison_data

        # Create visualizations
        visualizations.extend(self._create_comparison_visualizations(primary_cols, secondary_cols, group_by))

        summary = self._generate_comparison_summary(results_data, primary_cols, secondary_cols)

        return AnalysisResult(
            analysis_type="Comparison Analysis",
            summary=summary,
            data=results_data,
            visualizations=visualizations,
            insights=insights,
            metadata={
                'columns_analyzed': primary_cols + (secondary_cols or []),
                'group_by': group_by
            }
        )

    def _analyze_correlation(self, intent: QueryIntent) -> AnalysisResult:
        """Perform correlation analysis between variables"""
        primary_cols = intent.primary_columns
        secondary_cols = intent.secondary_columns

        if not primary_cols or len(primary_cols) < 1:
            raise AnalysisError("Need at least one column for correlation analysis")

        results_data = {}
        insights = []
        visualizations = []

        if len(primary_cols) >= 2:
            # Correlation between primary columns
            correlations = self._calculate_correlations(primary_cols)
            results_data['correlations'] = correlations

        elif primary_cols and secondary_cols:
            # Correlation between primary and secondary
            col1, col2 = primary_cols[0], secondary_cols[0]
            correlation_data = self._calculate_correlation_pair(col1, col2)
            results_data['correlation'] = correlation_data

        else:
            # Correlation of one column with all numeric columns
            col = primary_cols[0]
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if col in numeric_cols:
                numeric_cols.remove(col)

            correlations = {}
            for other_col in numeric_cols[:5]:  # Limit to top 5
                corr_data = self._calculate_correlation_pair(col, other_col)
                correlations[other_col] = corr_data

            results_data['correlations_with_all'] = correlations

        # Generate insights
        insights = self._generate_correlation_insights(results_data)

        # Create visualizations
        visualizations = self._create_correlation_visualizations(results_data)

        summary = self._generate_correlation_summary(results_data)

        return AnalysisResult(
            analysis_type="Correlation Analysis",
            summary=summary,
            data=results_data,
            visualizations=visualizations,
            insights=insights,
            metadata={'columns_analyzed': primary_cols + (secondary_cols or [])}
        )

    def _analyze_trends(self, intent: QueryIntent) -> AnalysisResult:
        """Perform trend analysis over time"""
        primary_cols = intent.primary_columns
        time_col = intent.time_column
        group_by = intent.group_by_columns

        if not primary_cols:
            raise AnalysisError("No columns specified for trend analysis")

        # Auto-detect time column if not specified
        if not time_col:
            date_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
            if date_cols:
                time_col = date_cols[0]
            else:
                raise AnalysisError("No time column found for trend analysis")

        results_data = {}
        insights = []
        visualizations = []

        for col in primary_cols:
            if col not in self.df.columns:
                continue

            # Prepare time series data
            ts_data = self.df[[time_col, col]].copy()
            ts_data = ts_data.sort_values(time_col)

            # Calculate trend metrics
            trend_data = self._calculate_trend_metrics(ts_data, time_col, col)
            results_data[col] = trend_data

            # Group by analysis if specified
            if group_by and group_by[0] in self.df.columns:
                grouped_trends = self._analyze_trends_by_group(ts_data, time_col, col, group_by[0])
                results_data[f"{col}_by_{group_by[0]}"] = grouped_trends

        # Generate insights
        insights = self._generate_trend_insights(results_data)

        # Create visualizations
        visualizations = self._create_trend_visualizations(results_data, time_col)

        summary = self._generate_trend_summary(results_data)

        return AnalysisResult(
            analysis_type="Trend Analysis",
            summary=summary,
            data=results_data,
            visualizations=visualizations,
            insights=insights,
            metadata={
                'time_column': time_col,
                'columns_analyzed': primary_cols,
                'group_by': group_by
            }
        )

    def _analyze_aggregation(self, intent: QueryIntent) -> AnalysisResult:
        """Perform aggregation analysis"""
        primary_cols = intent.primary_columns
        group_by = intent.group_by_columns
        agg_type = intent.aggregation_type or AggregationType.SUM

        if not primary_cols:
            raise AnalysisError("No columns specified for aggregation")

        results_data = {}
        insights = []
        visualizations = []

        for col in primary_cols:
            if col not in self.df.columns:
                continue

            if group_by and group_by[0] in self.df.columns:
                # Grouped aggregation
                agg_results = self._perform_grouped_aggregation(col, group_by[0], agg_type)
                results_data[f"{col}_by_{group_by[0]}"] = agg_results
            else:
                # Simple aggregation
                agg_result = self._perform_simple_aggregation(col, agg_type)
                results_data[col] = agg_result

        # Generate insights
        insights = self._generate_aggregation_insights(results_data, agg_type)

        # Create visualizations
        visualizations = self._create_aggregation_visualizations(results_data)

        summary = self._generate_aggregation_summary(results_data, agg_type)

        return AnalysisResult(
            analysis_type="Aggregation Analysis",
            summary=summary,
            data=results_data,
            visualizations=visualizations,
            insights=insights,
            metadata={
                'aggregation_type': agg_type.value,
                'columns_analyzed': primary_cols,
                'group_by': group_by
            }
        )

    def _analyze_top_n(self, intent: QueryIntent) -> AnalysisResult:
        """Perform top-N analysis"""
        primary_cols = intent.primary_columns
        secondary_cols = intent.secondary_columns
        limit = intent.limit or 10
        comparison_type = intent.comparison_type or ComparisonType.TOP

        if not primary_cols:
            raise AnalysisError("No columns specified for top-N analysis")

        results_data = {}
        insights = []
        visualizations = []

        for col in primary_cols:
            if col not in self.df.columns:
                continue

            if secondary_cols and secondary_cols[0] in self.df.columns:
                # Top N by another column
                top_data = self._get_top_n_by_column(col, secondary_cols[0], limit, comparison_type)
            else:
                # Simple top N
                top_data = self._get_simple_top_n(col, limit, comparison_type)

            results_data[col] = top_data

        # Generate insights
        insights = self._generate_top_n_insights(results_data, limit, comparison_type)

        # Create visualizations
        visualizations = self._create_top_n_visualizations(results_data)

        summary = self._generate_top_n_summary(results_data, limit)

        return AnalysisResult(
            analysis_type="Top-N Analysis",
            summary=summary,
            data=results_data,
            visualizations=visualizations,
            insights=insights,
            metadata={
                'limit': limit,
                'comparison_type': comparison_type.value if comparison_type else None
            }
        )

    def _analyze_summary_stats(self, intent: QueryIntent) -> AnalysisResult:
        """Generate summary statistics"""
        primary_cols = intent.primary_columns or self.df.select_dtypes(include=[np.number]).columns.tolist()
        group_by = intent.group_by_columns

        results_data = {}
        insights = []
        visualizations = []

        for col in primary_cols:
            if col not in self.df.columns:
                continue

            stats = self._get_comprehensive_stats(col)
            results_data[col] = stats

            if group_by and group_by[0] in self.df.columns:
                grouped_stats = self._get_grouped_stats(col, group_by[0])
                results_data[f"{col}_by_{group_by[0]}"] = grouped_stats

        # Generate insights
        insights = self._generate_summary_insights(results_data)

        # Create visualizations
        visualizations = self._create_summary_visualizations(results_data)

        summary = self._generate_summary_stats_summary(results_data)

        return AnalysisResult(
            analysis_type="Summary Statistics",
            summary=summary,
            data=results_data,
            visualizations=visualizations,
            insights=insights,
            metadata={'columns_analyzed': primary_cols, 'group_by': group_by}
        )

    def _analyze_time_series(self, intent: QueryIntent) -> AnalysisResult:
        """Perform specialized time series analysis"""
        return self._analyze_trends(intent)  # Delegate to trend analysis for now

    def _analyze_custom(self, intent: QueryIntent) -> AnalysisResult:
        """Handle custom analysis requests"""
        # Basic descriptive analysis as fallback
        return self._analyze_summary_stats(intent)

    # Helper methods for calculations

    def _get_column_stats(self, col: str) -> Dict[str, Any]:
        """Get basic statistics for a column"""
        if col not in self.df.columns:
            return {'type': 'missing', 'error': f'Column {col} not found'}

        series = self.df[col]

        if pd.api.types.is_numeric_dtype(series):
            return {
                'type': 'numeric',
                'count': int(series.count()),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'null_count': int(series.isnull().sum())
            }
        else:
            return {
                'type': 'categorical',
                'count': int(series.count()),
                'unique_count': int(series.nunique()),
                'most_common': series.value_counts().head().to_dict(),
                'null_count': int(series.isnull().sum())
            }

    def _calculate_correlation_pair(self, col1: str, col2: str) -> Dict[str, Any]:
        """Calculate correlation between two columns"""
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {'error': 'One or both columns not found'}

        # Only for numeric columns
        if not (pd.api.types.is_numeric_dtype(self.df[col1]) and pd.api.types.is_numeric_dtype(self.df[col2])):
            return {'error': 'Both columns must be numeric for correlation'}

        # Remove missing values
        clean_data = self.df[[col1, col2]].dropna()

        if len(clean_data) < 2:
            return {'error': 'Insufficient data for correlation'}

        # Pearson correlation
        pearson_r, pearson_p = pearsonr(clean_data[col1], clean_data[col2])

        # Spearman correlation
        spearman_r, spearman_p = spearmanr(clean_data[col1], clean_data[col2])

        return {
            'pearson_correlation': float(pearson_r),
            'pearson_p_value': float(pearson_p),
            'spearman_correlation': float(spearman_r),
            'spearman_p_value': float(spearman_p),
            'sample_size': len(clean_data),
            'strength': self._interpret_correlation_strength(pearson_r)
        }

    def _interpret_correlation_strength(self, r: float) -> str:
        """Interpret correlation strength"""
        abs_r = abs(r)
        if abs_r >= 0.8:
            return 'Very Strong'
        elif abs_r >= 0.6:
            return 'Strong'
        elif abs_r >= 0.4:
            return 'Moderate'
        elif abs_r >= 0.2:
            return 'Weak'
        else:
            return 'Very Weak'

    def _calculate_correlations(self, columns: List[str]) -> Dict[str, Any]:
        """Calculate correlation matrix for multiple columns"""
        numeric_cols = [col for col in columns if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col])]

        if len(numeric_cols) < 2:
            return {'error': 'Need at least 2 numeric columns'}

        corr_matrix = self.df[numeric_cols].corr()

        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strongest_correlations': self._find_strongest_correlations(corr_matrix),
            'columns': numeric_cols
        }

    def _find_strongest_correlations(self, corr_matrix: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """Find strongest correlations in a correlation matrix"""
        correlations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]

                correlations.append({
                    'column1': col1,
                    'column2': col2,
                    'correlation': float(corr_value),
                    'abs_correlation': abs(corr_value),
                    'strength': self._interpret_correlation_strength(corr_value)
                })

        return sorted(correlations, key=lambda x: x['abs_correlation'], reverse=True)[:top_n]

    # Placeholder methods for visualizations and other complex operations
    def _create_comparison_visualizations(self, primary_cols, secondary_cols, group_by):
        return [{'type': 'bar_chart', 'title': 'Comparison Chart', 'data': 'placeholder'}]

    def _create_correlation_visualizations(self, results_data):
        return [{'type': 'scatter_plot', 'title': 'Correlation Plot', 'data': 'placeholder'}]

    def _create_trend_visualizations(self, results_data, time_col):
        return [{'type': 'line_chart', 'title': 'Trend Chart', 'data': 'placeholder'}]

    def _create_aggregation_visualizations(self, results_data):
        return [{'type': 'bar_chart', 'title': 'Aggregation Chart', 'data': 'placeholder'}]

    def _create_top_n_visualizations(self, results_data):
        return [{'type': 'bar_chart', 'title': 'Top N Chart', 'data': 'placeholder'}]

    def _create_summary_visualizations(self, results_data):
        return [{'type': 'histogram', 'title': 'Distribution', 'data': 'placeholder'}]

    # Placeholder methods for summary generation
    def _generate_comparison_summary(self, results_data, primary_cols, secondary_cols):
        return f"Comparison analysis of {', '.join(primary_cols)} completed"

    def _generate_correlation_summary(self, results_data):
        return "Correlation analysis completed"

    def _generate_trend_summary(self, results_data):
        return "Trend analysis completed"

    def _generate_aggregation_summary(self, results_data, agg_type):
        return f"{agg_type.value.title()} aggregation completed"

    def _generate_top_n_summary(self, results_data, limit):
        return f"Top {limit} analysis completed"

    def _generate_summary_stats_summary(self, results_data):
        return "Summary statistics completed"

    # Placeholder methods for insights generation
    def _generate_correlation_insights(self, results_data):
        return ["Correlation analysis insights placeholder"]

    def _generate_trend_insights(self, results_data):
        return ["Trend analysis insights placeholder"]

    def _generate_aggregation_insights(self, results_data, agg_type):
        return ["Aggregation insights placeholder"]

    def _generate_top_n_insights(self, results_data, limit, comparison_type):
        return ["Top-N insights placeholder"]

    def _generate_summary_insights(self, results_data):
        return ["Summary statistics insights placeholder"]

    # Additional helper method placeholders
    def _compare_by_groups(self, primary_cols, group_col):
        return {'placeholder': 'grouped comparison data'}

    def _compare_columns(self, col1, col2, group_by):
        return {'placeholder': 'column comparison data'}

    def _calculate_trend_metrics(self, ts_data, time_col, value_col):
        return {'placeholder': 'trend metrics'}

    def _analyze_trends_by_group(self, ts_data, time_col, value_col, group_col):
        return {'placeholder': 'grouped trend data'}

    def _perform_grouped_aggregation(self, col, group_col, agg_type):
        return {'placeholder': 'grouped aggregation'}

    def _perform_simple_aggregation(self, col, agg_type):
        return {'placeholder': 'simple aggregation'}

    def _get_top_n_by_column(self, col, by_col, limit, comparison_type):
        return {'placeholder': 'top n by column'}

    def _get_simple_top_n(self, col, limit, comparison_type):
        return {'placeholder': 'simple top n'}

    def _get_comprehensive_stats(self, col):
        return self._get_column_stats(col)

    def _get_grouped_stats(self, col, group_col):
        return {'placeholder': 'grouped stats'}