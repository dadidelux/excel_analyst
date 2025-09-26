"""
Query Understanding and Parsing Module

Handles natural language query interpretation, intent recognition, and
conversion to structured analysis requests.
"""

import re
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
import pandas as pd

from ..utils.logger import get_logger
from ..utils.exceptions import QueryParsingError

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of analytical queries supported"""
    COMPARISON = "comparison"
    CORRELATION = "correlation"
    TREND_ANALYSIS = "trend_analysis"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    TOP_N = "top_n"
    SUMMARY_STATS = "summary_stats"
    TIME_SERIES = "time_series"
    CATEGORIZATION = "categorization"
    CUSTOM = "custom"


class ComparisonType(Enum):
    """Types of comparisons"""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL_TO = "="
    BETWEEN = "between"
    TOP = "top"
    BOTTOM = "bottom"


class AggregationType(Enum):
    """Types of aggregations"""
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    STD = "std"
    VARIANCE = "variance"


@dataclass
class QueryIntent:
    """Structured representation of query intent"""
    query_type: QueryType
    primary_columns: List[str]
    secondary_columns: List[str] = None
    aggregation_type: Optional[AggregationType] = None
    comparison_type: Optional[ComparisonType] = None
    filters: Dict[str, Any] = None
    time_column: Optional[str] = None
    group_by_columns: List[str] = None
    limit: Optional[int] = None
    confidence: float = 0.0
    clarifications_needed: List[str] = None


class QueryValidationResult(BaseModel):
    """Result of query validation"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)


class QueryParser:
    """
    Natural Language Query Parser for data analysis requests
    """

    def __init__(self):
        self.column_keywords = {}
        self.data_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.date_columns = []

        # Query pattern definitions
        self.query_patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[QueryType, List[Dict]]:
        """Initialize regex patterns for different query types"""
        patterns = {
            QueryType.COMPARISON: [
                {
                    "pattern": r"compare\s+(.+?)\s+(?:vs|versus|against|with)\s+(.+?)(?:\s+(?:by|across|for)\s+(.+?))?",
                    "groups": ["primary", "secondary", "group_by"]
                },
                {
                    "pattern": r"(?:difference|diff)\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+(?:by|across|for)\s+(.+?))?",
                    "groups": ["primary", "secondary", "group_by"]
                },
                {
                    "pattern": r"(.+?)\s+(?:vs|versus|against)\s+(.+?)(?:\s+comparison)?",
                    "groups": ["primary", "secondary"]
                }
            ],

            QueryType.CORRELATION: [
                {
                    "pattern": r"correlat(?:ion|e)\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+over\s+(.+?))?",
                    "groups": ["primary", "secondary", "time"]
                },
                {
                    "pattern": r"relationship\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+over\s+(.+?))?",
                    "groups": ["primary", "secondary", "time"]
                },
                {
                    "pattern": r"how\s+(?:does|do)\s+(.+?)\s+(?:affect|impact|relate\s+to)\s+(.+?)(?:\s+over\s+(.+?))?",
                    "groups": ["primary", "secondary", "time"]
                }
            ],

            QueryType.TREND_ANALYSIS: [
                {
                    "pattern": r"trend(?:s)?\s+(?:in|of|for)\s+(.+?)(?:\s+over\s+(.+?))?(?:\s+(?:by|across)\s+(.+?))?",
                    "groups": ["primary", "time", "group_by"]
                },
                {
                    "pattern": r"(?:how|what)\s+(?:has|have)\s+(.+?)\s+(?:changed|evolved|trended)\s+over\s+(.+?)(?:\s+(?:by|across)\s+(.+?))?",
                    "groups": ["primary", "time", "group_by"]
                },
                {
                    "pattern": r"(?:seasonal|temporal|time)\s+(?:pattern|trend)s?\s+(?:in|of|for)\s+(.+?)(?:\s+(?:by|across)\s+(.+?))?",
                    "groups": ["primary", "group_by"]
                }
            ],

            QueryType.TOP_N: [
                {
                    "pattern": r"top\s+(\d+)\s+(.+?)(?:\s+(?:by|in|for)\s+(.+?))?",
                    "groups": ["limit", "primary", "secondary"]
                },
                {
                    "pattern": r"(?:best|highest|largest)\s+(.+?)(?:\s+(?:by|in|for)\s+(.+?))?",
                    "groups": ["primary", "secondary"]
                },
                {
                    "pattern": r"(?:worst|lowest|smallest)\s+(.+?)(?:\s+(?:by|in|for)\s+(.+?))?",
                    "groups": ["primary", "secondary"]
                }
            ],

            QueryType.AGGREGATION: [
                {
                    "pattern": r"(?:total|sum|add\s+up)\s+(.+?)(?:\s+(?:by|across|for)\s+(.+?))?",
                    "groups": ["primary", "group_by"]
                },
                {
                    "pattern": r"(?:average|mean)\s+(.+?)(?:\s+(?:by|across|for)\s+(.+?))?",
                    "groups": ["primary", "group_by"]
                },
                {
                    "pattern": r"count\s+(?:of\s+)?(.+?)(?:\s+(?:by|across|for)\s+(.+?))?",
                    "groups": ["primary", "group_by"]
                }
            ],

            QueryType.SUMMARY_STATS: [
                {
                    "pattern": r"(?:summary|statistics|stats)\s+(?:of|for)\s+(.+?)(?:\s+(?:by|across)\s+(.+?))?",
                    "groups": ["primary", "group_by"]
                },
                {
                    "pattern": r"(?:describe|analyze)\s+(.+?)(?:\s+(?:by|across)\s+(.+?))?",
                    "groups": ["primary", "group_by"]
                },
                {
                    "pattern": r"(?:overview|breakdown)\s+(?:of\s+)?(.+?)(?:\s+(?:by|across)\s+(.+?))?",
                    "groups": ["primary", "group_by"]
                }
            ]
        }

        return patterns

    def set_data_context(self, df: pd.DataFrame):
        """
        Set the data context for query parsing and validation

        Args:
            df: DataFrame to analyze for column types and names
        """
        self.data_columns = df.columns.tolist()
        self.numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Detect potential date columns
        self.date_columns = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                self.date_columns.append(col)
            elif col.lower() in ['date', 'time', 'timestamp', 'datetime']:
                self.date_columns.append(col)

        # Build column keyword mapping
        self._build_column_keywords()

        logger.info(f"Data context set: {len(self.data_columns)} columns, "
                   f"{len(self.numeric_columns)} numeric, "
                   f"{len(self.categorical_columns)} categorical, "
                   f"{len(self.date_columns)} date")

    def _build_column_keywords(self):
        """Build mapping of keywords to potential column matches"""
        self.column_keywords = {}

        for col in self.data_columns:
            # Add exact column name
            key = col.lower().replace('_', ' ').replace('-', ' ')
            self.column_keywords[key] = col

            # Add individual words from column name
            words = key.split()
            for word in words:
                if len(word) > 2:  # Skip very short words
                    if word not in self.column_keywords:
                        self.column_keywords[word] = []
                    elif not isinstance(self.column_keywords[word], list):
                        self.column_keywords[word] = [self.column_keywords[word]]

                    if isinstance(self.column_keywords[word], list):
                        if col not in self.column_keywords[word]:
                            self.column_keywords[word].append(col)
                    else:
                        self.column_keywords[word] = col

    def parse_query(self, query: str) -> QueryIntent:
        """
        Parse a natural language query into structured intent

        Args:
            query: Natural language query string

        Returns:
            QueryIntent object representing the parsed query

        Raises:
            QueryParsingError: If query cannot be parsed
        """
        query = query.strip().lower()

        if not query:
            raise QueryParsingError("Empty query provided")

        logger.info(f"Parsing query: '{query}'")

        # Try to match against known patterns
        for query_type, patterns in self.query_patterns.items():
            for pattern_info in patterns:
                match = re.search(pattern_info["pattern"], query, re.IGNORECASE)
                if match:
                    return self._extract_intent_from_match(
                        query_type, match, pattern_info["groups"], query
                    )

        # Fallback: try to extract basic intent
        return self._extract_basic_intent(query)

    def _extract_intent_from_match(
        self,
        query_type: QueryType,
        match: re.Match,
        group_names: List[str],
        original_query: str
    ) -> QueryIntent:
        """Extract structured intent from regex match"""
        intent = QueryIntent(query_type=query_type, primary_columns=[])

        for i, group_name in enumerate(group_names):
            if i < len(match.groups()) and match.group(i + 1):
                value = match.group(i + 1).strip()

                if group_name == "primary":
                    intent.primary_columns = self._resolve_columns(value)
                elif group_name == "secondary":
                    intent.secondary_columns = self._resolve_columns(value)
                elif group_name == "group_by":
                    intent.group_by_columns = self._resolve_columns(value)
                elif group_name == "time":
                    intent.time_column = self._resolve_columns(value)[0] if self._resolve_columns(value) else None
                elif group_name == "limit":
                    try:
                        intent.limit = int(value)
                    except ValueError:
                        pass

        # Set aggregation type based on query type and content
        intent.aggregation_type = self._infer_aggregation_type(original_query, query_type)
        intent.comparison_type = self._infer_comparison_type(original_query, query_type)

        # Calculate confidence based on column matches
        intent.confidence = self._calculate_confidence(intent)

        # Identify needed clarifications
        intent.clarifications_needed = self._identify_clarifications(intent, original_query)

        logger.info(f"Extracted intent: {query_type.value}, confidence: {intent.confidence:.2f}")

        return intent

    def _extract_basic_intent(self, query: str) -> QueryIntent:
        """Extract basic intent when no specific patterns match"""
        # Look for key action words
        if any(word in query for word in ['compare', 'vs', 'versus', 'difference']):
            query_type = QueryType.COMPARISON
        elif any(word in query for word in ['correlat', 'relationship', 'relate']):
            query_type = QueryType.CORRELATION
        elif any(word in query for word in ['trend', 'over time', 'change', 'pattern']):
            query_type = QueryType.TREND_ANALYSIS
        elif any(word in query for word in ['top', 'best', 'highest', 'worst', 'lowest']):
            query_type = QueryType.TOP_N
        elif any(word in query for word in ['total', 'sum', 'average', 'count']):
            query_type = QueryType.AGGREGATION
        elif any(word in query for word in ['summary', 'stats', 'describe', 'overview']):
            query_type = QueryType.SUMMARY_STATS
        else:
            query_type = QueryType.CUSTOM

        # Try to extract column references
        potential_columns = []
        for keyword, columns in self.column_keywords.items():
            if keyword in query:
                if isinstance(columns, list):
                    potential_columns.extend(columns)
                else:
                    potential_columns.append(columns)

        intent = QueryIntent(
            query_type=query_type,
            primary_columns=list(set(potential_columns[:2])),  # Take first 2 unique
            confidence=0.5,  # Lower confidence for basic extraction
            clarifications_needed=["Please specify which columns to analyze"]
        )

        return intent

    def _resolve_columns(self, text: str) -> List[str]:
        """Resolve text references to actual column names"""
        resolved = []
        text = text.lower()

        # Try exact matches first
        if text in self.column_keywords:
            columns = self.column_keywords[text]
            if isinstance(columns, list):
                resolved.extend(columns)
            else:
                resolved.append(columns)

        # Try partial matches
        words = text.replace(',', ' ').replace(' and ', ' ').split()
        for word in words:
            if word in self.column_keywords:
                columns = self.column_keywords[word]
                if isinstance(columns, list):
                    resolved.extend(columns)
                else:
                    resolved.append(columns)

        return list(set(resolved))  # Remove duplicates

    def _infer_aggregation_type(self, query: str, query_type: QueryType) -> Optional[AggregationType]:
        """Infer the type of aggregation needed"""
        query = query.lower()

        if any(word in query for word in ['total', 'sum']):
            return AggregationType.SUM
        elif any(word in query for word in ['average', 'mean']):
            return AggregationType.AVERAGE
        elif 'count' in query:
            return AggregationType.COUNT
        elif any(word in query for word in ['max', 'maximum', 'highest']):
            return AggregationType.MAX
        elif any(word in query for word in ['min', 'minimum', 'lowest']):
            return AggregationType.MIN
        elif 'median' in query:
            return AggregationType.MEDIAN

        return None

    def _infer_comparison_type(self, query: str, query_type: QueryType) -> Optional[ComparisonType]:
        """Infer the type of comparison needed"""
        query = query.lower()

        if any(word in query for word in ['greater', 'more', 'above']):
            return ComparisonType.GREATER_THAN
        elif any(word in query for word in ['less', 'fewer', 'below']):
            return ComparisonType.LESS_THAN
        elif any(word in query for word in ['equal', 'same']):
            return ComparisonType.EQUAL_TO
        elif 'between' in query:
            return ComparisonType.BETWEEN
        elif any(word in query for word in ['top', 'best', 'highest']):
            return ComparisonType.TOP
        elif any(word in query for word in ['bottom', 'worst', 'lowest']):
            return ComparisonType.BOTTOM

        return None

    def _calculate_confidence(self, intent: QueryIntent) -> float:
        """Calculate confidence score for the parsed intent"""
        confidence = 0.0

        # Base confidence for pattern match
        confidence += 0.4

        # Add confidence for resolved columns
        if intent.primary_columns:
            confidence += 0.3
        if intent.secondary_columns:
            confidence += 0.2

        # Add confidence for specific parameters
        if intent.aggregation_type:
            confidence += 0.05
        if intent.comparison_type:
            confidence += 0.05

        return min(confidence, 1.0)

    def _identify_clarifications(self, intent: QueryIntent, query: str) -> List[str]:
        """Identify what clarifications might be needed"""
        clarifications = []

        if not intent.primary_columns:
            clarifications.append("Which columns should be analyzed?")

        if intent.query_type == QueryType.COMPARISON and not intent.secondary_columns:
            clarifications.append("What should be compared against?")

        if intent.query_type in [QueryType.TREND_ANALYSIS, QueryType.TIME_SERIES] and not intent.time_column:
            if self.date_columns:
                clarifications.append(f"Which time column to use? Available: {', '.join(self.date_columns)}")
            else:
                clarifications.append("No time columns found in the data")

        if intent.query_type == QueryType.TOP_N and not intent.limit:
            clarifications.append("How many top results do you want?")

        return clarifications

    def validate_query_intent(self, intent: QueryIntent) -> QueryValidationResult:
        """
        Validate a query intent against the available data

        Args:
            intent: QueryIntent to validate

        Returns:
            QueryValidationResult with validation details
        """
        result = QueryValidationResult(is_valid=True)

        # Check if primary columns exist
        missing_primary = []
        if intent.primary_columns:
            for col in intent.primary_columns:
                if col not in self.data_columns:
                    missing_primary.append(col)

        if missing_primary:
            result.missing_columns.extend(missing_primary)
            result.errors.append(f"Primary columns not found: {', '.join(missing_primary)}")
            result.is_valid = False

        # Check secondary columns
        missing_secondary = []
        if intent.secondary_columns:
            for col in intent.secondary_columns:
                if col not in self.data_columns:
                    missing_secondary.append(col)

        if missing_secondary:
            result.missing_columns.extend(missing_secondary)
            result.errors.append(f"Secondary columns not found: {', '.join(missing_secondary)}")
            result.is_valid = False

        # Check time column
        if intent.time_column and intent.time_column not in self.data_columns:
            result.missing_columns.append(intent.time_column)
            result.errors.append(f"Time column not found: {intent.time_column}")
            result.is_valid = False

        # Type-specific validations
        if intent.query_type in [QueryType.CORRELATION, QueryType.COMPARISON]:
            if intent.primary_columns and all(col in self.categorical_columns for col in intent.primary_columns):
                result.warnings.append("Correlation/comparison with categorical columns may need special handling")

        if intent.query_type == QueryType.AGGREGATION:
            if intent.primary_columns and all(col in self.categorical_columns for col in intent.primary_columns):
                if intent.aggregation_type not in [AggregationType.COUNT]:
                    result.warnings.append("Non-count aggregations on categorical columns may not be meaningful")

        # Generate suggestions
        if not intent.primary_columns:
            if self.numeric_columns:
                result.suggestions.append(f"Consider analyzing these numeric columns: {', '.join(self.numeric_columns[:3])}")

        if intent.query_type == QueryType.TREND_ANALYSIS and not intent.time_column:
            if self.date_columns:
                result.suggestions.append(f"Consider using these date columns: {', '.join(self.date_columns)}")

        return result

    def suggest_follow_up_questions(self, intent: QueryIntent) -> List[str]:
        """
        Suggest follow-up questions based on the current intent

        Args:
            intent: Current query intent

        Returns:
            List of suggested follow-up questions
        """
        suggestions = []

        if intent.query_type == QueryType.COMPARISON and intent.primary_columns:
            suggestions.append(f"What drives the differences in {intent.primary_columns[0]}?")
            if self.categorical_columns:
                suggestions.append(f"How does {intent.primary_columns[0]} vary by {self.categorical_columns[0]}?")

        if intent.query_type == QueryType.CORRELATION and intent.primary_columns and intent.secondary_columns:
            suggestions.append("Are there any outliers affecting this correlation?")
            suggestions.append("What other factors might influence this relationship?")

        if intent.query_type == QueryType.TREND_ANALYSIS and intent.primary_columns:
            suggestions.append("What caused the changes in this trend?")
            suggestions.append("Are there seasonal patterns?")

        if intent.query_type == QueryType.TOP_N:
            suggestions.append("What characteristics do the top performers share?")
            suggestions.append("How much better are they than average?")

        return suggestions[:3]  # Limit to 3 suggestions