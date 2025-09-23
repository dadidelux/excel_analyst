"""
Prompt Templates for LLM-Enhanced Query Processing

Contains structured prompts for query interpretation, validation,
and analysis explanation generation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

from .query_parser import QueryType, QueryIntent, AggregationType, ComparisonType


@dataclass
class PromptTemplate:
    """Base class for prompt templates"""
    system_prompt: str
    user_prompt_template: str
    expected_output_format: str


class QueryPrompts:
    """
    Collection of prompt templates for query processing
    """

    @staticmethod
    def get_query_interpretation_prompt() -> PromptTemplate:
        """
        Prompt template for interpreting natural language queries
        """
        system_prompt = """You are a data analysis assistant that interprets natural language queries about datasets. Your job is to understand what the user wants to analyze and provide structured information about their request.

You should:
1. Identify the type of analysis requested
2. Determine which columns/fields are referenced
3. Understand any specific metrics or comparisons needed
4. Recognize time-based analysis requirements
5. Identify grouping or filtering criteria

Be precise and ask for clarification when the request is ambiguous."""

        user_prompt_template = """Dataset columns available: {columns}
Column types:
- Numeric: {numeric_columns}
- Categorical: {categorical_columns}
- Date/Time: {date_columns}

User query: "{query}"

Please analyze this query and provide:

1. Analysis Type: What kind of analysis is being requested?
2. Primary Columns: Which columns are the main focus?
3. Secondary Columns: Any columns for comparison or grouping?
4. Metrics: What calculations or aggregations are needed?
5. Filters: Any conditions or constraints mentioned?
6. Time Dimension: Is time-based analysis needed?
7. Clarifications: What additional information would help provide better results?

Respond in a structured format that clearly addresses each point."""

        expected_output_format = """Analysis Type: [comparison/trend/correlation/aggregation/etc.]
Primary Columns: [list of main columns to analyze]
Secondary Columns: [columns for comparison/grouping]
Metrics: [specific calculations needed]
Filters: [any conditions mentioned]
Time Dimension: [time column if time-based analysis]
Clarifications: [questions to ask for better analysis]"""

        return PromptTemplate(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            expected_output_format=expected_output_format
        )

    @staticmethod
    def get_analysis_explanation_prompt() -> PromptTemplate:
        """
        Prompt template for generating analysis explanations
        """
        system_prompt = """You are a data analyst who explains analysis results in clear, business-friendly language. You help users understand what the data shows, why it matters, and what actions they might consider.

Your explanations should:
1. Start with the key findings
2. Provide context about what the numbers mean
3. Highlight important patterns or insights
4. Suggest potential implications or actions
5. Use clear, non-technical language
6. Structure information logically

Avoid jargon and focus on practical insights."""

        user_prompt_template = """Analysis performed: {analysis_type}
Data analyzed: {columns_analyzed}
Key results: {results_summary}

Data context:
{data_context}

Results details:
{detailed_results}

Please provide a clear explanation of these results that includes:

1. Executive Summary: What are the main takeaways?
2. Key Insights: What important patterns or findings emerged?
3. Context: What do these numbers mean in practical terms?
4. Implications: What might this suggest for decision-making?
5. Limitations: What should be considered when interpreting these results?
6. Next Steps: What additional analysis might be helpful?

Make this accessible to business users who may not have technical backgrounds."""

        expected_output_format = """## Executive Summary
[Main takeaways in 2-3 sentences]

## Key Insights
[Bullet points of important findings]

## Context
[What the numbers mean practically]

## Implications
[What this suggests for decisions]

## Limitations
[Important caveats]

## Suggested Next Steps
[Additional analysis recommendations]"""

        return PromptTemplate(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            expected_output_format=expected_output_format
        )

    @staticmethod
    def get_query_refinement_prompt() -> PromptTemplate:
        """
        Prompt template for refining unclear or ambiguous queries
        """
        system_prompt = """You are helping users refine their data analysis questions to get better results. When a query is unclear, incomplete, or could be interpreted multiple ways, you help clarify the intent and suggest improvements.

Your role is to:
1. Identify ambiguities or missing information
2. Suggest specific clarifying questions
3. Offer alternative interpretations
4. Recommend more precise wording
5. Help users think about what they really want to learn

Be helpful and educational, not just critical."""

        user_prompt_template = """Original query: "{original_query}"

Available data columns: {available_columns}

Current interpretation:
- Analysis type: {interpreted_type}
- Columns identified: {identified_columns}
- Missing information: {missing_info}

Issues identified:
{issues}

Please help refine this query by:

1. Clarifying Questions: What specific questions should we ask the user?
2. Alternative Interpretations: What other ways could this query be understood?
3. Suggested Refinements: How could the query be reworded for clarity?
4. Additional Context Needed: What information would improve the analysis?
5. Related Questions: What related questions might also be valuable?

Focus on helping the user get actionable insights from their data."""

        expected_output_format = """## Clarifying Questions
[Specific questions to ask the user]

## Alternative Interpretations
[Different ways to understand the query]

## Suggested Refinements
[How to reword for clarity]

## Additional Context Needed
[Information that would help]

## Related Questions
[Other valuable questions to consider]"""

        return PromptTemplate(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            expected_output_format=expected_output_format
        )

    @staticmethod
    def get_column_mapping_prompt() -> PromptTemplate:
        """
        Prompt template for mapping user references to actual column names
        """
        system_prompt = """You help map user references to actual column names in datasets. Users often refer to columns using natural language or partial names, and you need to identify which actual columns they mean.

Consider:
1. Exact matches (preferred)
2. Partial matches
3. Semantic similarity
4. Common abbreviations
5. Business terminology

Be conservative - only suggest matches you're confident about."""

        user_prompt_template = """User mentioned: "{user_reference}"

Available columns:
{column_list}

Column details:
{column_details}

Please identify which column(s) the user is likely referring to:

1. Best Matches: Most likely column(s) with confidence level
2. Possible Matches: Other potential matches with lower confidence
3. No Match: If no reasonable match exists
4. Clarification Needed: If more information would help

For each match, explain why it's a good fit."""

        expected_output_format = """Best Matches:
- [Column Name] (confidence: high/medium/low) - [explanation]

Possible Matches:
- [Column Name] (confidence: low) - [explanation]

Clarification Needed:
[Questions to ask if matches are uncertain]"""

        return PromptTemplate(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            expected_output_format=expected_output_format
        )

    @staticmethod
    def get_data_story_prompt() -> PromptTemplate:
        """
        Prompt template for generating data stories and insights
        """
        system_prompt = """You are a data storyteller who transforms analysis results into compelling narratives. You help users understand not just what the data shows, but why it matters and what story it tells.

Your stories should:
1. Have a clear narrative arc
2. Connect data points into meaningful insights
3. Provide business context
4. Suggest implications and actions
5. Be engaging and memorable
6. Balance detail with accessibility

Think like a consultant presenting insights to stakeholders."""

        user_prompt_template = """Analysis Results:
{analysis_results}

Business Context:
{business_context}

Key Metrics:
{key_metrics}

Trends Identified:
{trends}

Comparisons Made:
{comparisons}

Create a data story that includes:

1. The Big Picture: What's the overall story these numbers tell?
2. Key Characters: What are the main drivers/factors at play?
3. Plot Points: What are the most interesting findings?
4. Turning Points: Where do you see significant changes or patterns?
5. The Resolution: What conclusions can we draw?
6. What's Next: What questions does this raise for further investigation?

Make this engaging and actionable for business decision-makers."""

        expected_output_format = """# The Data Story

## The Big Picture
[Overall narrative of what the data reveals]

## Key Findings
[Most important insights with supporting data]

## What's Driving These Results
[Underlying factors and relationships]

## Critical Turning Points
[Significant changes or inflection points]

## What This Means
[Business implications and significance]

## Questions for Further Investigation
[What to explore next]"""

        return PromptTemplate(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            expected_output_format=expected_output_format
        )


class PromptFormatter:
    """
    Utility class for formatting prompts with context data
    """

    @staticmethod
    def format_query_interpretation(
        query: str,
        columns: List[str],
        numeric_columns: List[str],
        categorical_columns: List[str],
        date_columns: List[str]
    ) -> Dict[str, str]:
        """
        Format the query interpretation prompt with data context
        """
        template = QueryPrompts.get_query_interpretation_prompt()

        formatted_user_prompt = template.user_prompt_template.format(
            query=query,
            columns=", ".join(columns),
            numeric_columns=", ".join(numeric_columns) if numeric_columns else "None",
            categorical_columns=", ".join(categorical_columns) if categorical_columns else "None",
            date_columns=", ".join(date_columns) if date_columns else "None"
        )

        return {
            "system": template.system_prompt,
            "user": formatted_user_prompt,
            "expected_format": template.expected_output_format
        }

    @staticmethod
    def format_analysis_explanation(
        analysis_type: str,
        columns_analyzed: List[str],
        results_summary: str,
        detailed_results: str,
        data_context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Format the analysis explanation prompt
        """
        template = QueryPrompts.get_analysis_explanation_prompt()

        formatted_user_prompt = template.user_prompt_template.format(
            analysis_type=analysis_type,
            columns_analyzed=", ".join(columns_analyzed),
            results_summary=results_summary,
            detailed_results=detailed_results,
            data_context=data_context or "No additional context provided"
        )

        return {
            "system": template.system_prompt,
            "user": formatted_user_prompt,
            "expected_format": template.expected_output_format
        }

    @staticmethod
    def format_query_refinement(
        original_query: str,
        available_columns: List[str],
        interpreted_type: str,
        identified_columns: List[str],
        missing_info: List[str],
        issues: List[str]
    ) -> Dict[str, str]:
        """
        Format the query refinement prompt
        """
        template = QueryPrompts.get_query_refinement_prompt()

        formatted_user_prompt = template.user_prompt_template.format(
            original_query=original_query,
            available_columns=", ".join(available_columns),
            interpreted_type=interpreted_type,
            identified_columns=", ".join(identified_columns) if identified_columns else "None identified",
            missing_info="\n".join(f"- {item}" for item in missing_info),
            issues="\n".join(f"- {issue}" for issue in issues)
        )

        return {
            "system": template.system_prompt,
            "user": formatted_user_prompt,
            "expected_format": template.expected_output_format
        }

    @staticmethod
    def format_column_mapping(
        user_reference: str,
        column_list: List[str],
        column_details: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Format the column mapping prompt
        """
        template = QueryPrompts.get_column_mapping_prompt()

        # Format column details if provided
        if column_details:
            details_str = "\n".join([f"- {col}: {details}" for col, details in column_details.items()])
        else:
            details_str = "No additional column details provided"

        formatted_user_prompt = template.user_prompt_template.format(
            user_reference=user_reference,
            column_list="\n".join(f"- {col}" for col in column_list),
            column_details=details_str
        )

        return {
            "system": template.system_prompt,
            "user": formatted_user_prompt,
            "expected_format": template.expected_output_format
        }

    @staticmethod
    def format_data_story(
        analysis_results: str,
        business_context: Optional[str] = None,
        key_metrics: Optional[Dict[str, Any]] = None,
        trends: Optional[List[str]] = None,
        comparisons: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Format the data story prompt
        """
        template = QueryPrompts.get_data_story_prompt()

        # Format optional parameters
        context_str = business_context or "General business analysis"
        metrics_str = json.dumps(key_metrics, indent=2) if key_metrics else "No specific metrics provided"
        trends_str = "\n".join(f"- {trend}" for trend in trends) if trends else "No specific trends identified"
        comparisons_str = "\n".join(f"- {comp}" for comp in comparisons) if comparisons else "No comparisons made"

        formatted_user_prompt = template.user_prompt_template.format(
            analysis_results=analysis_results,
            business_context=context_str,
            key_metrics=metrics_str,
            trends=trends_str,
            comparisons=comparisons_str
        )

        return {
            "system": template.system_prompt,
            "user": formatted_user_prompt,
            "expected_format": template.expected_output_format
        }