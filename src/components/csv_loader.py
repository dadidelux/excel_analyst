"""
CSV Data Loader Module

Handles loading, validation, cleaning, and preview of CSV files with support for
various formats and encodings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import chardet
import io
from dataclasses import dataclass

from ..utils.logger import get_logger
from ..utils.exceptions import CSVLoadError

logger = get_logger(__name__)


@dataclass
class CSVMetadata:
    """Metadata about loaded CSV file"""
    filename: str
    shape: Tuple[int, int]
    encoding: str
    delimiter: str
    columns: List[str]
    dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    memory_usage: float


class CSVLoader:
    """
    Comprehensive CSV loader with validation, cleaning, and preview capabilities
    """

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Optional[CSVMetadata] = None
        self.raw_data: Optional[pd.DataFrame] = None

    def detect_encoding(self, file_path: Path, sample_size: int = 10000) -> str:
        """
        Detect file encoding using chardet

        Args:
            file_path: Path to CSV file
            sample_size: Number of bytes to sample for detection

        Returns:
            Detected encoding string
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)

                logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")

                # Fallback to utf-8 if confidence is too low
                if confidence < 0.7:
                    logger.warning(f"Low confidence in encoding detection, using utf-8")
                    encoding = 'utf-8'

                return encoding

        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}. Using utf-8")
            return 'utf-8'

    def detect_delimiter(self, file_path: Path, encoding: str) -> str:
        """
        Detect CSV delimiter by analyzing first few lines

        Args:
            file_path: Path to CSV file
            encoding: File encoding

        Returns:
            Detected delimiter
        """
        common_delimiters = [',', ';', '\t', '|', ':']

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Read first few lines
                lines = [f.readline().strip() for _ in range(5)]
                lines = [line for line in lines if line]  # Remove empty lines

            if not lines:
                return ','

            # Count occurrences of each delimiter
            delimiter_counts = {}
            for delimiter in common_delimiters:
                counts = [line.count(delimiter) for line in lines]
                # Check if delimiter appears consistently
                if len(set(counts)) == 1 and counts[0] > 0:
                    delimiter_counts[delimiter] = counts[0]

            if delimiter_counts:
                # Return delimiter with highest count
                best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                logger.info(f"Detected delimiter: '{best_delimiter}'")
                return best_delimiter

            # Fallback to pandas sniffer
            sample_data = '\n'.join(lines)
            sniffer = pd.io.common.CParserError
            try:
                # Try to let pandas detect
                df_sample = pd.read_csv(io.StringIO(sample_data), nrows=0)
                return ','  # Default if pandas succeeds
            except:
                pass

        except Exception as e:
            logger.warning(f"Delimiter detection failed: {e}")

        return ','  # Default fallback

    def load_csv(
        self,
        file_path: str,
        encoding: Optional[str] = None,
        delimiter: Optional[str] = None,
        **pandas_kwargs
    ) -> pd.DataFrame:
        """
        Load CSV file with automatic encoding and delimiter detection

        Args:
            file_path: Path to CSV file
            encoding: File encoding (auto-detected if None)
            delimiter: CSV delimiter (auto-detected if None)
            **pandas_kwargs: Additional arguments for pd.read_csv

        Returns:
            Loaded DataFrame

        Raises:
            CSVLoadError: If loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise CSVLoadError(f"File not found: {file_path}")

        if not file_path.suffix.lower() in ['.csv', '.txt']:
            logger.warning(f"Unexpected file extension: {file_path.suffix}")

        try:
            # Auto-detect encoding if not provided
            if encoding is None:
                encoding = self.detect_encoding(file_path)

            # Auto-detect delimiter if not provided
            if delimiter is None:
                delimiter = self.detect_delimiter(file_path, encoding)

            # Set default pandas parameters
            default_params = {
                'encoding': encoding,
                'sep': delimiter,
                'engine': 'python',  # More robust for various formats
                'na_values': ['', 'NULL', 'null', 'NaN', 'nan', '#N/A', 'N/A'],
                'keep_default_na': True,
                'skipinitialspace': True
            }

            # Update with user parameters
            default_params.update(pandas_kwargs)

            logger.info(f"Loading CSV: {file_path} with encoding={encoding}, delimiter='{delimiter}'")

            # Load the CSV
            df = pd.read_csv(file_path, **default_params)

            # Store raw data for reference
            self.raw_data = df.copy()

            # Generate metadata
            self.metadata = self._generate_metadata(df, file_path, encoding, delimiter)

            logger.info(f"Successfully loaded CSV with shape {df.shape}")

            return df

        except Exception as e:
            error_msg = f"Failed to load CSV {file_path}: {e}"
            logger.error(error_msg)
            raise CSVLoadError(error_msg) from e

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded CSV data and identify issues

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }

        # Check for empty DataFrame
        if df.empty:
            validation_results['errors'].append("DataFrame is empty")
            validation_results['is_valid'] = False
            return validation_results

        # Check for duplicate column names
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].tolist()
            validation_results['warnings'].append(f"Duplicate column names found: {duplicates}")

        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            validation_results['warnings'].append(f"Completely empty columns: {empty_cols}")
            validation_results['suggestions'].append("Consider removing empty columns")

        # Check for high null percentages
        null_percentages = (df.isnull().sum() / len(df)) * 100
        high_null_cols = null_percentages[null_percentages > 50].index.tolist()
        if high_null_cols:
            validation_results['warnings'].append(f"Columns with >50% null values: {high_null_cols}")

        # Check for potential date columns that weren't parsed
        potential_date_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            sample_values = df[col].dropna().head(10).astype(str)
            if any('/' in str(val) or '-' in str(val) for val in sample_values):
                try:
                    pd.to_datetime(sample_values.iloc[0])
                    potential_date_cols.append(col)
                except:
                    pass

        if potential_date_cols:
            validation_results['suggestions'].append(f"Consider parsing as dates: {potential_date_cols}")

        # Check for numeric columns stored as strings
        potential_numeric_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                # Check if values can be converted to numeric
                try:
                    pd.to_numeric(sample.str.replace(',', ''))
                    potential_numeric_cols.append(col)
                except:
                    pass

        if potential_numeric_cols:
            validation_results['suggestions'].append(f"Consider converting to numeric: {potential_numeric_cols}")

        return validation_results

    def clean_data(self, df: pd.DataFrame, auto_clean: bool = True) -> pd.DataFrame:
        """
        Clean the DataFrame based on common issues

        Args:
            df: DataFrame to clean
            auto_clean: Whether to apply automatic cleaning

        Returns:
            Cleaned DataFrame
        """
        cleaned_df = df.copy()

        if not auto_clean:
            return cleaned_df

        logger.info("Starting data cleaning process")

        # Remove completely empty rows
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(how='all')
        if len(cleaned_df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(cleaned_df)} completely empty rows")

        # Remove completely empty columns
        initial_cols = len(cleaned_df.columns)
        cleaned_df = cleaned_df.dropna(axis=1, how='all')
        if len(cleaned_df.columns) < initial_cols:
            logger.info(f"Removed {initial_cols - len(cleaned_df.columns)} completely empty columns")

        # Clean column names
        original_columns = cleaned_df.columns.tolist()
        cleaned_df.columns = cleaned_df.columns.str.strip()  # Remove whitespace
        cleaned_df.columns = cleaned_df.columns.str.replace(r'[^\w\s]', '', regex=True)  # Remove special chars
        cleaned_df.columns = cleaned_df.columns.str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscores

        if not cleaned_df.columns.equals(pd.Index(original_columns)):
            logger.info("Cleaned column names")

        # Handle duplicate column names
        cols = pd.Series(cleaned_df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        cleaned_df.columns = cols

        # Try to convert string numbers to numeric
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            # Skip if column has mostly non-numeric values
            sample = cleaned_df[col].dropna().head(100)
            if len(sample) == 0:
                continue

            try:
                # Try to convert after removing common formatting
                test_series = sample.astype(str).str.replace(',', '').str.replace('$', '').str.strip()
                numeric_series = pd.to_numeric(test_series, errors='coerce')

                # If more than 80% can be converted, apply to whole column
                if numeric_series.notna().sum() / len(sample) > 0.8:
                    cleaned_df[col] = pd.to_numeric(
                        cleaned_df[col].astype(str).str.replace(',', '').str.replace('$', '').str.strip(),
                        errors='coerce'
                    )
                    logger.info(f"Converted column '{col}' to numeric")
            except:
                pass

        self.data = cleaned_df
        logger.info("Data cleaning completed")

        return cleaned_df

    def get_preview(self, df: Optional[pd.DataFrame] = None, n_rows: int = 10) -> Dict[str, Any]:
        """
        Get a preview of the data with summary statistics

        Args:
            df: DataFrame to preview (uses loaded data if None)
            n_rows: Number of rows to show in preview

        Returns:
            Dictionary containing preview information
        """
        if df is None:
            df = self.data

        if df is None:
            return {"error": "No data loaded"}

        preview = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "head": df.head(n_rows).to_dict('records'),
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }

        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            preview["numeric_summary"] = df[numeric_cols].describe().round(2).to_dict()

        # Add value counts for categorical columns (top 5)
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_summary = {}
        for col in categorical_cols:
            if df[col].nunique() <= 20:  # Only for columns with reasonable number of unique values
                categorical_summary[col] = df[col].value_counts().head(5).to_dict()
        if categorical_summary:
            preview["categorical_summary"] = categorical_summary

        return preview

    def _generate_metadata(self, df: pd.DataFrame, file_path: Path, encoding: str, delimiter: str) -> CSVMetadata:
        """Generate metadata for loaded CSV"""
        return CSVMetadata(
            filename=file_path.name,
            shape=df.shape,
            encoding=encoding,
            delimiter=delimiter,
            columns=df.columns.tolist(),
            dtypes=df.dtypes.astype(str).to_dict(),
            null_counts=df.isnull().sum().to_dict(),
            memory_usage=df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        )

    def load_and_process(
        self,
        file_path: str,
        auto_clean: bool = True,
        **load_kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """
        Complete workflow: load, validate, clean, and preview CSV

        Args:
            file_path: Path to CSV file
            auto_clean: Whether to apply automatic cleaning
            **load_kwargs: Additional arguments for loading

        Returns:
            Tuple of (cleaned_dataframe, validation_results, preview)
        """
        # Load data
        df = self.load_csv(file_path, **load_kwargs)

        # Validate data
        validation_results = self.validate_data(df)

        # Clean data if requested
        if auto_clean:
            df = self.clean_data(df, auto_clean=True)
        else:
            self.data = df

        # Generate preview
        preview = self.get_preview(df)

        return df, validation_results, preview