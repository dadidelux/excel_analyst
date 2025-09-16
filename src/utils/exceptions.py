class ExcelAgentError(Exception):
    """Base exception for Excel Agent"""
    pass

class CSVLoadError(ExcelAgentError):
    """Raised when CSV loading fails"""
    pass

class QueryParsingError(ExcelAgentError):
    """Raised when query parsing fails"""
    pass

class AnalysisError(ExcelAgentError):
    """Raised when data analysis fails"""
    pass

class VisualizationError(ExcelAgentError):
    """Raised when chart generation fails"""
    pass

class ExcelExportError(ExcelAgentError):
    """Raised when Excel export fails"""
    pass

class ConfigurationError(ExcelAgentError):
    """Raised when configuration is invalid"""
    pass

class APIError(ExcelAgentError):
    """Raised when API calls fail"""
    pass