"""
FormatUtils helper for formatting metrics and logs for display.
Provides functions for rounding, unit conversion, and pretty printing.

References:
- Dashboard display requirements
- PEP 8, type hints, and docstring standards
"""
export function formatMetric(value: number, decimals: number = 2): string {
  return value.toFixed(decimals);
}

export function formatLog(log: any): string {
  return `[${log.timestamp}] ${log.level}: ${log.message}`;
}
