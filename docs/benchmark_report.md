# Atlas Framework Comparison Benchmark

This report compares the from-scratch ``atlascore`` research pipeline with the ``LangGraph`` implementation.

## Summary

- **queries**: 2
- **atlascore_avg_quality**: 0.7003074959387329
- **langgraph_avg_quality**: 0.7003074959387329
- **atlascore_total_ms**: 87
- **langgraph_total_ms**: 15
- **atlascore_total_cost_usd**: 0.0014
- **langgraph_total_cost_usd**: 0.0014
- **speedup_ratio**: 5.8

## Per-query results

| Query | Atlas cost | LangGraph cost | Atlas ms | LangGraph ms | Atlas quality | LangGraph quality |
|---|---|---|---|---|---|---|
| What is Atlas? | 0.0007 | 0.0007 | 45 | 9 | 0.74 | 0.74 |
| How does Atlas optimize research cost? | 0.0007 | 0.0007 | 42 | 6 | 0.66 | 0.66 |

## Dev-experience comparison

| Dimension | atlascore | LangGraph |
|---|---|---|
| lines_of_code | ~540 | ~120 |
| framework_abstractions | custom | StateGraph |
| debuggability | 9 | 7 |
| extensibility | 8 | 7 |
| learning_curve | 5 | 6 |
