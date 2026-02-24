# React Migration Feasibility Assessment

**PGS Catalog Explorer**
*February 2026*

---

## Executive Summary

**Recommendation: Hybrid architecture (FastAPI backend + React frontend)**

A full React migration would require 400-600 hours due to heavy pandas/networkx logic that doesn't port well to JavaScript. The hybrid approach achieves modern frontend benefits in 80-120 hours by keeping Python data processing intact.

---

## Current Technology Stack

| Component | Technology | Lines of Code |
|-----------|------------|---------------|
| UI Framework | Streamlit | 1,632 (app.py) |
| Data Layer | Python/pandas | 671 (data_layer.py) |
| Utilities | Python | 601 (utils.py) |
| Comparison Module | Python/networkx | 406 (compare.py) |
| **Total** | **Python only** | **3,310 lines** |

### Key Dependencies

- **pandas** — DataFrames, filtering, aggregation (~30% of logic)
- **networkx** — Graph algorithms for PGS correlation networks
- **plotly** — All visualizations (7 chart types across tabs)
- **requests** — PGS Catalog API integration with retry logic
- **streamlit** — Session state, caching, UI components

### Frontend/JavaScript Presence

**Zero** — No existing JavaScript, TypeScript, CSS, or build tooling.

---

## Migration Complexity Analysis

### Component-by-Component Assessment

| Component | Complexity | Effort | Notes |
|-----------|:----------:|:------:|-------|
| Tab navigation | Low | 8h | React Router or tab component |
| Form controls | Low | 16h | Filter dropdowns, sliders, checkboxes |
| Statistics tables | Low | 16h | AG-Grid or TanStack Table |
| Plotly charts | Low | 24h | react-plotly.js drop-in |
| Pandas data processing | **VERY HIGH** | 120h | No direct JS equivalent |
| NetworkX graph algorithms | **HIGH** | 80h | Would need graphology or custom |
| API data layer | Medium | 40h | Axios + React Query |
| Caching logic | Medium | 24h | React Query handles most |
| Quality tier classification | Medium | 16h | Business logic rewrite |

### Critical Blockers for Full Migration

1. **Pandas operations** — 200+ lines of DataFrame manipulation
   - Column transformations, groupby, merge operations
   - No JavaScript library offers equivalent ergonomics
   - Danfo.js exists but is immature and incomplete

2. **NetworkX dependency** — Graph analysis for Compare tab
   - Spring layout algorithm
   - Connected components detection
   - Would require graphology.js + significant rewrite

3. **Streamlit caching** — Decorator-based function memoization
   - `@st.cache_data` and `@st.cache_resource` throughout
   - React Query can handle this but requires restructuring

---

## Option A: Full React Migration

### What This Entails

- Rewrite all 3,310 lines of Python to TypeScript
- Port pandas logic to Danfo.js or Arrow.js
- Replace networkx with graphology.js
- Build new API client for PGS Catalog
- Create new test suite

### Effort Estimate

| Phase | Hours | Risk |
|-------|------:|:----:|
| Project setup, routing | 24 | Low |
| UI components (forms, tables) | 80 | Low |
| Data fetching layer | 40 | Medium |
| Pandas logic rewrite | 160 | **High** |
| NetworkX rewrite | 80 | **High** |
| Plotly integration | 40 | Low |
| Testing | 80 | Medium |
| Bug fixing, polish | 80 | Medium |
| **Total** | **584** | **High** |

### Risks

- DataFrame operations may behave differently in JS
- Graph layout algorithms may produce different results
- No existing Python tests to validate equivalence
- 2-3 month timeline with uncertain outcome

### When to Choose This Option

- Team is JavaScript-first with no Python expertise
- Performance requirements demand client-side computation
- Streamlit hosting becomes untenable
- Long-term maintenance by frontend team

---

## Option B: Hybrid Architecture (Recommended)

### Architecture Diagram

```
┌────────────────────────┐      HTTP/JSON      ┌────────────────────────┐
│                        │◄──────────────────►│                         │
│     React Frontend     │                     │     FastAPI Backend     │
│                        │                     │                         │
│  • Tab navigation      │                     │  • /api/scores         │
│  • Filter UI           │                     │  • /api/traits         │
│  • react-plotly.js     │                     │  • /api/comparisons    │
│  • TailwindCSS         │                     │  • /api/network        │
│                        │                     │                         │
└────────────────────────┘                     │  ───────────────────   │
                                               │  Existing Python code: │
                                               │  • pandas operations   │
                                               │  • networkx graphs     │
                                               │  • PGS Catalog API     │
                                               └────────────────────────┘
```

### What This Entails

- Create FastAPI wrapper around existing functions
- Keep ALL pandas/networkx logic unchanged
- Build React frontend for presentation only
- Frontend receives pre-computed data as JSON

### Effort Estimate

| Phase | Hours | Risk |
|-------|------:|:----:|
| FastAPI endpoints | 40 | Low |
| React shell + routing | 20 | Low |
| Component migration | 40 | Low |
| Polish, error handling | 20 | Low |
| **Total** | **120** | **Low** |

### Implementation Phases

**Phase 1: Backend API (40 hours)**
```python
# New: api/main.py
from fastapi import FastAPI
from data_layer import APIDataSource
from compare import load_comparison_data, build_network

app = FastAPI()

@app.get("/api/scores")
def list_scores(trait: str = None, quality_tier: str = None):
    # Wrap existing data_layer logic
    ...

@app.get("/api/comparisons")
def get_comparisons(trait: str = None, min_correlation: float = -1):
    stats_df, _ = load_comparison_data()
    # Filter and return as JSON
    ...

@app.get("/api/network/{trait}")
def get_network(trait: str, threshold: float = 0.5):
    G = build_network(stats_df, trait, threshold=threshold)
    # Serialize graph to JSON
    ...
```

**Phase 2: React Shell (20 hours)**
```tsx
// src/App.tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <TabNavigation />
        <Routes>
          <Route path="/" element={<ScoresTab />} />
          <Route path="/traits" element={<TraitsTab />} />
          <Route path="/compare" element={<CompareTab />} />
          {/* ... */}
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
```

**Phase 3: Component Migration (40 hours)**
- Port filter controls to React components
- Integrate react-plotly.js for charts
- Implement data tables with sorting/filtering
- Connect to FastAPI endpoints via React Query

**Phase 4: Polish (20 hours)**
- Loading states and error boundaries
- Responsive design with Tailwind
- Deployment configuration

### Benefits

- **Minimal risk** — Python logic unchanged and tested
- **Faster iteration** — React tooling for UI changes
- **Better UX** — Smoother interactions than Streamlit
- **Maintainable** — Clear separation of concerns
- **Testable** — Frontend and backend can be tested independently

### Deployment Options

1. **Single container** — FastAPI serves React static build
2. **Separate containers** — Scale frontend/backend independently
3. **Serverless** — FastAPI on Cloud Run, React on Vercel/Netlify

---

## Decision Framework

Choose **Full Migration** if:
- [ ] Team is exclusively JavaScript/TypeScript
- [ ] Streamlit is a hard blocker (not the case here)
- [ ] Long runway (3+ months) available
- [ ] Client-side computation is required

Choose **Hybrid** (recommended) if:
- [x] Team has Python expertise
- [x] Data processing logic is complex
- [x] Want faster time to value
- [x] Plan to iterate on UI frequently

Choose **Stay with Streamlit** if:
- [ ] Current UX is acceptable
- [ ] No frontend expertise available
- [ ] Minimal UI changes anticipated

---

## Recommended Next Steps

If proceeding with hybrid approach:

1. **Create FastAPI project structure** — Separate `api/` directory
2. **Define API contracts** — OpenAPI spec for all endpoints
3. **Spike react-plotly.js** — Verify Plotly Python charts translate
4. **Implement incrementally** — One tab at a time

---

## Appendix: Technology Choices

### Frontend Stack (Hybrid Option)

| Category | Recommendation | Rationale |
|----------|----------------|-----------|
| Framework | React 18+ | Industry standard, good ecosystem |
| Language | TypeScript | Type safety, better DX |
| Styling | Tailwind CSS | Rapid development, consistent design |
| Data Fetching | TanStack Query | Caching, loading states built-in |
| Charts | react-plotly.js | Direct port from Python Plotly |
| Tables | TanStack Table | Flexible, performant |
| Routing | React Router v6 | Standard, well-documented |
| Build Tool | Vite | Fast builds, good DX |

### Backend Stack (Hybrid Option)

| Category | Recommendation | Rationale |
|----------|----------------|-----------|
| Framework | FastAPI | Modern, async, auto-docs |
| Serialization | Pydantic v2 | FastAPI native, fast |
| CORS | fastapi.middleware.cors | Required for separate frontend |
