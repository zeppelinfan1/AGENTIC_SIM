                AGENTIC SIM EXPERIMENT JOB
                         │
                         ▼
                 1. PREPARE RUN
            config / seed / run_id / models
                         │
                         ▼
                2. PREPARE WORLD
          initial population / environment
                         │
                         ▼
                3. RUN SCENARIOS
                  For Each scenario
                         │
                         ▼
              ┌─────────────────────┐
              │   RUN SIMULATION    │
              │                     │
              │  initialize world   │
              │  initialize agents  │
              │  initialize bank    │
              │                     │
              │  ┌───────────────┐  │
              │  │ simulation    │  │
              │  │ loop          │  │
              │  │               │  │
              │  │ agents act    │  │
              │  │      ↓        │  │
              │  │ bank responds │  │
              │  │      ↓        │  │
              │  │ world changes │  │
              │  │      ↓        │  │
              │  │ rewards       │  │
              │  │      ↓        │  │
              │  │ learn         │  │
              │  │      ↺        │  │
              │  └───────────────┘  │
              └─────────────────────┘
                         │
                         ▼
                4. AGGREGATE RESULTS
                         │
                         ▼
                 5. EVALUATE RUN
          emergence / risk / performance
                         │
                         ▼
              6. PUBLISH ARTIFACTS
        Delta / MLflow / metrics / reports