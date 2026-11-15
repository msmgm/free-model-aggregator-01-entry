[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/msmgm/free-model-aggregator-01-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/msmgm/free-model-aggregator-01-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-01/reward?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

Built for the [make.fun bounty "The best free AI model aggregator"](https://make.fun/bounty/free-model-aggregator-01).

# Free Model Aggregator

## What it does
A high-performance Node 22 TypeScript server providing OpenAI Chat Completions and Anthropic Messages APIs across ten simulated free-tier model providers, routing incoming requests to the best free model with robust failover and catalog registration.

## How it routes
Implements a quality-ranked min-cost-flow plan with per-provider call caps and receding-horizon replanning. Fails over gracefully, prioritizing unused providers and verified free models matching capability requirements.

## Determinism
Engineered with strict determinism: zero reliance on the wall clock or randomness. Given the same briefing and provider responses, identical decisions and routing paths are guaranteed.
