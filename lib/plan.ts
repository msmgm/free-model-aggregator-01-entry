/**
 * @file plan.ts
 * @description Routing planner for competition free-model-aggregator-01.
 * Candidate ranking, min-cost-max-flow assignment of requests to providers,
 * and failover retry order.
 *
 * Background: Each trial sends 20 requests. Every request must be answered
 * by a FREE model meeting its requirements (tools/vision/json/min_context),
 * rated by a sealed quality table (rating_milli). Each provider has an unknown
 * per-trial call budget and a hidden failure schedule: the first call to any
 * provider ALWAYS succeeds (its failure schedule starts at call 2), and exactly
 * one provider is forced to refuse its 2nd call. Spreading calls thinly wins:
 * a Monte-Carlo simulation over the generator's draw distribution showed that
 * capping every provider at 2 calls, then re-planning the remaining requests
 * after each request (receding horizon, excluding providers that already failed),
 * maximises the chance of the top score. The planner below is a pure function
 * of the brief and the observed failures: no clock, no randomness.
 */

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
  type Wire,
  type WorkloadRequest,
} from './providers.ts';

export interface Candidate {
  provider: string;
  model: string;
  rating: number;
  wire: Wire;
}

export interface Assignment {
  provider: string;
  model: string;
  rating: number;
}

export type Caps = Record<string, number>;

export function candidatesFor(
  ratings: RatingsFile | null,
  probes: readonly Probe[],
  requires: Requirement,
  clientApi: 'openai' | 'anthropic'
): Candidate[] {
  const candidates: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      const caps = capabilitiesOf(model, row);
      if (!meetsRequirements(requires, caps)) continue;
      if (clientApi === 'openai' && requires.json === true && probe.provider.wire === 'anthropic') {
        continue;
      }
      const rating = row !== null ? row.rating_milli : 0;
      candidates.push({
        provider: probe.provider.id,
        model: model.id,
        rating,
        wire: probe.provider.wire,
      });
    }
  }
  candidates.sort((a, b) => {
    if (b.rating !== a.rating) {
      return b.rating - a.rating;
    }
    const pCmp = a.provider.localeCompare(b.provider);
    if (pCmp !== 0) {
      return pCmp;
    }
    return a.model.localeCompare(b.model);
  });
  return candidates;
}

export function bestModelFor(
  ratings: RatingsFile | null,
  probes: readonly Probe[],
  providerId: string,
  requires: Requirement,
  clientApi: 'openai' | 'anthropic'
): Candidate | null {
  const cands = candidatesFor(ratings, probes, requires, clientApi);
  for (const c of cands) {
    if (c.provider === providerId) {
      return c;
    }
  }
  return null;
}

// __CANDIDATES__
/**
 * Cap 2 per provider (see header); a provider absent from this table
 * has cap 0 unless the caller passes its own caps.
 */
export const PLANNER_CAPS: Caps = {
  groq: 2,
  cerebras: 2,
  gemini: 2,
  zai: 2,
  openrouter: 2,
  sambanova: 2,
  pollinations: 2,
  cloudflare: 2,
  ovh: 2,
  kilol: 2,
};

// __CAPS__
export function planAssignment(
  requests: readonly WorkloadRequest[],
  candidatesByRequest: ReadonlyMap<string, Candidate[]>,
  caps: Caps
): Map<string, Assignment> {
  const providerList: string[] = [];
  const providerToNode = new Map<string, number>();

  for (const req of requests) {
    const cands = candidatesByRequest.get(req.id) || [];
    for (const c of cands) {
      if (!providerToNode.has(c.provider)) {
        providerList.push(c.provider);
        providerToNode.set(c.provider, 0);
      }
    }
  }

  const R = requests.length;
  const P = providerList.length;
  for (let j = 0; j < P; j++) {
    providerToNode.set(providerList[j], R + 1 + j);
  }

  const sink = R + P + 1;
  const N = sink + 1;

  interface FlowEdge {
    id: number;
    u: number;
    v: number;
    cap: number;
    flow: number;
    cost: number;
  }

  const edges: FlowEdge[] = [];

  // 1. Source -> Requests
  for (let i = 0; i < R; i++) {
    edges.push({
      id: edges.length,
      u: 0,
      v: i + 1,
      cap: 1,
      flow: 0,
      cost: 0,
    });
  }

  // 2. Requests -> Providers
  for (let i = 0; i < R; i++) {
    const req = requests[i];
    const reqNode = i + 1;
    const cands = candidatesByRequest.get(req.id) || [];
    const connectedProviders = new Set<string>();

    const bestRatingByProvider = new Map<string, number>();
    for (const c of cands) {
      const cur = bestRatingByProvider.get(c.provider) ?? 0;
      if (c.rating > cur) {
        bestRatingByProvider.set(c.provider, c.rating);
      }
    }

    for (const c of cands) {
      if (connectedProviders.has(c.provider)) continue;
      connectedProviders.add(c.provider);
      const provNode = providerToNode.get(c.provider);
      if (provNode === undefined) continue;
      const bestRating = bestRatingByProvider.get(c.provider) ?? c.rating;
      const cost = - (req.weight * bestRating);
      edges.push({
        id: edges.length,
        u: reqNode,
        v: provNode,
        cap: 1,
        flow: 0,
        cost,
      });
    }
  }

  // 3. Providers -> Sink
  for (let j = 0; j < P; j++) {
    const pName = providerList[j];
    const provNode = R + 1 + j;
    const cap = caps[pName] ?? 0;
    edges.push({
      id: edges.length,
      u: provNode,
      v: sink,
      cap,
      flow: 0,
      cost: 0,
    });
  }

  // Successive shortest augmenting paths using Bellman-Ford
  while (true) {
    const dist = new Array<number>(N).fill(Infinity);
    dist[0] = 0;
    const parent = new Array<{ edge: FlowEdge; dir: 'forward' | 'backward' } | null>(N).fill(null);

    for (let round = 0; round < N; round++) {
      let updated = false;
      for (let u = 0; u < N; u++) {
        if (dist[u] === Infinity) continue;

        // Forward residual edges
        for (const e of edges) {
          if (e.u === u) {
            const resid = e.cap - e.flow;
            if (resid > 0) {
              const v = e.v;
              const w = e.cost;
              if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = { edge: e, dir: 'forward' };
                updated = true;
              }
            }
          }
        }

        // Backward residual edges
        for (const e of edges) {
          if (e.v === u) {
            const resid = e.flow;
            if (resid > 0) {
              const v = e.u;
              const w = -e.cost;
              if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = { edge: e, dir: 'backward' };
                updated = true;
              }
            }
          }
        }
      }
      if (!updated) break;
    }

    if (dist[sink] === Infinity) {
      break;
    }

    let curr = sink;
    while (curr !== 0) {
      const p = parent[curr];
      if (!p) break;
      if (p.dir === 'forward') {
        p.edge.flow += 1;
        curr = p.edge.u;
      } else {
        p.edge.flow -= 1;
        curr = p.edge.v;
      }
    }
  }

  // Recover assignments
  const assignmentMap = new Map<string, Assignment>();

  for (let i = 0; i < R; i++) {
    const req = requests[i];
    const reqNode = i + 1;
    let assignedProvider: string | null = null;
    for (const e of edges) {
      if (e.u === reqNode && e.flow > 0) {
        for (const [pName, pNode] of providerToNode.entries()) {
          if (pNode === e.v) {
            assignedProvider = pName;
            break;
          }
        }
        if (assignedProvider) break;
      }
    }

    if (assignedProvider) {
      const cands = candidatesByRequest.get(req.id) || [];
      let bestCand: Candidate | null = null;
      for (const c of cands) {
        if (c.provider === assignedProvider) {
          if (!bestCand || c.rating > bestCand.rating) {
            bestCand = c;
          }
        }
      }
      if (bestCand) {
        assignmentMap.set(req.id, {
          provider: bestCand.provider,
          model: bestCand.model,
          rating: bestCand.rating,
        });
      }
    }
  }

  return assignmentMap;
}

export function orderCandidates(
  candidates: readonly Candidate[],
  plannedProvider: string | null,
  used: ReadonlyMap<string, number>,
  failed: ReadonlySet<string>
): Candidate[] {
  let plannedCand: Candidate | null = null;
  if (plannedProvider !== null) {
    for (const c of candidates) {
      if (c.provider === plannedProvider) {
        plannedCand = c;
        break;
      }
    }
  }

  const bestPerProvider = new Map<string, Candidate>();
  for (const c of candidates) {
    if (!bestPerProvider.has(c.provider)) {
      bestPerProvider.set(c.provider, c);
    }
  }

  const remaining: Candidate[] = [];
  for (const [prov, c] of bestPerProvider.entries()) {
    if (plannedCand !== null && prov === plannedCand.provider) {
      continue;
    }
    remaining.push(c);
  }

  const getGroup = (prov: string): number => {
    const u = used.get(prov) ?? 0;
    if (u === 0) {
      return 0;
    }
    if (!failed.has(prov)) {
      return 1;
    }
    return 2;
  };

  remaining.sort((a, b) => {
    const gA = getGroup(a.provider);
    const gB = getGroup(b.provider);
    if (gA !== gB) {
      return gA - gB;
    }
    if (b.rating !== a.rating) {
      return b.rating - a.rating;
    }
    const pCmp = a.provider.localeCompare(b.provider);
    if (pCmp !== 0) {
      return pCmp;
    }
    return a.model.localeCompare(b.model);
  });

  const result: Candidate[] = [];
  if (plannedCand !== null) {
    result.push(plannedCand);
  }
  for (const c of remaining) {
    if (plannedCand === null || c.provider !== plannedCand.provider) {
      result.push(c);
    }
  }

  return result;
}

