# Project Amalfi — Productization Design (big-model sharding across many PCs)

Status: design / roadmap. The working v1 prototype (LAN cell) is the seed of this.

## Vision

A peer-to-peer network where **large models are sharded across many contributor PCs**
(none of which could hold the model alone). Consumers submit prompts; a cell of
network-close PCs computes the response via pipeline parallelism and streams it back;
contributors earn **credits** for the compute they provide.

This is the **Petals** design, adapted with an incentive layer. Petals already serves
Llama-70B+ across volunteer machines over the internet — it is the blueprint.

## The load-bearing constraint (measured in v1, do not fight it)

Sharding one model across machines is a **sequential relay**, not parallel work:

| Nodes in relay (7B, LAN) | Single-stream tok/s |
|---|---|
| 1 (Mac alone) | 25.4 |
| 2 (Mac + Mini) | 8.5 |
| 3 (Mac + 2 laptops) | 3.0 |

Each added node/hop adds latency; batching did **not** raise throughput across networked
nodes (0.57–0.74× vs 1.59× on a single machine). Therefore:

> Distributing a big model across consumer PCs buys **access** (run a model none could hold)
> and **aggregate throughput via many cells** — never low latency, never lower cost/token
> than a datacenter. Position on access, openness, privacy, and monetizing idle hardware.

## Architecture

```
Consumer prompt ─► Router (marketplace) ─► a CELL that holds the whole model
                        ▲                     ├ PC1: layers 1–20
                        │                     ├ PC2: layers 21–40   (pipeline)
                        └─ tokens streamed ───┴ PC3: layers 41–60
                     meter tokens · verify · credit each contributor
   Many cells run in parallel  → throughput + redundancy (data-parallel over cells)
```

| Piece | Role | Standard choice |
|---|---|---|
| Layer-block sharding | each PC holds a contiguous block of layers | Petals "blocks"; v1 `plan_split` |
| Geography-aware cells | group network-close PCs so hops stay fast | latency probing → cell formation |
| DHT discovery | nodes announce which blocks they serve | hivemind DHT / libp2p |
| NAT traversal | home PCs can't be dialed directly | libp2p hole-punching/relays; Tailscale for trusted fleets |
| Redundancy + reroute | replicate blocks; reroute instantly on drop (churn is brutal) | Petals block replication; v1 supervisor (re-form) |
| Many cells | N replicas = N× throughput + fault tolerance | data-parallel over cells |
| Metering + verification | pay per verified token/block served; block fraud | signed receipts + TOPLOC |
| Incentive ledger | credits: contribute to earn, spend to use | central ledger first, decentralize later |

## Churn math (why redundancy is mandatory)

A path needs ~all its nodes up. At 95% per-node availability, a 20-node path is up only
~0.95^20 ≈ 36% of the time. So each layer-block must be served by **multiple** nodes, with
instant reroute. This is the single biggest gap between the v1 demo and a real network.

## MVP roadmap

1. **Self-healing cell (v1.1, in progress).** Supervisor auto-forms and re-forms the cell as
   nodes join/drop (`scripts/supervisor.py`). Recovers by re-forming (engine has no per-block
   hot-failover). Fixes the churn/disconnect pain for a LAN/trusted fleet.
2. **Trusted-beta marketplace.** Workers-as-services + Tailscale (NAT/encryption for free) +
   a router that assigns prompts to cells + a simple credit ledger + spot-check verification.
   Known/invited contributors, no token.
3. **Internet-scale, semi-open.** libp2p/DHT discovery, block redundancy + hot reroute
   (adopt or fork Petals), node identity + reputation, TOPLOC verification.
4. **Open / permissionless.** Staked reputation, sybil resistance, incentive economy. Research
   frontier — the "open cell-formation protocol" is a genuine contribution.

## Tech choices
- **Petals** (`bigscience-workshop/petals`) + **hivemind** — study/fork; the internet-scale
  layer-sharding engine with DHT + NAT already solved.
- **libp2p** — P2P networking / NAT / identity substrate.
- **Tailscale / WireGuard** — pragmatic trusted-mesh shortcut for the beta.
- **llama.cpp RPC / exo** — LAN engine (v1); needs an internet/NAT layer to go wide.
- **TOPLOC** — output verification so incentives aren't gamed.
- **Prometheus + Grafana** — metrics; plus the live dashboard (`dashboard/`).

## Honest hard parts
- **Verification of untrusted compute** is the make-or-break once money/credits are involved.
- **Latency compounds** with pipeline length — keep cells short and network-close.
- **Economics don't beat datacenters on price** — pitch access/openness/idle-hardware.
- **Cold-start** — seed supply with your own/anchor nodes before opening to demand.
- **The fully-open version is a research project**, not a feature; ship trusted-beta first.
