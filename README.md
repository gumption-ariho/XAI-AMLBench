# XAI-AMLBench

> **See the frontend only (no Docker, no backend):**
> * (needs Node 18.18+) `cd frontend && npm install && npm run dev:demo` -> http://localhost:3000 (the real Next.js app on built-in sample data)
> * or just open `frontend/preview/demo.html` in a browser (needs internet once, for React from a CDN)

> **Python packages for local work:** one file, `requirements.txt` in the project root (it includes the four per-service
> files). Install everything fast with `bash install_all_requirements_v2.sh`. The per-service files stay because each Docker
> image installs only its own service's packages.

## Connect everything (website + API + model + narratives) on a normal laptop, no GPU
```bash
.venv/bin/python -m gnn_aml_core.train --data data --out models        # once: the API serves ./models
docker compose up -d                                                    # infrastructure
# build ONE image at a time (kind to a slow connection), smallest first:
docker compose build backend && docker compose build xai-narrative-lite && docker compose build gnn-detection-api && docker compose build frontend
docker compose up -d gnn-detection-api backend xai-narrative-lite frontend node_agent
python3 check_stack_v1.py                                               # tests every link and names the broken one
```
Then open http://localhost. (GPU machine with LLM narratives: use `--profile llm` instead of the lite service, not both.)

## 1. Start the infrastructure
```bash
cp .env.example .env            # edit every change_me value
docker compose config           # validate
docker compose up -d            # traefik, kafka, postgres, redis, neo4j, immudb
```

## 2. Generate data and train the model (in Docker, no local Python needed)
```bash
docker compose --profile ml build
docker compose --profile ml run --rm aml-synth-worker \
    python -m aml_synth.graph_generator --out data --to csv,kafka,neo4j
docker compose --profile ml run --rm gnn-detection-api \
    python -m gnn_aml_core.train --data /app/data --out /app/models --model gatv2
```
(`./data` and `./models` on your machine are mounted into the containers. Add
`- ./data:/app/data` under gnn-detection-api volumes if it is not there yet.)

## 3. Run everything
```bash
docker compose --profile app --profile ml --profile obs up -d --build
# add --profile llm once the GPU / model download is sorted
```

| What                | URL                          |
|---------------------|------------------------------|
| Compliance console  | http://localhost (or :3000)  |
| GNN API docs        | http://localhost/gnn/docs    |
| XAI API docs        | http://localhost/xai/docs    |
| Grafana             | http://localhost/grafana     |
| Neo4j Browser       | http://localhost:7474        |
| immudb console      | http://localhost:8080        |
| Traefik dashboard   | http://localhost:8081        |

## Local (no Docker) quick test of the data + model
```bash
pip install numpy pandas scikit-learn torch torch_geometric
python -m aml_synth.graph_generator --out data --to csv
python -m gnn_aml_core.train --data data --out models
```

## Frontend (TypeScript)
```bash
cd frontend
npm install
npm run typecheck     # tsc --noEmit, strict mode
npm run dev           # http://localhost:3000  (needs the backend for live data)
```
Shared API types live in `frontend/src/types.ts` and mirror the backend JSON.
