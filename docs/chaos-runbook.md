# samosaChaat — Chaos Testing Runbook

This runbook covers failure scenarios for live defense. Each scenario includes
how to simulate it, how to detect it via Grafana/Loki, and recovery steps.

## Prerequisites

- kubectl configured for the target cluster
- Grafana accessible at https://grafana.samosachaat.art
- Loki datasource configured in Grafana

---

## Scenario 1: Pod Crash / Kill

**Simulate:**
```bash
kubectl delete pod -l app.kubernetes.io/name=chat-api -n samosachaat-prod
```

**Detect (Grafana):**
- Dashboard: Application Performance → look for gap in request rate
- Alert: container restart count spike
- Panel query: `kube_pod_container_status_restarts_total{namespace="samosachaat-prod"}`

**Detect (Loki):**
```logql
{namespace="samosachaat-prod"} | json | level="error"
{namespace="samosachaat-prod",app="chat-api"} | json | message=~".*startup.*|.*shutdown.*"
```

**Recovery:**
- Kubernetes auto-restarts the pod (restartPolicy: Always)
- HPA scales up if CPU threshold exceeded during recovery
- PDB ensures other pods kept running during the kill
- No manual action needed unless crash-looping (check logs for root cause)

**Verify recovered:**
```bash
kubectl get pods -n samosachaat-prod -l app.kubernetes.io/name=chat-api
curl -s https://samosachaat.art/api/health | jq .
```

---

## Scenario 2: Node Failure

**Simulate:**
```bash
# Get a node instance ID
INSTANCE_ID=$(kubectl get nodes -o jsonpath='{.items[0].spec.providerID}' | cut -d/ -f5)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

**Detect (Grafana):**
- Dashboard: Node Health → node disappears from CPU/Memory panels
- Alert: HighCPU or HighMemory may fire on remaining nodes as pods redistribute
- Panel query: `kube_node_status_condition{condition="Ready",status="true"}`

**Detect (Loki):**
```logql
{namespace="kube-system"} | json | message=~".*NotReady.*|.*node.*removed.*"
```

**Recovery:**
- EKS auto-scaling group launches a replacement node (2-5 minutes)
- Pods on the failed node are rescheduled to healthy nodes
- PDBs prevent more than 1 pod per service from being unavailable
- No manual action needed

**Verify recovered:**
```bash
kubectl get nodes    # New node should appear with STATUS Ready
kubectl get pods -n samosachaat-prod -o wide   # All pods Running
```

---

## Scenario 3: Database Connection Pool Exhaustion

**Simulate:**
```bash
# Run a load test that exceeds the connection pool limit
kubectl run loadtest --image=busybox --restart=Never -n samosachaat-prod -- \
  sh -c 'for i in $(seq 1 200); do wget -q -O- http://chat-api:8002/api/health & done; wait'
```

**Detect (Grafana):**
- Dashboard: Application Performance → spike in p99 latency, increase in 5xx errors
- Alert: High5xxRate fires

**Detect (Loki):**
```logql
{app=~"auth|chat-api"} | json | message=~".*connection.*pool.*|.*timeout.*|.*asyncpg.*|.*QueuePool.*overflow.*"
```

**Recovery:**
1. Identify which service is affected from Loki logs
2. Check current connection count: `kubectl exec deploy/chat-api -n samosachaat-prod -- python -c "..."`
3. Restart affected pods: `kubectl rollout restart deploy/chat-api -n samosachaat-prod`
4. If persistent: increase pool size in service config (`SQLALCHEMY_POOL_SIZE` env var) and redeploy
5. Check RDS max_connections: `aws rds describe-db-parameters --db-parameter-group-name default.postgres15`

---

## Scenario 4: Inference Service OOM

**Simulate:**
```bash
# Set a low memory limit and load a large model
kubectl set resources deploy/inference -n samosachaat-prod --limits=memory=512Mi
# Or trigger by sending many concurrent requests
```

**Detect (Grafana):**
- Dashboard: Inference Service → memory spike, then sudden drop (OOM kill)
- Dashboard: Node Health → memory spike on the node hosting inference
- Alert: HighMemory fires

**Detect (Loki):**
```logql
{app="inference"} | json | message=~".*OOMKilled.*|.*memory.*|.*killed.*"
# Also check events:
# kubectl get events -n samosachaat-prod --sort-by='.lastTimestamp' | grep -i oom
```

**Recovery:**
1. Pod auto-restarts (but may crash-loop if model is too large for limit)
2. Check what model is loaded: `curl http://inference:8003/stats`
3. If model too large: swap to smaller model via `POST /models/swap`
4. If limit too low: increase memory limit in values-prod.yaml and `helm upgrade`
5. Restore original limits: `kubectl set resources deploy/inference -n samosachaat-prod --limits=memory=8Gi`

---

## Scenario 5: High Latency / Degraded Performance

**Simulate:**
```bash
# Flood inference with concurrent requests
kubectl run loadtest --image=curlimages/curl --restart=Never -n samosachaat-prod -- \
  sh -c 'for i in $(seq 1 50); do curl -s -X POST http://chat-api:8002/api/conversations/test/messages -H "Content-Type: application/json" -d "{\"content\":\"tell me a story\"}" & done; wait'
```

**Detect (Grafana):**
- Dashboard: Application Performance → p99 latency > 5s
- Dashboard: Inference Service → worker pool utilization at 100%, queue depth growing
- Alert: HighP99Latency fires

**Detect (Loki):**
```logql
{app="chat-api"} | json | inference_time_ms > 5000
{app="inference"} | json | message=~".*queue.*full.*|.*timeout.*|.*worker.*busy.*"
```

**Recovery:**
1. Check inference worker pool: `curl http://inference:8003/stats`
2. If all workers busy: HPA should scale inference pods (check HPA status)
3. Manual scale: `kubectl scale deploy/inference -n samosachaat-prod --replicas=5`
4. If single-pod bottleneck: check if model is too large for CPU inference, consider GPU nodes
5. Verify recovery: watch latency dashboard return to normal

---

## Scenario 6: SSL Certificate Issues

**Detect:**
- Users report "connection not secure" errors
- `curl -vI https://samosachaat.art 2>&1 | grep -i "expire\|ssl\|certificate"`

**Recovery:**
- ACM certificates auto-renew — check ACM console for renewal status
- If DNS validation failed: check Route53 CNAME records match ACM requirements
- `terraform apply` to reconcile if records drifted

---

## Quick Reference: Diagnostic Loki Queries

```logql
# All errors across all services
{namespace="samosachaat-prod"} | json | level="error" | line_format "{{.service}}: {{.message}}"

# Trace a request across services
{namespace="samosachaat-prod"} | json | trace_id="<TRACE_ID>"

# Auth failures
{app="auth"} | json | level="error" | message=~".*oauth.*|.*jwt.*|.*unauthorized.*"

# Inference issues
{app="inference"} | json | message=~".*error.*|.*timeout.*|.*OOM.*|.*worker.*"

# Slow database queries
{app=~"auth|chat-api"} | json | message=~".*slow.*query.*|.*timeout.*"

# Recent pod restarts
{namespace="samosachaat-prod"} | json | message=~".*started.*|.*shutdown.*|.*ready.*"
```
