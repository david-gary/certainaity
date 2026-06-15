# Kubernetes Deployment

Kubernetes manifests live in `k8s/`. They target a single-namespace deployment
suitable for a self-hosted GPU cluster or a cloud provider with GPU node pools
(GKE, EKS, AKS).

## Namespace

```bash
kubectl apply -f k8s/namespace.yaml
kubectl config set-context --current --namespace=certainaity
```

## Secrets

```bash
# JWT public key
kubectl create secret generic certainaity-jwt \
  --from-file=jwt_public.pem=secrets/jwt_public.pem

# Redis password (if using authenticated Redis)
kubectl create secret generic certainaity-redis \
  --from-literal=password=<REDIS_PASSWORD>
```

## Deploy

```bash
kubectl apply -f k8s/
```

This deploys:

| Resource | Description |
|----------|-------------|
| `Deployment/certainaity-api` | 2 replicas, CPU, port 8000 |
| `Deployment/certainaity-worker` | 1 replica, GPU (1× NVIDIA), Celery |
| `Deployment/certainaity-frontend` | 2 replicas, nginx, port 80 |
| `Service/certainaity-api` | ClusterIP :8000 |
| `Service/certainaity-frontend` | ClusterIP :80 |
| `HorizontalPodAutoscaler/certainaity-api` | Scale 2–10 on CPU > 70% |
| `ConfigMap/certainaity-config` | Non-secret environment variables |

## Ingress

Apply your ingress controller's ingress resource. Example (nginx-ingress):

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: certainaity
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts: [certainaity.example.com]
      secretName: certainaity-tls
  rules:
    - host: certainaity.example.com
      http:
        paths:
          - path: /v1
            pathType: Prefix
            backend:
              service: { name: certainaity-api, port: { number: 8000 } }
          - path: /
            pathType: Prefix
            backend:
              service: { name: certainaity-frontend, port: { number: 80 } }
```

## GPU node pool

Label GPU nodes and use a node selector or node affinity on the worker deployment:

```yaml
nodeSelector:
  cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

The worker `Deployment` in `k8s/worker.yaml` already includes:

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
```

## Scaling

The API HPA scales on CPU utilization. To also scale on custom Prometheus
metrics (e.g., Redis queue depth), install KEDA:

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda --namespace keda --create-namespace
kubectl apply -f k8s/keda-scaledobject.yaml
```
