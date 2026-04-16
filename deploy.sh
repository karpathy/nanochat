#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# samosaChaat Deploy Switch
#
# Usage:
#   ./deploy.sh ec2          Deploy monolith to EC2 via docker-compose
#   ./deploy.sh ec2-down     Stop services on EC2
#   ./deploy.sh eks          Provision EKS + deploy via Helm (demo/grading)
#   ./deploy.sh eks-down     Tear down EKS (save $$$)
#   ./deploy.sh status       Show what's currently running
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_ACCOUNT="883107058766"
AWS_REGION="us-west-2"
ECR_REGISTRY="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
EC2_HOST="52.10.243.118"
EC2_USER="ubuntu"
EC2_KEY="$HOME/.ssh/samosachaat.pem"  # adjust if your key is elsewhere
DOMAIN="samosachaat.art"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[samosaChaat]${NC} $1"; }
warn() { echo -e "${YELLOW}[samosaChaat]${NC} $1"; }
err()  { echo -e "${RED}[samosaChaat]${NC} $1" >&2; }

#─── EC2 MONOLITH ─────────────────────────────────────────────────────────────

ec2_deploy() {
    log "Deploying to EC2 monolith at ${EC2_HOST}..."

    # 1. Login to ECR locally to get credentials
    log "Logging into ECR..."
    aws ecr get-login-password --region ${AWS_REGION} | \
        ssh -i "${EC2_KEY}" -o StrictHostKeyChecking=no ${EC2_USER}@${EC2_HOST} \
        "docker login --username AWS --password-stdin ${ECR_REGISTRY}" 2>/dev/null

    # 2. Sync repo to EC2
    log "Syncing code to EC2..."
    ssh -i "${EC2_KEY}" ${EC2_USER}@${EC2_HOST} bash -s << 'REMOTE_SCRIPT'
        set -e
        cd /home/ubuntu

        # Clone or update repo
        if [ -d samosachaat ]; then
            cd samosachaat
            git fetch origin master
            git reset --hard origin/master
        else
            git clone https://github.com/manmohan659/nanochat.git samosachaat
            cd samosachaat
        fi

        # Ensure .env exists
        if [ ! -f .env ]; then
            cp .env.example .env
            echo "⚠️  Created .env from template — edit it with real values!"
        fi
REMOTE_SCRIPT

    # 3. Copy .env from local if it exists
    if [ -f "${SCRIPT_DIR}/.env" ]; then
        log "Syncing .env to EC2..."
        scp -i "${EC2_KEY}" "${SCRIPT_DIR}/.env" ${EC2_USER}@${EC2_HOST}:/home/ubuntu/samosachaat/.env
    fi

    # 4. Pull images and start services
    log "Pulling images and starting services..."
    ssh -i "${EC2_KEY}" ${EC2_USER}@${EC2_HOST} bash -s << REMOTE_DEPLOY
        set -e
        cd /home/ubuntu/samosachaat

        # Set ECR registry in environment
        export ECR_REGISTRY=${ECR_REGISTRY}
        export IMAGE_TAG=dev-latest

        # Pull latest images
        docker compose -f docker-compose.yml -f docker-compose.prod.yml pull

        # Start everything
        docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

        # Run DB migrations
        echo "Running database migrations..."
        sleep 5  # wait for postgres to be ready
        docker compose exec -T chat-api alembic upgrade head 2>/dev/null || \
            echo "Migrations skipped (may need manual .env setup)"

        echo ""
        docker compose ps
REMOTE_DEPLOY

    # 5. Setup SSL if not already done
    log "Checking SSL..."
    ssh -i "${EC2_KEY}" ${EC2_USER}@${EC2_HOST} bash -s << 'SSL_CHECK'
        if [ ! -d /etc/letsencrypt/live/samosachaat.art ]; then
            echo "Setting up SSL with certbot..."
            sudo apt-get update -qq && sudo apt-get install -y -qq certbot > /dev/null 2>&1
            sudo certbot certonly --standalone --non-interactive \
                --agree-tos -m manmohan659@gmail.com \
                -d samosachaat.art -d www.samosachaat.art \
                --pre-hook "docker compose -f /home/ubuntu/samosachaat/docker-compose.yml -f /home/ubuntu/samosachaat/docker-compose.prod.yml stop nginx" \
                --post-hook "docker compose -f /home/ubuntu/samosachaat/docker-compose.yml -f /home/ubuntu/samosachaat/docker-compose.prod.yml start nginx"
        else
            echo "SSL already configured."
        fi
SSL_CHECK

    echo ""
    log "EC2 deploy complete!"
    log "  App:     https://${DOMAIN}"
    log "  Grafana: https://${DOMAIN}/grafana/"
    log "  EC2:     ${EC2_HOST}"
}

ec2_down() {
    log "Stopping services on EC2..."
    ssh -i "${EC2_KEY}" ${EC2_USER}@${EC2_HOST} \
        "cd /home/ubuntu/samosachaat && docker compose -f docker-compose.yml -f docker-compose.prod.yml down"
    log "EC2 services stopped."
}

#─── EKS CLUSTER ──────────────────────────────────────────────────────────────

eks_deploy() {
    local ENV="${1:-dev}"
    log "Provisioning EKS cluster (${ENV})... This takes ~15-20 minutes."

    cd "${SCRIPT_DIR}/terraform/environments/${ENV}"

    # Init & apply Terraform
    log "Running terraform init..."
    terraform init

    log "Running terraform apply..."
    terraform apply -auto-approve

    # Get cluster info
    local CLUSTER_NAME=$(terraform output -raw eks_cluster_name 2>/dev/null || echo "samosachaat-${ENV}")
    log "Configuring kubectl for ${CLUSTER_NAME}..."
    aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region ${AWS_REGION}

    # Install ALB Ingress Controller
    log "Installing ALB Ingress Controller..."
    helm repo add eks https://aws.github.io/eks-charts 2>/dev/null || true
    helm repo update
    helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName="${CLUSTER_NAME}" \
        --set serviceAccount.create=true \
        --set serviceAccount.name=aws-load-balancer-controller \
        --wait --timeout 5m 2>/dev/null || warn "ALB controller may need IRSA setup"

    # Deploy observability stack
    log "Deploying observability stack..."
    helm dependency build "${SCRIPT_DIR}/helm/observability" 2>/dev/null || true
    helm upgrade --install observability "${SCRIPT_DIR}/helm/observability" \
        --namespace monitoring --create-namespace \
        --wait --timeout 10m 2>/dev/null || warn "Observability deploy needs review"

    # Deploy samosaChaat
    local VALUES_FILE="${SCRIPT_DIR}/helm/samosachaat/values-${ENV}.yaml"
    log "Deploying samosaChaat to EKS..."
    helm upgrade --install samosachaat "${SCRIPT_DIR}/helm/samosachaat" \
        -f "${VALUES_FILE}" \
        --set global.imageRegistry="${ECR_REGISTRY}" \
        --set global.imageTag="dev-latest" \
        --set ingress.acmCertArn="$(terraform output -raw acm_certificate_arn 2>/dev/null || echo '')" \
        --namespace samosachaat --create-namespace \
        --wait --timeout 10m

    echo ""
    log "EKS deploy complete!"
    log "  Cluster: ${CLUSTER_NAME}"
    kubectl get pods -n samosachaat
    echo ""
    kubectl get ingress -n samosachaat 2>/dev/null || true
}

eks_down() {
    local ENV="${1:-dev}"
    warn "Tearing down EKS cluster (${ENV})... This saves ~\$0.10/hr + node costs."

    cd "${SCRIPT_DIR}/terraform/environments/${ENV}"

    # Remove Helm releases first (cleans up ALB, etc.)
    local CLUSTER_NAME=$(terraform output -raw eks_cluster_name 2>/dev/null || echo "samosachaat-${ENV}")
    aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region ${AWS_REGION} 2>/dev/null || true

    log "Removing Helm releases..."
    helm uninstall samosachaat -n samosachaat 2>/dev/null || true
    helm uninstall observability -n monitoring 2>/dev/null || true
    helm uninstall aws-load-balancer-controller -n kube-system 2>/dev/null || true

    # Destroy infrastructure
    log "Running terraform destroy..."
    terraform destroy -auto-approve

    log "EKS cluster destroyed. Costs stopped."
}

#─── STATUS ───────────────────────────────────────────────────────────────────

show_status() {
    echo ""
    log "=== samosaChaat Deployment Status ==="
    echo ""

    # Check EC2
    echo "EC2 Monolith (${EC2_HOST}):"
    if ssh -i "${EC2_KEY}" -o ConnectTimeout=5 ${EC2_USER}@${EC2_HOST} \
        "docker compose -f /home/ubuntu/samosachaat/docker-compose.yml -f /home/ubuntu/samosachaat/docker-compose.prod.yml ps --format 'table {{.Name}}\t{{.Status}}'" 2>/dev/null; then
        echo ""
    else
        echo "  Not running or unreachable."
    fi

    # Check EKS
    echo "EKS Cluster:"
    if kubectl get nodes 2>/dev/null; then
        echo ""
        kubectl get pods -n samosachaat 2>/dev/null || echo "  No samosachaat namespace."
    else
        echo "  No EKS cluster configured."
    fi

    # Check ECR images
    echo ""
    echo "ECR Images (latest):"
    for svc in frontend auth chat-api inference; do
        TAG=$(aws ecr describe-images --repository-name samosachaat/${svc} --region ${AWS_REGION} \
            --query 'sort_by(imageDetails,&imagePushedAt)[-1].imageTags[0]' --output text 2>/dev/null || echo "none")
        echo "  samosachaat/${svc}: ${TAG}"
    done
}

#─── MAIN ─────────────────────────────────────────────────────────────────────

case "${1:-help}" in
    ec2)        ec2_deploy ;;
    ec2-down)   ec2_down ;;
    eks)        eks_deploy "${2:-dev}" ;;
    eks-down)   eks_down "${2:-dev}" ;;
    status)     show_status ;;
    *)
        echo "samosaChaat Deploy Switch"
        echo ""
        echo "Usage: ./deploy.sh <mode>"
        echo ""
        echo "Modes:"
        echo "  ec2          Deploy monolith to EC2 (cheap, always-on)"
        echo "  ec2-down     Stop EC2 services"
        echo "  eks [env]    Provision EKS + deploy (demo/grading) [dev|uat|prod]"
        echo "  eks-down     Tear down EKS (save \$\$\$)"
        echo "  status       Show what's running"
        ;;
esac
