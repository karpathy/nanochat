output "cluster_name" {
  description = "EKS cluster name."
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS API server endpoint."
  value       = module.eks.cluster_endpoint
}

output "cluster_certificate_authority_data" {
  description = "Base64-encoded cluster CA certificate."
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_security_group_id" {
  description = "Cluster control-plane security group."
  value       = module.eks.cluster_security_group_id
}

output "node_security_group_id" {
  description = "Security group attached to managed node group ENIs (used by RDS / EFS to allow inbound traffic from nodes)."
  value       = module.eks.node_security_group_id
}

output "oidc_provider_arn" {
  description = "IRSA OIDC provider ARN."
  value       = module.eks.oidc_provider_arn
}

output "oidc_provider_url" {
  description = "IRSA OIDC issuer URL (without https://)."
  value       = module.eks.oidc_provider
}

output "current_node_ami_id" {
  description = "The current EKS-optimized AMI ID used by the node group."
  value       = data.aws_ssm_parameter.eks_ami_id.value
}
