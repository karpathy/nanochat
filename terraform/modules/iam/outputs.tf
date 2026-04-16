output "eks_node_role_arn" {
  description = "ARN of the EKS managed-node-group instance role."
  value       = aws_iam_role.eks_node.arn
}

output "eks_node_role_name" {
  description = "Name of the EKS node role."
  value       = aws_iam_role.eks_node.name
}

output "eks_node_instance_profile_name" {
  description = "Instance profile attached to EKS nodes."
  value       = aws_iam_instance_profile.eks_node.name
}

output "alb_controller_role_arn" {
  description = "IAM role to bind to the aws-load-balancer-controller ServiceAccount via IRSA."
  value       = try(aws_iam_role.alb_controller[0].arn, "")
}

output "github_actions_role_arn" {
  description = "Role to assume from GitHub Actions for CI/CD (empty if not enabled)."
  value       = try(aws_iam_role.github_actions[0].arn, "")
}

output "github_oidc_provider_arn" {
  description = "GitHub OIDC provider ARN (empty if not enabled)."
  value       = try(aws_iam_openid_connect_provider.github[0].arn, "")
}
