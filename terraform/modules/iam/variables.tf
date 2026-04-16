variable "name_prefix" {
  description = "Prefix for IAM resource names (e.g. samosachaat-dev)."
  type        = string
}

variable "oidc_provider_arn" {
  description = "EKS OIDC provider ARN. Pass empty string to skip ALB controller role creation."
  type        = string
  default     = ""
}

variable "oidc_provider_url" {
  description = "EKS OIDC issuer hostname (no scheme, e.g. oidc.eks.us-west-2.amazonaws.com/id/XXX)."
  type        = string
  default     = ""
}

variable "create_github_oidc" {
  description = "Create the GitHub Actions OIDC provider + CI role. Set to true exactly once per AWS account."
  type        = bool
  default     = false
}

variable "github_repositories" {
  description = "GitHub repositories allowed to assume the CI role (e.g. [\"manmohan659/nanochat\"])."
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
