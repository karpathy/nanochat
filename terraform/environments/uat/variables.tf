variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (dev/uat/prod)."
  type        = string
  default     = "uat"
}

variable "domain_name" {
  description = "Apex domain — must already have a Route53 hosted zone."
  type        = string
  default     = "samosachaat.art"
}

variable "github_repositories" {
  description = "GitHub repos that may assume the CI role."
  type        = list(string)
  default     = ["manmohan659/nanochat"]
}
