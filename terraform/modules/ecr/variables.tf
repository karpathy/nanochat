variable "repository_names" {
  description = "ECR repositories to create."
  type        = list(string)
  default = [
    "samosachaat-frontend",
    "samosachaat-auth",
    "samosachaat-chat-api",
    "samosachaat-inference",
  ]
}

variable "force_delete" {
  description = "Allow Terraform to destroy repositories even if they contain images (true for dev only)."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
