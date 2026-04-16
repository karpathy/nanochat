variable "name" {
  description = "Filesystem name (used in tags and the creation token)."
  type        = string
}

variable "vpc_id" {
  description = "VPC the mount targets live in."
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnets that get mount targets (one per AZ)."
  type        = list(string)
}

variable "eks_node_security_group_id" {
  description = "Node SG allowed to mount the filesystem."
  type        = string
}

variable "performance_mode" {
  description = "EFS performance mode."
  type        = string
  default     = "generalPurpose"
}

variable "throughput_mode" {
  description = "EFS throughput mode."
  type        = string
  default     = "bursting"
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
