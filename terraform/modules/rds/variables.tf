variable "identifier" {
  description = "DB instance identifier (also used as name prefix)."
  type        = string
}

variable "vpc_id" {
  description = "VPC the database lives in."
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnets for the DB subnet group (>= 2 AZs)."
  type        = list(string)
}

variable "eks_node_security_group_id" {
  description = "Node SG that should be allowed inbound to PostgreSQL."
  type        = string
}

variable "instance_class" {
  description = "RDS instance class (e.g. db.t3.micro for dev, db.t3.medium for prod)."
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "Initial database name."
  type        = string
  default     = "samosachaat"
}

variable "db_username" {
  description = "Master username."
  type        = string
  default     = "samosachaat_admin"
}

variable "allocated_storage" {
  description = "Initial storage (GB)."
  type        = number
  default     = 20
}

variable "max_allocated_storage" {
  description = "Storage autoscaling cap (GB)."
  type        = number
  default     = 100
}

variable "multi_az" {
  description = "Enable Multi-AZ (recommended for prod)."
  type        = bool
  default     = false
}

variable "skip_final_snapshot" {
  description = "Skip the final snapshot when destroying (true for dev)."
  type        = bool
  default     = true
}

variable "deletion_protection" {
  description = "Block accidental deletion (recommended for prod)."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
