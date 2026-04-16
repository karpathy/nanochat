output "vpc_id" {
  description = "VPC identifier."
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "Primary CIDR block of the VPC."
  value       = module.vpc.vpc_cidr_block
}

output "private_subnet_ids" {
  description = "Private subnet identifiers (one per AZ)."
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "Public subnet identifiers (one per AZ)."
  value       = module.vpc.public_subnets
}

output "private_subnet_cidrs" {
  description = "Private subnet CIDR blocks."
  value       = module.vpc.private_subnets_cidr_blocks
}

output "azs" {
  description = "AZs in use."
  value       = module.vpc.azs
}

output "natgw_ids" {
  description = "NAT gateway identifiers."
  value       = module.vpc.natgw_ids
}
