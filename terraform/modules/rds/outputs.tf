output "db_instance_endpoint" {
  description = "Endpoint of the form host:port."
  value       = module.db.db_instance_endpoint
}

output "db_instance_address" {
  description = "Hostname of the DB instance."
  value       = module.db.db_instance_address
}

output "db_instance_port" {
  description = "Listening port."
  value       = module.db.db_instance_port
}

output "db_instance_name" {
  description = "Initial database name."
  value       = module.db.db_instance_name
}

output "db_instance_username" {
  description = "Master username."
  value       = module.db.db_instance_username
  sensitive   = true
}

output "db_password" {
  description = "Generated master password (write to Secrets Manager / Parameter Store from your env config)."
  value       = random_password.db.result
  sensitive   = true
}

output "db_security_group_id" {
  description = "Security group attached to the DB."
  value       = aws_security_group.db.id
}
