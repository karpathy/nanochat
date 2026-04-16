output "file_system_id" {
  description = "EFS filesystem ID — pass to the EFS CSI driver StorageClass."
  value       = aws_efs_file_system.this.id
}

output "file_system_arn" {
  description = "Filesystem ARN."
  value       = aws_efs_file_system.this.arn
}

output "dns_name" {
  description = "Mount DNS name."
  value       = "${aws_efs_file_system.this.id}.efs.${data.aws_region.current.name}.amazonaws.com"
}

output "access_point_id" {
  description = "Access point for the model-weights directory."
  value       = aws_efs_access_point.model_weights.id
}

output "security_group_id" {
  description = "EFS security group."
  value       = aws_security_group.efs.id
}

data "aws_region" "current" {}
