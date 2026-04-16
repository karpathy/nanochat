output "certificate_arn" {
  description = "ARN of the issued certificate (use on the ALB Ingress annotation alb.ingress.kubernetes.io/certificate-arn)."
  value       = aws_acm_certificate.this.arn
}

output "domain_validation_options" {
  description = "Raw validation options from AWS — used by the route53 module."
  value       = aws_acm_certificate.this.domain_validation_options
}

output "validation_records" {
  description = "Map keyed by domain → { name, type, record } ready to plug into route53.acm_validation_records."
  value = {
    for dvo in aws_acm_certificate.this.domain_validation_options :
    dvo.domain_name => {
      name   = dvo.resource_record_name
      type   = dvo.resource_record_type
      record = dvo.resource_record_value
    }
  }
}

output "validation_record_fqdns" {
  description = "List of FQDNs to feed into aws_acm_certificate_validation."
  value = [
    for dvo in aws_acm_certificate.this.domain_validation_options :
    dvo.resource_record_name
  ]
}
