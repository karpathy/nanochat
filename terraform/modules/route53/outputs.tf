output "zone_id" {
  description = "Hosted zone ID for the apex domain."
  value       = data.aws_route53_zone.this.zone_id
}

output "name_servers" {
  description = "Authoritative name servers (configure these at the registrar)."
  value       = data.aws_route53_zone.this.name_servers
}

output "apex_record_fqdn" {
  description = "FQDN of the apex A record (empty until alb_dns_name is supplied)."
  value       = try(aws_route53_record.apex[0].fqdn, "")
}
