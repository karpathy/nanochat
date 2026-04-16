terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

# Use an existing hosted zone (created out-of-band when registering the domain).
data "aws_route53_zone" "this" {
  name         = var.domain_name
  private_zone = false
}

# alb_dns_name / alb_zone_id come from the AWS Load Balancer Controller after the
# Ingress is created (look up via `kubectl get ingress` or a data source). Pass
# empty strings to skip A-record creation on the first apply, then re-apply.
resource "aws_route53_record" "apex" {
  count = var.alb_dns_name == "" ? 0 : 1

  zone_id = data.aws_route53_zone.this.zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = var.alb_dns_name
    zone_id                = var.alb_zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "subdomains" {
  for_each = var.alb_dns_name == "" ? toset([]) : toset(var.subdomains)

  zone_id = data.aws_route53_zone.this.zone_id
  name    = "${each.key}.${var.domain_name}"
  type    = "A"

  alias {
    name                   = var.alb_dns_name
    zone_id                = var.alb_zone_id
    evaluate_target_health = true
  }
}

# ACM DNS-validation CNAMEs. Pass the map exported by the ACM module.
resource "aws_route53_record" "acm_validation" {
  for_each = var.acm_validation_records

  zone_id         = data.aws_route53_zone.this.zone_id
  name            = each.value.name
  type            = each.value.type
  records         = [each.value.record]
  ttl             = 60
  allow_overwrite = true
}
