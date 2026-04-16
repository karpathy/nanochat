variable "domain_name" {
  description = "Apex domain (e.g. samosachaat.art). A hosted zone for this domain must already exist."
  type        = string
}

variable "subdomains" {
  description = "Subdomains to alias to the ALB (e.g. [\"grafana\", \"api\"])."
  type        = list(string)
  default     = ["grafana"]
}

variable "alb_dns_name" {
  description = "ALB DNS name from the AWS Load Balancer Controller. Empty string skips A-record creation (first-apply bootstrap)."
  type        = string
  default     = ""
}

variable "alb_zone_id" {
  description = "ALB hosted-zone ID (region-specific)."
  type        = string
  default     = ""
}

variable "acm_validation_records" {
  description = "Map keyed by domain → { name, type, record } — pass module.acm.validation_records here."
  type = map(object({
    name   = string
    type   = string
    record = string
  }))
  default = {}
}
