variable "domain_name" {
  description = "Primary domain on the certificate (e.g. samosachaat.art)."
  type        = string
}

variable "subject_alternative_names" {
  description = "SAN list (e.g. [\"*.samosachaat.art\"])."
  type        = list(string)
  default     = []
}

variable "wait_for_validation" {
  description = "Block apply until DNS validation succeeds. Disable on first apply if Route53 records are created in the same plan."
  type        = bool
  default     = true
}

variable "validation_record_fqdns" {
  description = "FQDNs of the DNS validation records (passed in from the route53 module)."
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
