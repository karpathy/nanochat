# Shared remote-state configuration.
#
# Each environment overrides the `key` via `terraform init -backend-config="key=..."`
# (the harness in environments/<env>/main.tf passes `terraform { backend "s3" {} }`
# with no inline values so the same bucket can host multiple state files).
#
# Bootstrap (run once, manually):
#
#   aws s3api create-bucket \
#     --bucket samosachaat-terraform-state \
#     --region us-west-2 \
#     --create-bucket-configuration LocationConstraint=us-west-2
#   aws s3api put-bucket-versioning \
#     --bucket samosachaat-terraform-state \
#     --versioning-configuration Status=Enabled
#   aws s3api put-bucket-encryption \
#     --bucket samosachaat-terraform-state \
#     --server-side-encryption-configuration \
#       '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
#   aws dynamodb create-table \
#     --table-name samosachaat-terraform-locks \
#     --attribute-definitions AttributeName=LockID,AttributeType=S \
#     --key-schema AttributeName=LockID,KeyType=HASH \
#     --billing-mode PAY_PER_REQUEST \
#     --region us-west-2

terraform {
  backend "s3" {
    bucket         = "samosachaat-terraform-state"
    key            = "global/placeholder.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "samosachaat-terraform-locks"
  }
}
