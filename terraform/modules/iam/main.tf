terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = ">= 4.0"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

##############################################
# EKS managed-node-group instance role
##############################################

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eks_node" {
  name               = "${var.name_prefix}-eks-node"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
  tags               = var.tags
}

locals {
  eks_node_managed_policies = {
    worker     = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEKSWorkerNodePolicy"
    cni        = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEKS_CNI_Policy"
    ecr_pull   = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
    ebs_csi    = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
    efs_csi    = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonEFSCSIDriverPolicy"
    ssm        = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }
}

resource "aws_iam_role_policy_attachment" "eks_node" {
  for_each   = local.eks_node_managed_policies
  role       = aws_iam_role.eks_node.name
  policy_arn = each.value
}

resource "aws_iam_instance_profile" "eks_node" {
  name = aws_iam_role.eks_node.name
  role = aws_iam_role.eks_node.name
}

##############################################
# AWS Load Balancer Controller IRSA role
##############################################

data "aws_iam_policy_document" "alb_irsa_assume" {
  count = var.oidc_provider_arn == "" ? 0 : 1

  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [var.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${var.oidc_provider_url}:sub"
      values   = ["system:serviceaccount:kube-system:aws-load-balancer-controller"]
    }
    condition {
      test     = "StringEquals"
      variable = "${var.oidc_provider_url}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "alb_controller" {
  count              = var.oidc_provider_arn == "" ? 0 : 1
  name               = "${var.name_prefix}-alb-controller"
  assume_role_policy = data.aws_iam_policy_document.alb_irsa_assume[0].json
  tags               = var.tags
}

resource "aws_iam_policy" "alb_controller" {
  count       = var.oidc_provider_arn == "" ? 0 : 1
  name        = "${var.name_prefix}-alb-controller"
  description = "Permissions required by the AWS Load Balancer Controller."
  policy      = file("${path.module}/policies/alb_controller.json")
}

resource "aws_iam_role_policy_attachment" "alb_controller" {
  count      = var.oidc_provider_arn == "" ? 0 : 1
  role       = aws_iam_role.alb_controller[0].name
  policy_arn = aws_iam_policy.alb_controller[0].arn
}

##############################################
# GitHub Actions OIDC provider + CI/CD role
##############################################

data "tls_certificate" "github" {
  count = var.create_github_oidc ? 1 : 0
  url   = "https://token.actions.githubusercontent.com"
}

resource "aws_iam_openid_connect_provider" "github" {
  count           = var.create_github_oidc ? 1 : 0
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = data.tls_certificate.github[0].certificates[*].sha1_fingerprint
  tags            = var.tags
}

data "aws_iam_policy_document" "github_assume" {
  count = var.create_github_oidc ? 1 : 0

  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github[0].arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = [for r in var.github_repositories : "repo:${r}:*"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  count              = var.create_github_oidc ? 1 : 0
  name               = "${var.name_prefix}-github-actions"
  assume_role_policy = data.aws_iam_policy_document.github_assume[0].json
  tags               = var.tags
}

# Permissions the CI role needs to push images, update kubeconfig, and apply manifests.
data "aws_iam_policy_document" "github_actions" {
  count = var.create_github_oidc ? 1 : 0

  statement {
    sid = "ECRAuth"
    actions = [
      "ecr:GetAuthorizationToken",
    ]
    resources = ["*"]
  }

  statement {
    sid = "ECRPushPull"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
      "ecr:DescribeRepositories",
      "ecr:ListImages",
    ]
    resources = ["arn:${data.aws_partition.current.partition}:ecr:*:${data.aws_caller_identity.current.account_id}:repository/samosachaat-*"]
  }

  statement {
    sid       = "EKSDescribe"
    actions   = ["eks:DescribeCluster", "eks:ListClusters"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "github_actions" {
  count  = var.create_github_oidc ? 1 : 0
  name   = "${var.name_prefix}-github-actions"
  policy = data.aws_iam_policy_document.github_actions[0].json
}

resource "aws_iam_role_policy_attachment" "github_actions" {
  count      = var.create_github_oidc ? 1 : 0
  role       = aws_iam_role.github_actions[0].name
  policy_arn = aws_iam_policy.github_actions[0].arn
}
