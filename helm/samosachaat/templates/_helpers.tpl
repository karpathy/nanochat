{{/*
Common helpers for the samosachaat umbrella chart.
*/}}

{{/* Chart name truncated to 63 chars (k8s label limit). */}}
{{- define "samosachaat.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Fully-qualified release name used as the chart-wide prefix. */}}
{{- define "samosachaat.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* Common labels. */}}
{{- define "samosachaat.labels" -}}
app.kubernetes.io/name: {{ include "samosachaat.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: samosachaat
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end -}}

{{/*
Compose a service-specific name. Honors .Values.deployment.slot so that a
"green" slot (during blue/green prod releases) produces e.g. "frontend-green".
Usage:  {{ include "samosachaat.svcName" (dict "root" . "svc" "frontend") }}
*/}}
{{- define "samosachaat.svcName" -}}
{{- $root := .root -}}
{{- $svc := .svc -}}
{{- $slot := default "" $root.Values.deployment.slot -}}
{{- if $slot -}}
{{- printf "%s-%s" $svc $slot | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $svc | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{/* Per-service selector labels. */}}
{{- define "samosachaat.selectorLabels" -}}
{{- $root := .root -}}
{{- $svc := .svc -}}
app.kubernetes.io/name: {{ $svc }}
app.kubernetes.io/instance: {{ $root.Release.Name }}
app.kubernetes.io/component: {{ $svc }}
{{- with $root.Values.deployment.slot }}
app.kubernetes.io/slot: {{ . }}
{{- end }}
{{- end -}}

{{/* Render a full image reference given a service's .image block. */}}
{{- define "samosachaat.image" -}}
{{- $root := .root -}}
{{- $svc := .svc -}}
{{- $registry := $root.Values.global.imageRegistry | default "" -}}
{{- $repo := $svc.image.repository -}}
{{- $tag := $root.Values.global.imageTag | default "dev-latest" -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repo $tag -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end -}}

{{/* Namespace that every resource should land in. */}}
{{- define "samosachaat.namespace" -}}
{{- default .Release.Namespace .Values.namespace.name -}}
{{- end -}}
