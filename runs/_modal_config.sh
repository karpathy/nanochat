#!/bin/bash

# Modal configuration: shared Volume name. Source this in runmodal.sh and modal_peek.sh.
# Default: nanochat-modal. Override with env var to use a different Volume.

export NANOCHAT_MODAL_VOLUME="${NANOCHAT_MODAL_VOLUME:-nanochat-modal}"
