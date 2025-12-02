#!/bin/bash

PROJECT="nzp-nanochat"
MACHINE_TYPE="g2-standard-4"  # Smallest L4 machine type
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"

# Parse debug flag
DEBUG=false
if [[ "${1:-}" == "--debug" ]]; then
    DEBUG=true
    echo "Debug mode enabled - will show GCP error messages"
    echo ""
fi

echo "=== Testing L4 GPU Availability Across ALL Global Regions ==="
echo "This will attempt to create small L4 instances and immediately delete them"
echo "Order: US -> Europe -> Others. Stops at first success."
echo ""

# Get all regions dynamically
echo "Fetching all GCP regions..."
ALL_REGIONS=$(gcloud compute regions list --project="$PROJECT" --format="value(name)" 2>/dev/null | sort)
REGION_COUNT=$(echo "$ALL_REGIONS" | wc -l | tr -d ' ')
echo "Found $REGION_COUNT regions to test"
echo ""

RESULTS_FILE=$(mktemp)
ERROR_LOG=$(mktemp)

# Order regions: US first, then Europe, then others
ordered_regions=$(echo "$ALL_REGIONS" | tr ' ' '\n' | grep '^us-' || true)
ordered_regions+=$'\n'
ordered_regions+=$(echo "$ALL_REGIONS" | tr ' ' '\n' | grep '^europe-' || true)
ordered_regions+=$'\n'
ordered_regions+=$(echo "$ALL_REGIONS" | tr ' ' '\n' | grep -vE '^(us-|europe-)' || true)

# Remove empty lines
ordered_regions=$(echo "$ordered_regions" | sed '/^$/d')

current=0
found_any=false

# Iterate over ordered list
for region in $ordered_regions; do
    current=$((current + 1))
    echo "[$current/$REGION_COUNT] Testing region: $region"
    
    # Get zones for region
    zones=$(gcloud compute zones list --project="$PROJECT" --filter="region:$region" --format="value(name)" 2>/dev/null)
    if [ -z "$zones" ]; then
        echo "  ⚠️  No zones found for region $region"
        continue
    fi
    
    found_capacity=false
    available_zone=""
    
    for zone in $zones; do
        echo -n "  Checking zone $zone... "
        
        instance_name="test-l4-capacity-$$-$(date +%s)"
        
        # Try to create instance - capture stderr
        error_output=$(mktemp)
        if gcloud compute instances create "$instance_name" \
            --zone="$zone" \
            --machine-type="$MACHINE_TYPE" \
            --accelerator="type=nvidia-l4,count=1" \
            --image-family="$IMAGE_FAMILY" \
            --image-project="$IMAGE_PROJECT" \
            --boot-disk-size=200GB \
            --boot-disk-type=pd-standard \
            --network="nanochat-network" \
            --no-address \
            --shielded-secure-boot \
            --maintenance-policy=TERMINATE \
            --project="$PROJECT" \
            --quiet \
            2>"$error_output"; then
            
            echo "✅ AVAILABLE"
            available_zone="$zone"
            found_capacity=true
            
            # Delete instance
            gcloud compute instances delete "$instance_name" --zone="$zone" --project="$PROJECT" --quiet 2>/dev/null || true
            
            rm -f "$error_output"
            break
        else
            echo "❌ No capacity"
            if [ "$DEBUG" = true ]; then
                echo "    ERROR DETAILS:"
                sed 's/^/    /' "$error_output"
                cat "$error_output" >> "$ERROR_LOG"
            fi
            rm -f "$error_output"
        fi
    done
    
    if [ "$found_capacity" = true ]; then
        echo "$region: ✅ Available in $available_zone" >> "$RESULTS_FILE"
        echo ""
        echo "✅ Found capacity in $region ($available_zone). Stopping further checks."
        found_any=true
        break
    else
        echo "$region: ❌ No capacity in any zone" >> "$RESULTS_FILE"
    fi
    
    echo ""
done

# Print summary (will only contain up to first successful region)
echo "=========================================================="
echo "         L4 GPU AVAILABILITY SUMMARY (GLOBAL)            "
echo "=========================================================="
cat "$RESULTS_FILE" | sort
echo "=========================================================="
echo ""

if [ "$found_any" = true ]; then
    echo "✅ Recommendation: Use the region marked with ✅ above."
else
    echo "❌ No L4 capacity found in any tested region."
fi

# Cleanup
rm -f "$RESULTS_FILE"
if [ "$DEBUG" = true ]; then
    echo "Debug log: $ERROR_LOG"
else
    rm -f "$ERROR_LOG"
fi
