#!/bin/bash
# Generate RS256 JWT key pair for samosaChaat auth service
set -e

openssl genrsa -out jwt_private.pem 2048
openssl rsa -in jwt_private.pem -pubout -out jwt_public.pem

echo ""
echo "Keys generated: jwt_private.pem, jwt_public.pem"
echo ""
echo "Add to your .env file:"
echo "JWT_PRIVATE_KEY=\"$(cat jwt_private.pem | tr '\n' '|')\""
echo "JWT_PUBLIC_KEY=\"$(cat jwt_public.pem | tr '\n' '|')\""
echo ""
echo "The | characters will be replaced back to newlines by the services."
