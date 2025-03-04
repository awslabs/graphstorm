#!/bin/bash

# Function to renew the session
renew_session() {
  account_b_role_arn=$1
  session_name=$2
  region=${3:-"us-east-1"}  # Default to "us-west-2" if no region is provided

  # Clear any existing AWS session cache
  unset AWS_ACCESS_KEY_ID
  unset AWS_SECRET_ACCESS_KEY
  unset AWS_SESSION_TOKEN

  # Assume the role using AWS CLI and capture the temporary credentials
  credentials=$(aws sts assume-role \
    --role-arn "$account_b_role_arn" \
    --role-session-name "$session_name" \
    --duration-seconds 3599 \
    --region "$region")

  # Extract the credentials from the JSON response
  AWS_ACCESS_KEY_ID=$(echo "$credentials" | jq -r .Credentials.AccessKeyId)
  AWS_SECRET_ACCESS_KEY=$(echo "$credentials" | jq -r .Credentials.SecretAccessKey)
  AWS_SESSION_TOKEN=$(echo "$credentials" | jq -r .Credentials.SessionToken)

  # Export the environment variables with the new credentials
  export AWS_ACCESS_KEY_ID
  export AWS_SECRET_ACCESS_KEY
  export AWS_SESSION_TOKEN
  export AWS_DEFAULT_REGION="$region"

  echo "Session renewed with new credentials."
}

# Example usage
# renew_session "arn:aws:iam::account-b-id:role/role-name" "session-name"
renew_session "$1" "$2" "$3"
