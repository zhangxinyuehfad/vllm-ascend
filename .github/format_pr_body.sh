#!/bin/bash

set -eux

# ensure 2 argument is passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <pr_number> <new_message>"
    exit 1
fi

PR_NUMBER=$1
NEW_MESSAGE=$2
OLD=/tmp/orig_pr_body.txt
NEW=/tmp/new_pr_body.txt

gh pr view --json body --template "{{.body}}" "${PR_NUMBER}" > "${OLD}"
cp "${OLD}" "${NEW}"

# Remove "FIX #xxxx (*link existing issues this PR will resolve*)"
sed -i '/<!--/,/-->/d' "${NEW}"
echo "$NEW_MESSAGE" >> "${NEW}"

# Run this only if ${NEW} is different than ${OLD}
if ! cmp -s "${OLD}" "${NEW}"; then
    echo
    echo "Updating PR body:"
    echo
    cat "${NEW}"

    gh pr edit --body-file "${NEW}" "${PR_NUMBER}"

else
    echo "No changes needed"
fi
