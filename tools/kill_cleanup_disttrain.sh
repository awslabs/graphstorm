#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 ip_list_file graph_name"
  exit 1
fi

ips=()
while IFS= read -r ip; do
    ips+=($ip)
done < "$1"

graph_name=$2

for ip in "${ips[@]}"; do
    echo "Clean up $ip"
    ssh -n $ip "killall -9 python3"
    ssh -n $ip "rm -rf /dev/shm/node:*"
    ssh -n $ip "rm -rf /dev/shm/edge:*"
    ssh -n $ip "rm -rf /dev/shm/$graph_name*"
done
