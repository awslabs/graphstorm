#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 ip_list_file port_num graph_name"
  exit 1
fi

ips=()
while IFS= read -r ip; do
    ips+=($ip)
done < "$1"

port_num=$2

graph_name=$3

for ip in "${ips[@]}"; do
    echo "Clean up $ip"
    ssh -p $port_num -n $ip "killall -9 python3"
    ssh -p $port_num -n $ip "rm -rf /dev/shm/node:*"
    ssh -p $port_num -n $ip "rm -rf /dev/shm/edge:*"
    ssh -p $port_num -n $ip "rm -rf /dev/shm/$graph_name*"
done
