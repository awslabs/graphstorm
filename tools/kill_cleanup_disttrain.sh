#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 ip_list_file graph_name port_num"
  exit 1
fi

ips=()
while IFS= read -r ip; do
    ips+=($ip)
done < "$1"

graph_name=$2

if [ -z "$3" ]; then
    port_num=2222
else
    port_num=$3
fi

for ip in "${ips[@]}"; do
    echo "Clean up $ip"
    ssh -p $port_num -n $ip "killall -9 python"
    ssh -p $port_num -n $ip "killall -9 python3"
    ssh -p $port_num -n $ip "rm -rf /dev/shm/node:*"
    ssh -p $port_num -n $ip "rm -rf /dev/shm/edge:*"
    ssh -p $port_num -n $ip "rm -rf /dev/shm/$graph_name*"
done
