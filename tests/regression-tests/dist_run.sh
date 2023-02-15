if [ "$#" -lt 2 ]; then
  echo "Usage: $0 ip_list_file command"
  exit 1
fi

ips=`cat $1`

ips=()
while IFS= read -r ip; do
    ips+=($ip)
done < "$1"

remote_command=$2
ssh_port=22

for ip in "${ips[@]}"; do
    echo "run on $ip"
    echo "ssh -p $ssh_port $ip $remote_command"
    ssh -p $ssh_port $ip $remote_command
done
