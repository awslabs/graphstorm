# Entry point for Docker container, executes passed-in command
set -ex
exec "$@"
