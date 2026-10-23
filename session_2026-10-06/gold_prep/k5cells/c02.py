## Runtime contract and artifact checks

The first cells pin checkpoint hashes, dependency versions, memory limits, and
the D4 serving contract before expensive inference begins. Any missing or
changed required asset stops the run instead of silently changing predictions.
