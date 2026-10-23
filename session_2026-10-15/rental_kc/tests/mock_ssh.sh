#!/bin/bash
# mock_ssh.sh CMD (tests only): stands in for "ssh root@instance CMD" in the box CPU mock. Runs CMD locally with bash -c,
# stdin/stdout passed through (tar streams work), so driver.sh / fetch_loop.sh run unchanged against a mock instance root.
exec bash -c "$*"
