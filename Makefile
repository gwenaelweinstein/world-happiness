.DEFAULT: help
.SILENT:
SHELL=bash

help:
	echo
	echo "   ----------      --------------------------------------"
	echo "    start           Start and rebuild image if necessary "
	echo "   ----------      --------------------------------------"
	echo "    stop            Stop                                 "
	echo "   ----------      --------------------------------------"
	echo "    connect         SSH to container                     "
	echo "   ----------      --------------------------------------"
	echo

start:
	docker compose up -d --build

stop:
	docker compose down

connect:
	docker exec -it world-happiness /bin/bash
