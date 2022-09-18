env-create:
	tox -e streaming-app

env-compile:
	pip-compile requirements.in

test:
	pytest tests

lint:
	pylint streaming_app

run-worker:
	BROKER_URI=$(BROKER_URI); faust -A streaming_app worker -l info
