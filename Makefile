env-create:
	tox -e streaming-app

env-compile:
	pip-compile requirements.in

test:
	pytest tests

lint:
	pylint streaming_app