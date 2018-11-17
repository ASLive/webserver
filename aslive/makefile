HOST_PORT=6379
PY=python3

run: migrate
	sudo docker run -p $(HOST_PORT):$(HOST_PORT) -d redis:2.8 || \
	$(PY) manage.py runserver

migrate:
	$(PY) manage.py migrate