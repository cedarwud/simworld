.PHONY: up down down-v clean build

up: down
	docker compose up

down:
	docker compose down
	docker system prune -f

down-v:
	docker compose down -v
	docker volume prune -f
	docker system prune -f

clean:
	docker compose down --rmi all --volumes --remove-orphans
	docker volume prune -f
	docker system prune -af

build:
	docker compose build --no-cache
