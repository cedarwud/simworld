make up：make down && docker compose up
make down：docker compose down.docker network prune -f.docker system prune -f
make down-v：docker compose down -v.docker volume prune -f.docker network prune -f.docker system prune -f
make clean：docker compose down --rmi all --volumes --remove-orphans.docker volume prune -f.docker network prune -f.docker system prune -af
make build：docker compose build --no-cache