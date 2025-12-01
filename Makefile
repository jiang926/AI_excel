PY=python3.12
PIP=$(PY) -m pip
APP_DIR=/mnt/sde/AI_excel

.PHONY: venv install run run-html docker-build docker-run format

venv:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip

install:
	$(PIP) install -r requirements.txt

run:
	$(PY) cli_plot.py --help

run-html:
	$(PY) cli_plot.py --prompt "示例：根据时间和价格画图" --file "./demo.xlsx" --output "./out.html" --mode "deepseek-chat"

docker-build:
	docker build -t ai-excel:latest .

docker-run:
	docker run --rm -it \
		-e DEEPSEEK_API_KEY=$$DEEPSEEK_API_KEY \
		-v $(APP_DIR):/app \
		ai-excel:latest


