"""
WorkBuddy RAG - 统一入口 Makefile
所有常用操作一行命令搞定，无需记脚本路径

用法:
  make help          查看所有命令
  make install       安装依赖（pip install -r requirements.txt）
  make install-dev   安装含 dev/test 的依赖
  make download-models  下载 bge-m3 + bge-reranker 模型
  make health        健康度检查
  make ingest        入库当前工作区的 memory 文件
  make ingest-dry    dry-run，看会扫到哪些
  make query Q="xxx" 检索
  make distill       跑一次蒸馏
  make distill-cron  注册每日 03:00 自动蒸馏
  make bootstrap     安装 Windows Run 键（启动钩子）
  make bootstrap-uninstall  卸载启动钩子
  make daemon        启动 watchdog 守护进程（前台）
  make daemon-install  把守护进程注册为系统服务
  make test          跑全部测试
  make clean         清理临时索引
"""
.PHONY: help install install-dev download-models health ingest ingest-dry query distill distill-cron bootstrap bootstrap-uninstall daemon daemon-install daemon-uninstall test clean docker-build docker-run

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PROJECT_ROOT := $(shell pwd)
VENV_DIR ?= $(PROJECT_ROOT)/.venv
VENV_PYTHON := $(VENV_DIR)/Scripts/python.exe

# Linux/Mac 兼容
ifeq ($(OS),Windows_NT)
	VENV_ACTIVATE := $(VENV_DIR)/Scripts/activate
	PYTHON_BIN := $(VENV_PYTHON)
else
	VENV_ACTIVATE := $(VENV_DIR)/bin/activate
	PYTHON_BIN := $(VENV_DIR)/bin/python
endif

help:
	@echo "WorkBuddy RAG - 可用命令"
	@echo ""
	@echo "  安装:"
	@echo "    make install              安装核心依赖"
	@echo "    make install-dev          安装含 dev/test 的依赖"
	@echo "    make download-models      下载 bge-m3 + bge-reranker 模型（一次性）"
	@echo ""
	@echo "  数据:"
	@echo "    make ingest               入库默认 memory 目录"
	@echo "    make ingest-dry           dry-run，看会扫到哪些"
	@echo "    make query Q='SkillFather'  检索（demo）"
	@echo "    make distill              跑一次蒸馏"
	@echo "    make health               健康度检查"
	@echo ""
	@echo "  自动化:"
	@echo "    make distill-cron         注册每日 03:00 自动蒸馏"
	@echo "    make bootstrap            安装 Windows Run 键（启动钩子）"
	@echo "    make bootstrap-uninstall  卸载启动钩子"
	@echo "    make daemon-install       注册 watchdog 守护进程"
	@echo "    make daemon               启动守护进程（前台）"
	@echo ""
	@echo "  开发:"
	@echo "    make test                 跑全部测试"
	@echo "    make clean                清理临时索引"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build         构建镜像"
	@echo "    make docker-run           启动容器"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[test,watchdog]"
	@if [ "$(OS)" = "Windows_NT" ]; then $(PIP) install pywin32>=306; fi

download-models:
	@if [ ! -f "$$HOME/.cache/huggingface/models--BAAI--bge-m3" ]; then \
		echo "Downloading bge-m3..."; \
		$(PYTHON) download_bge_m3.py; \
	else \
		echo "✓ bge-m3 already cached"; \
	fi
	@if [ ! -d "$$HOME/.cache/huggingface/models--BAAI--bge-reranker-v2-m3" ]; then \
		echo "Downloading bge-reranker-v2-m3..."; \
		$(PYTHON) download_bge_reranker.py; \
	else \
		echo "✓ bge-reranker-v2-m3 already cached"; \
	fi

health:
	$(PYTHON) -m scripts.health

ingest:
	$(PYTHON) -m scripts.ingest_wb_memory --verbose

ingest-dry:
	$(PYTHON) -m scripts.ingest_wb_memory --dry-run --verbose

query:
	@if [ -z "$(Q)" ]; then echo "用法: make query Q='你的问题'"; exit 1; fi
	$(PYTHON) -m scripts.query "$(Q)" --top-k 5

distill:
	$(PYTHON) -m scripts.distill --index-dir "$$HOME/.workbuddy/rag-index" --verbose

distill-cron:
	$(PYTHON) -m scripts.install_distill_cron

distill-cron-uninstall:
	$(PYTHON) -m scripts.install_distill_cron --uninstall

bootstrap:
	$(PYTHON) -m scripts.install_bootstrap

bootstrap-uninstall:
	$(PYTHON) -m scripts.install_bootstrap --uninstall

bootstrap-run-now:
	$(PYTHON) -m scripts.install_bootstrap --run-now

daemon:
	$(PYTHON) ~/.workbuddy/rag-daemon/daemon.py --watch "$$HOME/.workbuddy/memory"

daemon-install:
	$(PYTHON) ~/.workbuddy/rag-daemon/install.py

daemon-uninstall:
	$(PYTHON) ~/.workbuddy/rag-daemon/uninstall.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf .index/ .pytest_cache/ **/__pycache__/ *.pyc

docker-build:
	docker build -t workbuddy-rag-memory:latest .

docker-run:
	docker run -it --rm \
		-v "$$HOME/.workbuddy:/root/.workbuddy" \
		-v "$$HOME/.cache/huggingface:/root/.cache/huggingface" \
		workbuddy-rag-memory:latest