# Работа с этим репозиторием

## Пользователь
- Пишет по-русски. Отвечать по-русски: кратко, формально, без воды и лишних примеров. Технические термины — латиницей (VLA, cross-attention, guidance, RTC, chunk).
- Уточнять объём, если запуск дорогой (часы GPU) и запрос неоднозначен; иначе делать.

## Проект
- Постановка: `README.md`. Состояние, результаты, грабли, план: `docs/STATUS.md` — читать первым.
- `experiments/h1` — офлайн-проверка сигнала (uv-проект, LeRobot на закреплённом коммите).
- `experiments/libero_pro` — замкнутый цикл: `closed_loop.py` (расписание на каждую среду, задержка, guidance), `planner.py` (Fixed / Budget / Adaptive), `hstar.py`, `scripts/stages/lib.sh` (параллельные запуски с ожиданием памяти, повторами, таймаутом).
- `experiments/robotwin` — RoboTwin 2.0 через серверы симулятора (`robotwin_server.py`, `remote_env.py`, `wire.py`), модель в venv `libero_pro`.
- `experiments/kinetix` — Kinetix через `real-time-chunking-kinetix@9296f31` (`setup.sh` ставит в `third_party/` и качает политики `bc/`); `run_points.py` (через `run.sh`) — naive/RTC в выбранных точках (d, s) на `eval_flow.eval`; `budget.py` (планировщики fixed/budget/adaptive и пошаговый цикл с расписанием на каждую среду, JAX) и `budget_loop.py` (`./run.sh budget_loop.py --config label:key=val,...`, калибровка `--calibrate`).
- Запуски, тесты и установку делает пользователь на сервере; локально только код и git (skill `remote-server`).
- Ветка `h1-offline-check`, PR #1 в `github.com/DanElis/confidence-budget`.

## Сервер
- A100 80 GB, общий с другими пользователями (bentoml, jupyter ~6 GB). Пользователь `eliseevds`, репозиторий `~/work/cv/confidence-budget`. Долгие запуски — в tmux.
- Перед запуском проверять занятость GPU (`nvidia-smi`) и что код свежий (`git log -1`).
- Логи `lib.sh` в UTC (MSK − 3). Предупреждение uv про `VIRTUAL_ENV=.../nlp/.venv` безвредно.

## Правила экспериментов
- Сравнения парные: одинаковые сцены/seed'ы и шум у всех конфигураций; отчёт — Δ с 95% бутстрэп-CI (`libero_pro/scripts/compare.py`).
- Главный признак/правило решения фиксируется до запуска; остальные — с поправкой на множественные сравнения. Подбор параметров — на отдельных данных, отчёт — на других; «лучший из N на тех же эпизодах» подтверждать на новых seed'ах.
- Адаптивные методы сравнивать с постоянной парой при том же среднем горизонте.
- Сбои инфраструктуры не должны попадать в результаты как провалы политики.
- Новый код сначала проверять вхолостую (`--fake-sim --fake-policy`, `EXTRA=...`, заглушки) и тестами.

## Тесты
```bash
cd experiments/libero_pro && uv run python -m pytest -q tests ../robotwin/tests ../h1/tests
cd experiments/kinetix && uv run --project third_party/real-time-chunking-kinetix --with pytest python -m pytest -q tests
```

## Git
- Коммитить и пушить по завершении каждой законченной части; сообщения на английском.
