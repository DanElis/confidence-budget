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
- `experiments/kinetix/night_learned.sh` — ночное обучение отдельного state-based контроллера (s,r) при d=2: `budget_head.py`, `learn_budget.py`. On-policy PPO, замороженная flow-политика; validation-сетка констант, 3 training seed, парный test, случайный и близкий по частоте постоянный контроль. Скрипт сначала запускает CPU-тесты, GPU-проверку эквивалентности и полный smoke; инструкция — `experiments/kinetix/README.md`.
- `experiments/kinetix/night_mask.sh` — новый пилот: обучаем только текущую маску r до генерации, `s=4`, `d=2`, `r=0/1/2` (`--mask-before`). Вход MLP — свежее состояние и старый leftover, без нового chunk. По умолчанию 2 уровня/1 seed/8 итераций, детерминированный GPU; контроли — постоянная и случайная маски. Для GPU подтверждён недетерминизм обычного исполнения; deterministic flag устранил его в preflight, но замедлил повтор ~54×.
- `experiments/kinetix/night_joint.sh` — совместный выбор текущего r и следующего s ДО генерации (`--mask-before --strides 2 4`). Вход как у night_mask, без нового chunk; допустимость r зависит от возраста старого плана. Маска допустимых действий сохраняется в PPO batch. Ограничения s и r относятся к разным chunk, старое ограничение суммы не навязывается. Постоянные контроли — только повторяемо допустимые пары. Оригинальный после-генерационный вариант сохранён.
- `experiments/libero_dynamic` — LIBERO-Dynamic (движущиеся объекты, задержка в секундах): `setup.sh` ставит бенчмарк в свой venv (Python 3.10), `policy_server.py` — наша π0.5 на их websocket-протоколе (venv `libero_pro`), `run_dynamic.sh NAME sync|async|rtc K LATENCY [server args]` — их раннер и стратегии без изменений. `correction/` — головка коррекции (A2C2): `prepare_data.py`, `replay.py`, `train_head.py`, `correction_server.py`, стратегия `corrected_rtc_queue.py`, конвейер `night.sh`.
- Запуски, тесты и установку делает пользователь на сервере; локально только код и git (skill `remote-server`).
- `experiments/libero_dynamic/night_budget.sh` — joint before-generation PPO для текущих s,r с полным H=50; заморожены π0.5 и ResNet-18. `learned_budget_queue.py` хранит полный план и явно планирует запрос t+s; `budget_runtime.py` выбирает r до генерации с execution_horizon=d+r. Train/val/test разделены по траекториям, сравнения с постоянными парами и marginal-random; инструкция `BUDGET.md`. Пока только статическая проверка, запуск/тесты на сервере.
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
cd experiments/libero_pro && uv run --with websockets --with msgpack python -m pytest -q ../libero_dynamic/tests
```

## Git
- Коммитить и пушить по завершении каждой законченной части; сообщения на английском.
