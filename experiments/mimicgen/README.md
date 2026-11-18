# MimicGen Threading_D0: проверка идей на другой модели и задаче

Модель — π0.5 `vdb_pi05_threading_d0`, дообученная на MimicGen Threading_D0 (openpi @215abfb2, LoRA слита):
- H = 50, 20 Гц;
- успех на её собственной оценке 40% при s = 10 без задержки (200 эпизодов);
- провалы почти все точностные.

Среда — robosuite 1.4.1 + MimicGen, `Threading_D0`, MuJoCo 2.3.7. Задача длиннее и точнее LIBERO, а чанк в 5 раз длиннее, чем у `pi05_libero` (H = 10). Поэтому у шага s и окна RTC r здесь гораздо больше свободы: при d = 2 допустимо 2 ≤ s ≤ 48.

## Как устроено

Новый клиент не пишется: используется общий `libero_pro/scripts/eval_openpi.py` с `--env mimicgen`. Все режимы работают как на LIBERO: постоянный шаг, задержка, RTC статьи, AAC, gate, бюджеты `--budget`.
- `mimicgen_env.py` — среда (`MimicGenEnv`, интерфейс `closed_loop.run_batch`), сцены и наблюдение.
  - Окружение строится из `env_args` датасета MimicGen. Камеры `agentview` и `robot0_eye_in_hand`, 224×224.
  - Запрос собирается как в нашем датасете: изображения `raw[::-1, ::-1]`, state = позиция схвата, axis-angle, `gripper_qpos` (8). Действия OSC_POSE исполняются как есть.
  - Успех — `env._check_success()`, лимит 400 шагов.
- Сцены. Сцена — это seed глобального генератора numpy, он задаётся перед `env.reset()`. Набор сцен — это «suite»: `mg:y1` или метки стадий `pos:y0.1`. В MimicGen это не уровни сложности, а непересекающиеся наборы: y — подбор, x — test. «Task» — блок сцен, по умолчанию 10 блоков, init state — сцена в блоке. В каждом эпизоде пишется `init_hash` (CRC состояния симулятора после reset). `check_pairing.py RUN...` проверяет, что за каждым элементом во всех прогонах стоит одна и та же сцена.
- Сервер — `libero_pro/scripts/openpi_sample_server.py --config ... --config-module velocity_debias.openpi_configs --checkpoint <JAX> --norm-stats <dir>`. Сэмплеры JAX, поэтому нужен слитый JAX-чекпойнт `.../seed42_10k/9999`, а не PyTorch.
- `profile.sh` переключает на MimicGen все стадии `libero_pro/scripts/stages/openpi_*.sh`. Он задаёт venv сервера и клиента, модель, `EVAL_ENV=mimicgen`, `RUNS=runs/mimicgen`, `MAX_STEPS=400`, `HORIZON=50`. По умолчанию оба процесса используют venv, в котором модель обучалась и оценивалась: `~/work/cv/tmp/velocity-debias/.venv`. Там есть openpi (JAX), robosuite, MimicGen, torch.
- `latency.py` — медиана времени запроса одним клиентом без RTC и с RTC; из неё d = ⌈мс / 50 мс⌉.

## Порядок (из `experiments/libero_pro`, всё в tmux)

Каждая стадия сначала прогоняет smoke. Главные правила записаны в начале каждого скрипта до запуска. Сетки стадий 2–5 выбираются по итогам стадии 1, на y. Итоговые сравнения — на x.

1. База: проверки, задержка, постоянный s без задержки, naive и RTC:
   ```bash
   tmux new -d -s mg_base "source ../mimicgen/profile.sh && bash scripts/stages/mimicgen_base.sh"
   ```
   Первым делом проверить, что `mg_sync_s10` даёт около 40%. Это значит, что среда и наблюдение совпадают с оценкой модели. Если сильно ниже — остановиться и сверить с их скриптом оценки (лимит шагов, ориентация, state).
2. AAC, без задержки и с d (x, контроль — постоянный s той же частоты):
   ```bash
   tmux new -d -s mg_aac "source ../mimicgen/profile.sh && DELAYS='0 2' bash scripts/stages/openpi_aac_async.sh"
   ```
3. Адаптивный RTC (gate), s — из стадии 1:
   ```bash
   source ../mimicgen/profile.sh && STRIDE=<s> DELAY=2 bash scripts/stages/openpi_gate_tune.sh
   source ../mimicgen/profile.sh && STRIDE=<s> DELAY=2 bash scripts/stages/openpi_gate_report.sh
   ```
4. Бюджет шага (option budget, naive), сетка и референсы — из стадии 1:
   ```bash
   source ../mimicgen/profile.sh && DELAY=2 GRID="5 10 20 30" REF_STRIDES="<r1> <r2>" bash scripts/stages/openpi_budget.sh
   ```
5. Бюджет (s, r) с RTC: сначала постоянные пары, потом бюджет:
   ```bash
   source ../mimicgen/profile.sh && DELAY=2 PAIRS="10:5 10:20 20:5 20:10" bash scripts/stages/openpi_rtc_d2.sh
   source ../mimicgen/profile.sh && DELAY=2 PAIR_GRID="..." REF_STRIDES="..." bash scripts/stages/openpi_budget_rtc.sh
   ```

Числа в командах 3–5 — заготовки. Их нужно заменить после стадии 1.

## Не проверено

- Есть ли в venv клиента все пакеты — это проверяет preflight `mimicgen_base.sh`.
- Используют ли робот и размещение объектов в `Threading_D0` глобальный генератор numpy — это проверяет `mimicgen_env.py --check`. Если нет, сцены не воспроизводятся и проверка падает.
- Лимит 400 шагов и формат `env_args` взяты по умолчанию. С их оценкой (40%) не сверялись.
