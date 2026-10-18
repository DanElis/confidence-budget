# Замкнутый контур на LIBERO / LIBERO-Pro

Оценка π0.5 в симуляторе LIBERO с имитацией задержки инференса: синхронное исполнение чанков, наивное асинхронное переключение и RTC. Наборы LIBERO-Pro со сдвигом позиций объектов подключаются к обычному LIBERO из LeRobot без замены пакета.

## Режимы

Симулятор стоит, пока модель считает, поэтому задержка $d$ имитируется расписанием, как в симуляционных экспериментах RTC:

- первый чанк эпизода считается синхронно;
- каждые $s$ шагов запускается инференс по текущему наблюдению; ещё $d$ шагов робот исполняет старый чанк, затем переключается на новый с индекса $d$;
- с guidance новый чанк привязывается к остатку старого (RTC).

| Режим | Флаги |
|---|---|
| синхронный | `--delay 0 --guidance none` |
| наивный асинхронный | `--delay d --guidance none` |
| RTC | `--delay d --guidance lerobot` (или `exact`) |

## LIBERO-Pro

[LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) (Zhou et al., 2025) содержит задачи LIBERO-Object со сдвигом области размещения целевого объекта по x или y. Уровни 0.1…0.5 соответствуют сдвигу на 7…35 см. Здесь они регистрируются как наборы `libero_object_pos_x0.1` … `libero_object_pos_y0.5` (в `--suites`: `pos:x0.2`, `pos:all`).

Опубликованная база π0.5 (AAC, GeoAAC; синхронно, горизонт 5): 53.2% при x0.2, 30.9% в среднем по x0.2–x0.4.

## Установка

Linux, NVIDIA GPU, uv. Компилятор и заголовки EGL не нужны: пакеты `egl-probe` исключены в `pyproject.toml`, симулятору они не требуются.

```bash
cd confidence-budget/experiments/libero_pro
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git third_party/LIBERO-PRO
git -C third_party/LIBERO-PRO checkout eafdb809426b13153aa1e4c42d6601844217dfec
uv sync
uv run pytest -q
```

Если сервер PyTorch недоступен, используйте `uv sync --no-sources` и запускайте через `.venv/bin/python` вместо `uv run python`.

При первом запуске LIBERO создаёт `~/.libero/config.yaml` и скачивает ассеты сцен с Hugging Face в `~/.cache/libero/assets`. Рендеринг без монитора: `MUJOCO_GL=egl` (выставляется скриптом).

## Запуск

Пробный прогон (1 задача, 2 эпизода):

```bash
uv run python scripts/eval_libero.py --suites libero_object,pos:x0.2 --task-ids 0 \
    --episodes-per-task 2 --batch-size 2 --stride 5 --out runs/smoke
```

Синхронная база (воспроизведение опубликованных чисел), шаг 5 и 10, 20 эпизодов на задачу:

```bash
bash scripts/run_baseline.sh
```

Асинхронный режим и RTC, например $s = 10$, $d = 4$:

```bash
uv run python scripts/eval_libero.py --suites pos:x0.2 --stride 10 --delay 4 --guidance none    --out runs/naive_s10_d4
uv run python scripts/eval_libero.py --suites pos:x0.2 --stride 10 --delay 4 --guidance lerobot --out runs/rtc_s10_d4
```

`--record-entropy` сохраняет энтропию cross-attention каждого чанка (для калибровки $h^*$), `--save-actions` — исполненные действия.

## Этапы экспериментов

### Всё одним запуском (рекомендуется)

```bash
tmux new -s all
bash scripts/stages/run_all.sh                 # около 21 ч при JOBS=5
EPISODES_X=20 bash scripts/stages/run_all.sh   # около 11 ч, шире доверительные интервалы на x
```

Скрипт калибрует $h^*$ по данным H1 (`../h1/runs`), затем запускает всю сетку конфигураций сразу на y-уровнях (10 эпизодов на задачу) и на x-уровнях (50), плюс абляции (20). В конце `scripts/select_report.py` выбирает $s^*$, $\lambda^*$ и $z_{\max}^*$ **только по y** и пишет `runs/report.md`: основная таблица — выбранные конфигурации на x, полная сетка на x — справочно. Правило выбора зафиксировано заранее, поэтому результат на x честный, хотя считаются все варианты. Ход работы: `runs/all.log`; повторный запуск пропускает готовое. Отчёт можно собрать и в процессе: `uv run python scripts/select_report.py --runs runs --out runs/report.md`.

### По этапам (вручную, дешевле по вычислениям)

Параметры подбираются на уровнях сдвига по y (y0.1–y0.3), итоговые числа считаются на исходном `libero_object` и x0.1–x0.3 (опубликованные результаты сопоставимы на x0.2 и x0.3). Скрипты лежат в `scripts/stages/`, запускаются из `experiments/libero_pro`. Каждый запуск пишет `runs/<имя>/`, готовые наборы пропускаются; наборы считаются параллельно (`JOBS`, по умолчанию 4 процесса, около 10 ГБ видеопамяти на процесс). Значения, выбранные на одном этапе, передаются в следующий переменными окружения.

| Этап | Команда | Что выбрать по итогам |
|---|---|---|
| 1. Асинхронная база, $d = 4$ | `bash scripts/stages/stage1_async_baselines.sh` | `S_BEST` — лучший шаг RTC в `runs/stage1_d4.md` |
| 1b. Другие задержки | `S_BEST=… bash scripts/stages/stage1_async_baselines.sh` | — |
| 2. Калибровка $h^*$ | `S_BEST=… bash scripts/stages/stage2_calibrate_hstar.sh` | правило в `runs/hstar.json`; оно должно обгонять константу |
| 3. Сетка по $\lambda$ | `S_BEST=… bash scripts/stages/stage3_lambda_grid.sh` | `LAM_BEST` по `runs/stage3_d4.md` |
| 4. $\lambda$ по энтропии | `S_BEST=… LAM_BEST=… bash scripts/stages/stage4_lambda_entropy.sh` | `ZMAX_BEST` по `runs/stage4_d4.md` |
| 5. Финальная оценка | `S_BEST=… S_SYNC=… LAM_BEST=… ZMAX_BEST=… bash scripts/stages/stage5_final.sh` | таблицы `runs/final_d{2,4,8}.md` |
| 6. Абляции | `S_BEST=… LAM_BEST=… ZMAX_BEST=… bash scripts/stages/stage6_ablations.sh` | `runs/ablations_d4.md` |

`S_SYNC` — лучший синхронный шаг (5 или 10) из базового прогона. Если сервер PyTorch недоступен, перед запуском задайте `PY=.venv/bin/python`.

Таблицы строит `scripts/compare.py`: успех по наборам, разность с эталоном (RTC с $s^*$), парная по начальным состояниям, с 95% бутстрэп-интервалом, число вызовов модели, SPARC по командам скорости, скачок действия на стыках, средний шаг $s$.

Бюджет уверенности (`--planner budget`): после каждого чанка $h^*$ считается по энтропии (`hstar.py`), затем $s = d + \lambda\rho$, $r = r_{\min} + (1-\lambda)\rho$, $\rho = h^* - 2d - r_{\min}$ (`planner.py`); следующий чанк генерируется с RTC и мягкой маской длины $r$. У каждой среды своё расписание; среды, которым чанк нужен на одном шаге, считаются одним вызовом модели.

## Результаты

В `--out`: по файлу `<набор>.json` на набор (успех по эпизодам и задачам, число инференсов, скачок действия при переключении чанка) и `summary.md`. Готовые наборы при перезапуске пропускаются.

`|Δa| at switch` — средняя норма изменения действия в момент переключения на новый чанк, `|Δa| other` — на остальных шагах. Их отношение показывает рывки на стыках чанков: RTC должен приближать его к 1.
