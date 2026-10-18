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

## Результаты

В `--out`: по файлу `<набор>.json` на набор (успех по эпизодам и задачам, число инференсов, скачок действия при переключении чанка) и `summary.md`. Готовые наборы при перезапуске пропускаются.

`|Δa| at switch` — средняя норма изменения действия в момент переключения на новый чанк, `|Δa| other` — на остальных шагах. Их отношение показывает рывки на стыках чанков: RTC должен приближать его к 1.
