"""Public browser interface over NeuroCricket's original controllers and Brian2."""
import json
import math
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from neurocricket.config.settings import Settings
from neurocricket.controllers.base import ALGORITHMS, ALGORITHM_LABELS
from neurocricket.simulation.scenarios import SCENARIOS, SCENARIO_LABELS, make_world
from neurocricket.web_runner import run_isolated

COLORS = {"classical": "#64748b", "rate": "#2563eb", "burst": "#d97706"}
SENSOR_LABELS = ["Передний левый", "Центр", "Передний правый", "Боковой левый", "Боковой правый"]
STATUS_LABELS = {"goal": "Цель достигнута", "model_limit": "Цель не достигнута за выбранное время",
    "server_timeout": "Расчёт прерван сервером - результат неполный"}


@st.cache_resource
def simulation_lock():
    # One expensive calculation at a time per server, with no unbounded waiting queue.
    return threading.Lock()


def world_figure(world, results=(), selected=None, frame_index=0):
    fig = Figure(figsize=(10, 5.8), layout="constrained")
    ax = fig.subplots()
    ax.set(xlim=(0, world.width), ylim=(world.height, 0), aspect="equal",
        xlabel="x, м", ylabel="y, м", title=SCENARIO_LABELS[world.name])
    ax.set_facecolor("#f8fafc")
    for rect in world.obstacles:
        ax.add_patch(Rectangle((rect.x, rect.y), rect.w, rect.h, facecolor="#334155"))
    ax.scatter(*world.start, c="#16a34a", s=70, label="Старт", zorder=4)
    ax.add_patch(Circle(world.goal, Settings().goal_radius, facecolor="#bbf7d0", edgecolor="#16a34a"))
    ax.scatter(*world.goal, c="#16a34a", marker="*", s=140, label="Цель", zorder=4)
    for result in results:
        frames = result["frames"]
        stop = frame_index + 1 if result is selected else len(frames)
        xy = np.array([[f["x"], f["y"]] for f in frames[:stop]])
        algorithm = result["metrics"]["algorithm"]
        ax.plot(xy[:, 0], xy[:, 1], color=COLORS[algorithm], lw=2,
            alpha=1 if result is selected else .45, label=ALGORITHM_LABELS[algorithm])
    if selected:
        frame = selected["frames"][frame_index]
        x, y = frame["x"], frame["y"]
        for ex, ey in frame["endpoints"]:
            ax.plot([x, ex], [y, ey], color="#ef4444", lw=1, alpha=.65)
        ax.add_patch(Circle((x, y), selected["settings"]["robot_radius"], facecolor="#0f172a", zorder=5))
        ax.arrow(x, y, .35 * math.cos(frame["theta"]), .35 * math.sin(frame["theta"]),
            width=.025, color="#0f172a", zorder=6)
    ax.grid(alpha=.15)
    ax.legend(loc="upper right", fontsize=9)
    return fig


def result_table(results):
    rows = []
    for result in results:
        m = result["metrics"]
        rows.append({"Алгоритм": ALGORITHM_LABELS[m["algorithm"]], "Результат": STATUS_LABELS[result["status"]],
            "Время до цели, с": m["completion_time_s"], "Время модели, с": round(m["elapsed_simulation_s"], 1),
            "Путь, м": round(m["path_length_m"], 3), "Столкновения": m["collision_count"],
            "Спайки всего": m["spike_count"] if m["algorithm"] != "classical" else None,
            "Входные спайки": m["input_spike_count"] if m["algorithm"] != "classical" else None,
            "Время расчёта на сервере, с": round(m["simulation_runtime_s"], 2)})
    return pd.DataFrame(rows)


def show_neural(result):
    data = result["neural"]
    if data is None:
        st.info("Classical использует правила управления, а не спайковую сеть.")
        return
    n = data["sensor_count"]
    st.caption("Реально зарегистрированная активность Brian2 за последние 5 секунд опыта. Входные, возбуждающие, тормозные и моторные нейроны имеют общую нумерацию.")
    fig = Figure(figsize=(10, 3.3), layout="constrained")
    ax = fig.subplots()
    if data["spikes"]:
        events = np.asarray(data["spikes"])
        ax.scatter(events[:, 0], events[:, 1], marker="|", s=24, color=COLORS[result["metrics"]["algorithm"]])
    for y in (n - .5, 2 * n - .5, 3 * n - .5):
        ax.axhline(y, color="#94a3b8", lw=.7)
    end = result["metrics"]["elapsed_simulation_s"]
    ax.set(xlabel="Время модели, с", ylabel="Номер нейрона", ylim=(-.5, 3 * n + 3.5),
        xlim=(max(0, end - data["window_seconds"]), max(.1, end)))
    st.pyplot(fig)
    neuron = st.selectbox("Мембранный потенциал нейрона", list(range(n, 3 * n + 4)), key="voltage_neuron")
    if data["states"]:
        frame = pd.DataFrame({"Время, с": [row[0] for row in data["states"]],
            "Потенциал, мВ": [row[1][neuron - n] for row in data["states"]],
            "Порог, мВ": [row[1][neuron - n] for row in data["thresholds"]]})
        st.line_chart(frame.set_index("Время, с"))


def show_research():
    st.subheader("Сохранённые результаты исследования")
    st.caption("Это ранее проведённая серия из 150 опытов, а не результат текущего запуска в браузере.")
    st.dataframe(pd.DataFrame({"Алгоритм": ["Classical", "SNN Rate", "SNN Burst"],
        "Достигли цели": ["50/50", "45/50", "47/50"], "Успешность": ["100%", "90%", "94%"],
        "Столкновения": [0, 0, 0]}), hide_index=True)
    st.write("В парном сравнении Burst - Rate: +4 п.п. успешности; разность времени до цели -0,451 с на 45 взаимных успехах с равным весом десяти карт. Изолированный контроль одинакового входа зарегистрировал 335/335 входных спайков.")
    st.info("Это описательные результаты данной конфигурации. Они не доказывают универсальное преимущество Burst. Сеть не обучается; биологическая точность и энергоэффективность не измерялись.")


def main():
    st.set_page_config(page_title="NeuroCricket - интерактивная модель", page_icon="🦗", layout="wide")
    st.title("NeuroCricket")
    st.write("Интерактивная программная модель обхода препятствий: Classical, SNN Rate и SNN Burst.")
    st.caption("Работает прямо в браузере. Rate и Burst рассчитываются в Brian2 на сервере, не заменяются готовой анимацией.")

    with st.sidebar:
        st.header("Параметры опыта")
        with st.form("trial_parameters"):
            scenario = st.selectbox("Карта", SCENARIOS, index=1, format_func=SCENARIO_LABELS.get)
            algorithm = st.selectbox("Алгоритм", ALGORITHMS, index=2, format_func=ALGORITHM_LABELS.get)
            map_seed = int(st.number_input("Seed карты", min_value=0, max_value=999999, value=42, step=1))
            trial_seed = int(st.number_input("Seed нейронного кодирования", min_value=0, max_value=999999, value=42, step=1))
            max_time = st.selectbox("Лимит модельного времени, с", [15, 30, 60], index=2)
            single = st.form_submit_button("Запустить опыт", type="primary")
            compare = st.form_submit_button("Сравнить три алгоритма")
        st.caption("Сравнение выполняет три опыта на одной карте с одинаковыми seed и настройками. Полная серия из 150 опытов здесь не запускается.")
        st.caption("Модельное время отличается от времени ожидания. Если сервер занят, повторите запуск чуть позже.")

    simulation_tab, research_tab, about_tab = st.tabs(["Симуляция", "Результаты исследования", "О модели"])
    with simulation_tab:
        if single or compare:
            lock = simulation_lock()
            if not lock.acquire(blocking=False):
                st.warning("Сервер сейчас выполняет другой опыт. Повторите запуск через некоторое время.")
            else:
                try:
                    results = []
                    choices = ALGORITHMS if compare else (algorithm,)
                    with st.spinner("Выполняется расчёт модели..."):
                        progress = st.progress(0)
                        for i, choice in enumerate(choices):
                            result = run_isolated(scenario, choice, map_seed, trial_seed, max_time)
                            results.append(result)
                            progress.progress((i + 1) / len(choices), text=ALGORITHM_LABELS[choice])
                        progress.empty()
                    st.session_state["results"] = results
                    st.session_state["result_algorithm"] = algorithm
                    for key in list(st.session_state):
                        if key.startswith("timeline_"):
                            del st.session_state[key]
                except (RuntimeError, ValueError) as exc:
                    st.error(str(exc))
                finally:
                    lock.release()

        results = st.session_state.get("results", [])
        if not results:
            st.info("Выберите параметры слева и нажмите «Запустить опыт». После расчёта можно перемещаться по времени и изучать датчики и нейронную активность.")
            st.pyplot(world_figure(make_world(scenario, map_seed)))
        else:
            m = results[0]["metrics"]
            st.subheader("Результат последнего выполненного запуска")
            st.caption(f"Карта: {SCENARIO_LABELS[m['scenario']]} | seed карты: {m['map_seed']} | seed кодирования: {m['trial_seed']} | лимит: {results[0]['settings']['max_time']:g} с. Изменение параметров слева не пересчитывает этот результат до нажатия кнопки.")
            st.dataframe(result_table(results), hide_index=True)
            if any(r["status"] == "server_timeout" for r in results):
                st.warning("Неполные опыты прерваны по лимиту сервера. Их нельзя считать неуспешными завершёнными опытами или использовать для вывода о преимуществе алгоритма.")
            choices = [r["metrics"]["algorithm"] for r in results]
            if st.session_state.get("result_algorithm") not in choices:
                st.session_state["result_algorithm"] = choices[0]
            selected_algorithm = st.selectbox("Просмотреть алгоритм", choices,
                format_func=ALGORITHM_LABELS.get, key="result_algorithm")
            result = next(r for r in results if r["metrics"]["algorithm"] == selected_algorithm)
            frames = result["frames"]
            key = "timeline_" + selected_algorithm
            index = st.slider("Шаг траектории - перемещайте ползунок для просмотра датчиков",
                0, len(frames) - 1, len(frames) - 1, key=key) if len(frames) > 1 else 0
            frame = frames[index]
            st.caption(f"Время модели: {frame['time_s']:.1f} с | скорость: {frame['speed_m_s']:.3f} м/с | {STATUS_LABELS[result['status']]}")
            left, right = st.columns([3, 2])
            with left:
                st.pyplot(world_figure(make_world(m["scenario"], m["map_seed"]), results, result, index))
            with right:
                st.subheader("Датчики в выбранный момент")
                st.dataframe(pd.DataFrame({"Датчик": SENSOR_LABELS,
                    "Расстояние, м": np.round(frame["distances_m"], 3),
                    "Близость, 0-1": np.round(frame["proximity"], 3)}), hide_index=True)
                st.caption("Красные лучи на карте - реальные показания сенсоров. Близость 1 соответствует препятствию у поверхности робота.")
            with st.expander("Показания датчиков за весь опыт"):
                history = pd.DataFrame([f["proximity"] for f in frames], columns=SENSOR_LABELS,
                    index=pd.Index([f["time_s"] for f in frames], name="Время, с"))
                st.line_chart(history)
            with st.expander("Спайки и мембранный потенциал Brian2"):
                show_neural(result)
            st.caption("Пустое время до цели означает, что цель не достигнута или опыт прерван. Один опыт не оценивает общую успешность алгоритма. В навигации входы Rate и Burst могут различаться из-за обратной связи через движение робота.")
            st.download_button("Сохранить результат опыта (необязательно)",
                json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
                file_name="neurocricket_browser_results.json", mime="application/json")
    with research_tab:
        show_research()
    with about_tab:
        st.write("NeuroCricket - программная сенсомоторная модель, а не физиологическая копия сверчка и не аппаратный мемристорный робот.")
        st.write("Classical использует правила. Rate распределяет входные события по окну, Burst группирует их в пачки. Нейронная часть - LIF-сеть Brian2 с фиксированными весами. Общее притяжение к цели и физика одинаковы для трёх алгоритмов.")
        st.write("Шаг управления: 100 мс; шаг Brian2: 1 мс; интегрирование: Euler; backend: numpy CPU. Обучения и подстройки весов в процессе опыта нет.")
        st.caption("На бесплатном хостинге приложение может засыпать после простоя. При первом открытии дождитесь пробуждения или нажмите кнопку пробуждения Streamlit, если она появится.")


if __name__ == "__main__":
    main()
