import time
from datetime import datetime
from typing import Tuple, NoReturn

import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pandas import DataFrame, Series

import streamlit as st
from streamlit_folium import folium_static

from css import annotation_css, annotation_css_sidebar, header_css, logo_css

title_app = "Панель инструментов анализа простоев техники"

st.set_page_config(
    layout="wide",
    page_title=title_app,
    initial_sidebar_state="expanded",
)


def main_elements():
    """
    Создает шапку страницы
    """
    # === Верхняя часть шапки ===
    row1_1, row1_2 = st.beta_columns([2, 1])

    with row1_1:
        logo_css("АО РОСГЕОЛОГИЯ", align="left", clr="#1E2022", size=33)
        # logo_css(
        #     "<i>Департамент по восстановлению и утилизации<br>трубной продукции</i>",
        #     align="left",
        #     clr="#52616B",
        #     size=20,
        # )

    with row1_2:
        pass

    header_css(
        f"<i>{title_app}</i>",
        align="left",
        clr="#52616B",
        size=26,
    )

    row1, row2 = st.beta_columns([2, 1])
    with row1:
        # st.markdown(
        #     "_Ниже приводится упрощенное представление карты. Коммерческий вариант "
        #     "приложения на усмотрение Заказчика может быть "
        #     "построен с помощью [MapBox](https://www.mapbox.com/maps)_"
        # )
        pass

    with row2:
        pass

    # === Нижняя часть шапки ===
    row2_1, row2_2 = st.beta_columns([3, 1])

    with row2_1:
        uploaded_file = st.file_uploader(
            "Загрузить данные...",
            type=["xls", "xlsx", "csv", "tsv"],
            accept_multiple_files=False,
        )
        if uploaded_file is not None:
            st.error("В демонстрационной версии приложения оперировать "
                     "можно только тестовым набором данных. "
                     "Загрузить тестовый набор можно кнопкой "
                     "'Создать тестовый набор данных' на боковой панели"
            )
    
    
def sidebar_elements():
    annotation_css_sidebar(
        "Сводные отчеты",
        align="left",
        size=18,
        clr="#1E2022",
    )
    selected_type_report = st.sidebar.radio(
        "Выберите формат отчета",
        ("Excel (табличное представление)", "LaTeX (аналитика)"),
    )

    if st.sidebar.button("Выгрузить отчет"):
        pass

    annotation_css_sidebar(
        "Работа с базой данных маркеров слоя",
        align="left",
        size=18,
        clr="#1E2022",
    )
    
    if st.sidebar.button("Создать тестовый набор данных"):
        duration_downtime_plot()
        
    selected_fleet_equipment = st.selectbox(
        "Выберите парк спец. техники",
        []
    )
    
    
def time_step_gen_base(A: float = 100, b: float = 1.56) -> float:
    return A*np.random.weibull(b)


def trend_for_duration_downtime(
        A: float = 5.0,
        b: float = 2.0,
        freq: float = 0.05,
        offset: float = 20,
        N: int = 100
) -> Series:
    """
    Возвращает детерминированный тренд
    """
    x = np.arange(N)
    trend = A*np.exp(-b*x)*np.sin(freq*x) + offset
    return Series(trend)


def var_for_duration_downtime():
    """
    Возвращает детерминированную модель изменения
    дисперсии временного ряда
    """
    pass
    

def duration_downtime_plot() -> NoReturn:
    """
    Отрисовывает несколько временных рядов прдолжительности проестоев
    """
    df_args = (
        dict(start_date="2020/5/14", days=250, seed=42),
        dict(start_date="2020/7/18", days=200, seed=2),
        dict(start_date="2020/4/3", days=180, seed=4242),
    )
    
    labels = (
        "Установка УРБ-2А2",
        "Установка УРБ-4Т",
        "Установка ПБУ-74"
    )
    
    line_attrs = (
        dict(color="firebrick", width=1.5),
        dict(color="royalblue", width=1.5),
        dict(color="green", width=1.5)
    )
    
    fig = go.Figure()
    
    for idx in range(3):
        df = create_fake_data(**df_args[idx])
        duration_dt = Series( # аддитивно-мультипликативная модель временного ряда
            df.loc[:, "Продолжительность простоя"].apply(
                lambda elem: elem.seconds/(60*60)
            )
        ) + trend_for_duration_downtime(
            A = 5*np.random.randn() + 20,
            freq = 10**(-2)*np.random.randn() + 0.05,
            N = df_args[idx]["days"]
        )
        
        date_idx = df.loc[:, "Дата"]
        
        fig.add_trace(go.Scatter(
            x=date_idx,
            y=duration_dt.rolling(window=7).mean(),
            name=labels[idx],
            line=line_attrs[idx],
            mode="lines+markers"
        ))
        
    fig.update_layout(
        title=dict(
            text="<i>Временной ряд значений продолжительности простоя</i>",
            font=dict(
                family="Arial",
                size=18,
                color="#52616B",
            )
        ),
        xaxis_title="<i>Временная метка</i>",
        yaxis_title="<i>Продолжительность простоя, час</i>",
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=15,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=1.5,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=15,
                color='rgb(82, 82, 82)',
            ),
        ),
        autosize=False,
        margin=dict(
            autoexpand=False,
            l=70,
            r=10,
            t=50,
        ),
        showlegend=True,
        plot_bgcolor='white',
        legend_title_text='<i>Единицы спец. техники</i>',
        legend=dict(
            orientation="v",
            yanchor = "bottom",
            y = 0.01,
            xanchor = "right",
            x = 0.99,
            font=dict(
                family="Arial",
                size=12,
                color="black"
            ),
        ),
        font=dict(
            family="Arial",
            size=13,
        )
    )
    # fig.update_xaxes(title_font_family="Courier New, monospace")
    
    st.plotly_chart(fig, use_container_width=True)


def create_fake_timestamp_and_timedelta() -> Tuple[pd.Timestamp, pd.Timedelta]:
    fake_hour = np.random.randint(0, 23)
    fake_minute = np.random.randint(0, 59)
    fake_second = np.random.randint(0, 59)
    return (
        pd.Timestamp( # файковая временная метка
            datetime(2020, 1, 1, fake_hour, fake_minute, fake_second)),
        pd.Timedelta( # файковое смещение по времени
            "{time_step} hours".format(
                **{"time_step" : time_step_gen_base(80, 2.5)}
            )
        )
    )


@st.cache
def create_fake_data(
    start_date: str = "2020/07/18",
    days: int = 10,
    seed: int = 42
) -> DataFrame:    
    downtime_type = np.array([
        "Аварийный",
        "Регламентный",
        "Логистический",
        "Технологический",
        "Почвенный",
        "Погодный"
    ])
    
    datetime_idx_list = []
    start_time_end_time_delta = []
    for _ in np.arange(days):
        start_date = pd.Timestamp(
            (pd.Timestamp(start_date) + pd.Timedelta(
                "{time_step:.0f} hours".format(
                    **{"time_step" : time_step_gen_base()}
                )
            )).strftime("%Y/%m/%d"))
    
        timestamp, timedelta = create_fake_timestamp_and_timedelta()
        datetime_idx_list.append(start_date)
        start_time_end_time_delta.append(
            (
                datetime.time(timestamp),
                datetime.time(timestamp + timedelta),
                timedelta
            )
        )
        
    start_time_end_time_delta_df = DataFrame(
        start_time_end_time_delta, columns = [
            "Начало временного интервала",
            "Конец временного интервала",
            "Продолжительность простоя"
        ]
    )
    
    data_fake = pd.concat([
        DataFrame({
            "Дата" : datetime_idx_list,
            "Тип простоя" : downtime_type[
                np.random.RandomState(seed).randint(0, len(downtime_type), size=days)
            ]
        }),
        start_time_end_time_delta_df
    ], axis=1)
    
    economic_costs = 10000*np.random.RandomState(seed).randn(days) + 20000
    data_fake = pd.concat([
        data_fake,
        DataFrame(
            economic_costs,
            columns=["Экономические потери, руб."]
    )], axis=1)
    return data_fake



if __name__ == "__main__":
    main_elements()
    sidebar_elements()