import time
from datetime import datetime
from typing import Tuple, List, Dict, NoReturn

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
    
    equipment_list = [
        ("УРБ-2А2", "УРБ-4Т", "ПБУ-74"),
        ("МБШ-303", "УБН-Т", "МБШ-812", "МБШ-509", "БКМ-307", "БКМ-303"),
        ("СБУ-115", "СБУ-125"),
    ]
        
    full_data_for_plot: List[Dict[str, Dict[str, Tuple]]] = []
    for idx, fleet_name in enumerate([
            "Серпуховской ПТСН",
            "Челябинский ПТСН",
            "Екатеринбургский ПТСН"
    ]):
        dict_fleet: Dict[str, Dict[str, Tuple]] = prepare_duration_downtime_for_plot(
            fleet_name,
            equipment_list[idx]
        )
        full_data_for_plot.append(dict_fleet)
        
    fleet_names_for_selectbox = [
        list(fleet.keys())[0] for fleet in full_data_for_plot
    ]
    selected_fleet_equipment = st.selectbox(
        "Выберите парк спец. техники",
        fleet_names_for_selectbox
    )
    
    selected_data_for_plot: Dict[str, Tuple] = [
        fleet[selected_fleet_equipment]
        for fleet in full_data_for_plot
        if selected_fleet_equipment in fleet.keys()
    ][0]
    # st.write(selected_data_for_plot["УРБ-2А2"][0])
    duration_downtime_plot(selected_data_for_plot)


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
    

def create_start_date_str() -> str:
    """
    Возвращает файковую дату в формате строки
    """
    fake_day = np.random.randint(1, 30)
    fake_month = np.random.randint(1, 12)
    return f"2020/{fake_month}/{fake_day}"
    

def prepare_duration_downtime_for_plot(
        fleet_name: str,
        equipment_names: List[str]
) -> Dict[str, Dict[str, Tuple]]:
    """
    Возвращает словарь {
        "имя_парка" : {
            "модель_техники" :
                (массив временных меток,
                 массив значений продолжительности простоя),
            ...
        }
    }
    для нескольких машин одного парка
    """
    equipment_num = len(equipment_names)
    lst_dicts_for_equipment = [ # создает список словарей для каждой единицы техники
        dict(
            start_date=create_start_date_str(),
            days=np.random.randint(135, 210)
        ) for _ in range(equipment_num)
    ]
    
    dict_date_idx_dur_dt = {}
    for idx in range(equipment_num):
        df = create_fake_data(**lst_dicts_for_equipment[idx])

        date_idx = df.loc[:, "Дата"]
        duration_dt = Series( # аддитивно-мультипликативная модель временного ряда
            df.loc[:, "Продолжительность простоя"].apply(
                lambda elem: elem.seconds/(60*60)
            )) + trend_for_duration_downtime(
                A = 5*np.random.randn() + 20,
                freq = 10**(-2)*np.random.randn() + 0.05,
                N = lst_dicts_for_equipment[idx]["days"]
            )
        dict_date_idx_dur_dt[equipment_names[idx]] = (date_idx, duration_dt)
    
    return {f"{fleet_name}" : dict_date_idx_dur_dt}
    


def duration_downtime_plot(
    data: Dict[str, Tuple]
) -> NoReturn:
    """
    Отрисовывает несколько временных рядов прдолжительности простоев,
    ассоциированных с заданной маркой спец. техники для
    заданного парка
    """
    equipment_names = list(data.keys())
    
    fig = go.Figure()
    
    for idx, eq_name in enumerate(equipment_names):
        equipment = data[eq_name]
        fig.add_trace(go.Scatter(
            x=equipment[0],
            y=equipment[1].rolling(window=7).mean(),
            name=eq_name,
            # line=...,
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
                **{"time_step" : np.random.randint(5, 100)}
            )
        )
    )


def create_fake_data(
    start_date: str = "2020/7/18",
    days: int = 10
) -> DataFrame:    
    downtime_type = np.array([
        "Аварийный",
        "Регламентный",
        "Логистический",
        "Технологический",
        "Почвенный",
        "Погодный"
    ])
    
    date_idx_list = []
    start_time_end_time_delta = []
    for _ in np.arange(days):
        start_date = pd.Timestamp(
            (pd.Timestamp(start_date) + pd.Timedelta(
                "{time_step:.0f} hours".format(
                    **{"time_step" : np.random.randint(5, 100)}
                )
            )).strftime("%Y/%m/%d")
        )
    
        start_timestamp, timedelta = create_fake_timestamp_and_timedelta()
        date_idx_list.append(start_date)
        start_time_end_time_delta.append(
            (
                datetime.time(start_timestamp),
                datetime.time(start_timestamp + timedelta),
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
            "Дата" : date_idx_list,
            "Тип простоя" : downtime_type[
                np.random.randint(0, len(downtime_type), size=days)
            ]
        }),
        start_time_end_time_delta_df
    ], axis=1)
    
    economic_costs = 10000*np.random.randn(days) + 20000
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