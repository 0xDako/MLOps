"""Streamlit-интерфейс DigitPad: рисование, фото и метрики."""

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from client import ApiError, get_models, predict_digit, predict_photo
from formatting import build_metrics_rows, build_photo_comparison_rows, draw_boxes

st.set_page_config(
    page_title="DigitPad", layout="wide", initial_sidebar_state="collapsed"
)
st.markdown(
    "<style>[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True
)
st.title("DigitPad")


def _probability_chart(probabilities: list[float]) -> alt.Chart:
    data = pd.DataFrame({"цифра": range(10), "вероятность": probabilities})
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("цифра:O"),
            y=alt.Y("вероятность:Q", scale=alt.Scale(domain=[0, 1])),
        )
    )


tab_draw, tab_photo, tab_metrics = st.tabs(["Нарисовать", "Фото", "Метрики"])

with tab_draw:
    st.subheader("Нарисуйте цифру")
    canvas = st_canvas(
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        return_image_data=True,
        key="draw_canvas",
    )

    if st.button("Распознать", key="recognize_digit"):
        if canvas.image_data is None:
            st.warning("Сначала нарисуйте цифру")
        else:
            image = canvas.image_data.astype("uint8")
            _, buffer = cv2.imencode(".png", image)
            try:
                result = predict_digit(buffer.tobytes(), model="all")
            except ApiError as exc:
                st.error(f"API недоступен: {exc}")
            else:
                columns = st.columns(len(result["predictions"]))
                for column, (name, prediction) in zip(
                    columns, result["predictions"].items(), strict=True
                ):
                    with column:
                        st.metric(
                            name, prediction["digit"], f"{prediction['confidence']:.1%}"
                        )
                        st.altair_chart(
                            _probability_chart(prediction["probabilities"]),
                            use_container_width=True,
                        )

with tab_photo:
    st.subheader("Фото с цифрами")
    source = st.radio("Источник фото", ["Файл", "Камера"], horizontal=True)

    image_bytes = None
    if source == "Файл":
        uploaded = st.file_uploader("Загрузите фото", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
    else:
        camera_image = st.camera_input("Сфотографируйте")
        if camera_image is not None:
            image_bytes = camera_image.getvalue()

    if image_bytes is not None:
        try:
            result = predict_photo(image_bytes)
        except ApiError as exc:
            st.error(f"API недоступен: {exc}")
        else:
            if not result["boxes"]:
                st.warning("Цифры на фото не найдены")
            else:
                array = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(array, cv2.IMREAD_COLOR)

                primary_model = (
                    "cnn_robust"
                    if "cnn_robust" in result["models"]
                    else next(iter(result["models"]))
                )
                primary = result["models"][primary_model]
                annotated = draw_boxes(
                    image, result["boxes"], primary["digits"], primary["confidences"]
                )
                st.image(
                    annotated,
                    channels="BGR",
                    caption=f"Рамки по модели {primary_model}",
                )

            st.table(build_photo_comparison_rows(result))

with tab_metrics:
    st.subheader("Сравнение моделей")
    try:
        metrics = get_models()
    except ApiError as exc:
        st.error(f"API недоступен: {exc}")
    else:
        st.table(build_metrics_rows(metrics))
        for name, values in metrics.items():
            if "confusion_matrix" in values:
                st.write(f"Матрица ошибок — {name}")
                st.table(values["confusion_matrix"])
