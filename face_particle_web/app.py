import streamlit as st
import cv2
import numpy as np


# ----------------------------
# ページ設定
# ----------------------------

st.set_page_config(
    page_title="fusion of yours",
    page_icon="✨",
    layout="centered"
)


# ----------------------------
# 設定
# ----------------------------

PARTICLE_SIZE = 5
MAX_PARTICLE = 5
GLOW = 20
BACKGROUND = "black"


# ----------------------------
# 色を作る
# ----------------------------

def get_color(value):

    value = np.clip(value, 0, 1)

    colors = np.array([
        [80, 20, 5],
        [180, 50, 10],
        [255, 140, 10],
        [100, 0, 255],
        [225, 140, 255],
        [255, 255, 255]
    ], dtype=np.float32)

    position = value * (len(colors) - 1)

    index = min(int(position), len(colors) - 2)
    amount = position - index

    color = colors[index] * (1 - amount) + \
            colors[index + 1] * amount

    return tuple(int(x) for x in color)


# ----------------------------
# 粒子画像を作る
# ----------------------------

def make_particle_image(image):

    height, width = image.shape[:2]

    scale = min(1.0, 1000 / max(height, width))

    if scale < 1:
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale))
        )

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    ).apply(gray)

    gray = gray.astype(np.float32) / 255.0

    darkness = 1.0 - gray
    darkness = np.power(darkness, 1.3)

    result = np.zeros(
        (height, width, 3),
        dtype=np.float32
    )

    glow_layer = np.zeros_like(result)
    particle_layer = np.zeros_like(result)

    for y in range(
        PARTICLE_SIZE // 2,
        height,
        PARTICLE_SIZE
    ):

        for x in range(
            PARTICLE_SIZE // 2,
            width,
            PARTICLE_SIZE
        ):

            value = darkness[y, x]

            radius = int(
                0.5 + MAX_PARTICLE * value
            )

            if radius < 1:
                radius = 1

            color = get_color(value)

            cv2.circle(
                particle_layer,
                (x, y),
                radius,
                color,
                -1,
                cv2.LINE_AA
            )

            if value > 0.3:

                glow_size = int(
                    1 + value * GLOW * 0.25
                )

                cv2.circle(
                    glow_layer,
                    (x, y),
                    glow_size,
                    color,
                    -1,
                    cv2.LINE_AA
                )

    result = 255 - (
        (255 - result) *
        (255 - glow_layer) / 255
    )

    result = np.maximum(
        result,
        particle_layer
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    # 透過画像を作成
    alpha = np.max(result, axis=2).copy()
    alpha[alpha < 15] = 0

    transparent_result = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2BGRA
    )

    transparent_result[:, :, 3] = alpha

    return result, transparent_result


# ----------------------------
# ウェブページの表示
# ----------------------------

st.title("fusion of yours")

st.write("あなたらしさと光を融合させてみよう")

st.write(
    "画像をアップロードすると、"
    "粒子状の画像に自動で加工されます。"
)

uploaded_file = st.file_uploader(
    "画像を選択してください",
    type=["jpg", "jpeg", "png"]
)


# ----------------------------
# 画像を加工
# ----------------------------

if uploaded_file is not None:

    file_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error("画像を読み込めませんでした。")

    else:

        with st.spinner("粒子を生成しています..."):

            black_result, transparent_result = \
                make_particle_image(image)

        st.success("加工が完成しました！")

        st.subheader("加工結果")

        st.image(
            cv2.cvtColor(
                black_result,
                cv2.COLOR_BGR2RGB
            ),
            caption="黒背景版",
            use_container_width=True
        )

        # 黒背景版のダウンロード
        success, black_encoded = cv2.imencode(
            ".png",
            black_result
        )

        if success:

            st.download_button(
                label="黒背景版を保存",
                data=black_encoded.tobytes(),
                file_name="particle_black.png",
                mime="image/png"
            )

        # 透過版の表示
        st.subheader("背景透過版")

        transparent_rgba = cv2.cvtColor(
            transparent_result,
            cv2.COLOR_BGRA2RGBA
        )

        st.image(
            transparent_rgba,
            caption="背景透過PNG",
            use_container_width=True
        )

        success, transparent_encoded = cv2.imencode(
            ".png",
            transparent_result
        )

        if success:

            st.download_button(
                label="透過版を保存",
                data=transparent_encoded.tobytes(),
                file_name="particle_transparent.png",
                mime="image/png"
            )

        st.caption(
            "アップロードした写真は、このプログラム内で加工するだけで、"
            "画像ファイルとして保存しません。"
        )
