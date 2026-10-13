
import streamlit as st
import cv2
import numpy as np

# ---------------------------------
# ページ設定
# ---------------------------------
st.set_page_config(
    page_title="FUSION OF YOURS",
    page_icon="✨",
    layout="centered"
)

# ---------------------------------
# サイトデザイン
# ---------------------------------
st.markdown("""
<style>

/* ページ全体 */
.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(35, 65, 125, 0.20),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 40%,
            rgba(180, 65, 110, 0.12),
            transparent 35%
        ),
        #080D1B;
    color: #F5F6FC;
}

/* コンテンツの幅 */
.block-container {
    max-width: 780px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

/* 文字 */
.stApp p {
    color: #C3CBE0;
    line-height: 1.9;
}

.stApp h1,
.stApp h2,
.stApp h3 {
    color: #FFFFFF;
}

/* アップロード欄 */
[data-testid="stFileUploader"] {
    background: rgba(25, 37, 65, 0.65);
    border: 1px dashed #697B9F;
    border-radius: 14px;
    padding: 18px;
}

/* アップロード欄の文字 */
[data-testid="stFileUploader"] label {
    color: #FFFFFF;
}

/* ボタン */
.stButton button,
.stDownloadButton button {
    background: linear-gradient(
        100deg,
        #D879A0,
        #F1A0B6
    );
    color: #11182A;
    border: none;
    border-radius: 9px;
    font-weight: bold;
}

/* ボタンのホバー */
.stButton button:hover,
.stDownloadButton button:hover {
    background: linear-gradient(
        100deg,
        #F19ABB,
        #FFC0CE
    );
    color: #11182A;
    border: none;
}

/* 区切り線 */
hr {
    border-color: #34415B;
}

/* キャプション */
.stCaption {
    color: #8795B2;
}

/* ダウンロードボタン */
.stDownloadButton button {
    width: 100%;
}

/* スマートフォン対応 */
@media (max-width: 600px) {
    .block-container {
        padding-top: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------
# 粒子加工の設定
# ---------------------------------
PARTICLE_SIZE = 5
MAX_PARTICLE = 5
GLOW = 20
BACKGROUND = "black"


# ---------------------------------
# 粒子の色
# ---------------------------------
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

    index = min(
        int(position),
        len(colors) - 2
    )

    amount = position - index

    color = (
        colors[index] * (1 - amount)
        + colors[index + 1] * amount
    )

    return tuple(int(x) for x in color)


# ---------------------------------
# 粒子画像の生成
# ---------------------------------
def make_particle_image(image):

    height, width = image.shape[:2]

    scale = min(
        1.0,
        1000 / max(height, width)
    )

    if scale < 1:

        image = cv2.resize(
            image,
            (
                int(width * scale),
                int(height * scale)
            )
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
        (255 - result)
        * (255 - glow_layer)
        / 255
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

    # 背景透過版
    alpha = np.max(
        result,
        axis=2
    ).copy()

    alpha[alpha < 15] = 0

    transparent_result = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2BGRA
    )

    transparent_result[:, :, 3] = alpha

    return result, transparent_result


# ---------------------------------
# ヘッダー
# ---------------------------------
st.markdown("""
<div style="
    text-align:center;
    margin-bottom:50px;
">

    <div style="
        color:#91A5C9;
        font-size:11px;
        letter-spacing:5px;
        margin-bottom:20px;
    ">
        DIGITAL ART PROJECT
    </div>

    <div style="
        font-size:clamp(28px, 6vw, 46px);
        font-weight:700;
        letter-spacing:3px;
        color:#FFFFFF;
    ">
        FUSION OF YOURS
    </div>

    <div style="
        color:#F19ABB;
        font-size:16px;
        letter-spacing:2px;
        margin-top:17px;
    ">
        あなたの顔を、光の粒子に変える。
    </div>

</div>
""", unsafe_allow_html=True)


# ---------------------------------
# コンセプト説明
# ---------------------------------
st.markdown("""
<div style="
    background:rgba(28, 40, 68, 0.55);
    border:1px solid rgba(120, 145, 190, 0.25);
    border-radius:15px;
    padding:27px;
    margin-bottom:40px;
">

    <div style="
        color:#F19ABB;
        font-size:11px;
        letter-spacing:3px;
        margin-bottom:15px;
    ">
        ABOUT THE PROJECT
    </div>

    <div style="
        color:#FFFFFF;
        font-size:20px;
        font-weight:600;
        margin-bottom:15px;
    ">
        小さな光が集まり、ひとつの顔になる。
    </div>

    <div style="
        color:#C3CBE0;
        font-size:14px;
        line-height:2;
    ">
        無数の細やかな粒子が集まり、
        互いに影響しながら光を生み出していく。
        核融合反応のように、
        あなたの顔を新しい光のかたちへと変換します。
    </div>

</div>
""", unsafe_allow_html=True)


# ---------------------------------
# アップロード案内
# ---------------------------------
st.markdown("""
<div style="margin-bottom:15px;">

    <div style="
        color:#F19ABB;
        font-size:11px;
        letter-spacing:3px;
    ">
        01 / UPLOAD YOUR PHOTO
    </div>

    <div style="
        color:#FFFFFF;
        font-size:19px;
        font-weight:600;
        margin-top:10px;
    ">
        あなたの写真を選んでください
    </div>

    <div style="
        color:#AAB7D0;
        font-size:13px;
        margin-top:7px;
    ">
        顔がはっきり写った写真がおすすめです。
    </div>

</div>
""", unsafe_allow_html=True)


# ---------------------------------
# 写真アップロード
# ---------------------------------
uploaded_file = st.file_uploader(
    "写真をアップロード",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------
# 加工と結果表示
# ---------------------------------
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

        st.error(
            "画像を読み込めませんでした。"
        )

    else:

        with st.spinner(
            "光の粒子を生成しています..."
        ):

            black_result, transparent_result = (
                make_particle_image(image)
            )

        st.success(
            "粒子の生成が完成しました！"
        )

        st.markdown("""
        <div style="
            color:#F19ABB;
            font-size:11px;
            letter-spacing:3px;
            margin-top:35px;
        ">
            02 / YOUR RESULT
        </div>

        <div style="
            color:#FFFFFF;
            font-size:21px;
            font-weight:600;
            margin-top:10px;
            margin-bottom:20px;
        ">
            あなたから生まれた光
        </div>
        """, unsafe_allow_html=True)

        # 黒背景版
        st.subheader("黒背景版")

        st.image(
            cv2.cvtColor(
                black_result,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

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

        # 背景透過版
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


# ---------------------------------
# フッター
# ---------------------------------
st.markdown("""
<div style="
    text-align:center;
    margin-top:70px;
    padding-top:25px;
    border-top:1px solid #34415B;
">

    <div style="
        color:#91A5C9;
        font-size:11px;
        letter-spacing:4px;
    ">
        FUSION OF YOURS
    </div>

    <div style="
        color:#64728F;
        font-size:10px;
        margin-top:12px;
        letter-spacing:2px;
    ">
        A DIGITAL PARTICLE EXPERIMENT
    </div>

</div>
""", unsafe_allow_html=True)
