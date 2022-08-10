import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("画像認識アプリ")
st.sidebar.write("犬と猫の画像認識モデルを使って判定する。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースをせんたくしてください。",
                             ("画像をアップロード", "カメラで撮影"))

if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中…"):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        #予測
        results = predict(img) 

        #結果の表示
        st.subheader("判定結果")
        n_top = 2
        for result in results[:n_top]:
            st.write(str(round(result[2]*100, 2)) + "%の確率で" + result[0] + "です。")

        #円グラフ
        pie_labels = [result[1] for result in results[:n_top]]
        pie_labels.append("others")
        pie_probs = [result[2] for result in results[:n_top]]
        pie_probs.append(sum([result[2] for result in results[n_top:]]))
        fig, ax = plt.subplots()
        wedgeprops = {"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)
        st.pyplot(fig)
