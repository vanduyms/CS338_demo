import streamlit as st
from PIL import Image
import detect
import decode_ocr
import compareDate

def main():
    st.set_page_config(layout="wide")
    st.title("Expiry Date Recognition")

    col1, col2 = st.columns(2)

    res = None
    all_res = []
    with col1:
      col_1, col_2 = st.columns([3, 1])
      with col_1: 
        st.title("Image")
      uploaded_file = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"])
      if uploaded_file is not None:
          image = Image.open(uploaded_file).convert("RGB")
          new_width = int(image.width * 1)
          new_height = int(image.height * 1)
          resized_image = image.resize((new_width, new_height))

          st.image(resized_image, caption="Image Recognition", use_column_width='auto')
          
          with col_2:
            if st.button("Submit"):
              all_box = detect.pred(image)

              for i in all_box:
                res = decode_ocr.OCR(i)
                all_res.append(res)

            col_2.markdown(
              """
              <style>
              .stButton>button {
                margin-top: 30px;
                margin-left: 60px;
              }
              </style>
              """,
              unsafe_allow_html=True
            )
              
    with col2: 
      st.title("Result")
      if len(all_res) > 0: 
        if len(all_res) == 2:
          result = compareDate.get_max_date(all_res[0], all_res[1])
          st.subheader("Ngày hết hạn: " + result)
        else:
          st.subheader("Ngày hết hạn: " + all_res[0])
            

if __name__ == '__main__':
    main()