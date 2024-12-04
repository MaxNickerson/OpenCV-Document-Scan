# client.py is the Streamlit app that allows users to upload an image and send it to the FastAPI backend for processing. The processed image is then displayed to the user.
import streamlit as st
import requests
from io import BytesIO
from PIL import Image

def main():
    st.title('Document Scanner')

    st.write("Upload an image, and we'll process it for you.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

        # When the user clicks the 'Process Image' button
        if st.button('Process Image'):
            with st.spinner('Processing...'):
                # Send the image to the FastAPI backend
                files = {'file': uploaded_file.getvalue()}
                response = requests.post('http://127.0.0.1:8000/process', files={'file': uploaded_file})

                if response.status_code == 200:
                    # Load the image from the response
                    image = Image.open(BytesIO(response.content))
                    st.success('Image processed successfully!')

                    # Display the processed image
                    st.image(image, caption='Processed Image', use_container_width=True)

                    # Provide a download button
                    img_bytes = BytesIO()
                    image.save(img_bytes, format='JPEG')
                    img_bytes.seek(0)

                    st.download_button(
                        label='Download Processed Image',
                        data=img_bytes,
                        file_name='processed_image.jpg',
                        mime='image/jpeg'
                    )
                else:
                    st.error(f"Error: {response.json().get('error')}")

if __name__ == '__main__':
    main()
