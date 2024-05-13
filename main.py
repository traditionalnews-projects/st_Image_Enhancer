import streamlit as st
from PIL import Image
import os
import numpy as np
import torch
import RRDBNet_arch as arch
import io
import zipfile
import requests
import tempfile

# set page config
st.set_page_config(page_title="Image Enhancer",
                   page_icon=":sparkles:",
                   layout='centered',
                   initial_sidebar_state="auto")

# Use cpu for the process
device = torch.device('cpu')

@st.cache_data()
def download_file(url, destination):
    response = requests.get(url)
    response.raise_for_status()
    with open(destination, 'wb') as f:
        f.write(response.content)

# Replace with your direct download link
url = 'https://drive.google.com/uc?export=download&id=1eOGKAebdrF8nq-Z54ztYwFbYd_yGhK3y'



def enhance_image(image):

    # Create a temporary file
    temp_dir = tempfile.TemporaryDirectory()
    temp_file = tempfile.NamedTemporaryFile(delete=True, dir=temp_dir.name)
    # Close the temporary file
    temp_file.close()
  
    try:
        # Download the file
        download_file(url, temp_file.name)

        # Use the file
        model_path = temp_file.name
        # Load your model here
  
        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        model = model.to(device)

        img = np.array(image)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        
    finally:
        # The temporary file and directory will be deleted when they are closed
        temp_file.close()
        temp_dir.cleanup()

    return Image.fromarray(output.astype('uint8')), model_path

url2 = "https://github.com/xinntao/ESRGAN"
url3 = "https://esrgan.readthedocs.io/en/latest/"




st.title(':sparkles: Pixel Perfection: Your Image Enhancement Wizard',
         help = 'This app increases resolution of an image by 4x, enhancing its details. This results in a 16x increase in total pixels. Read more about the models [here.](%s)" %  url2')
st.write('')
st.write('')

col1, col2, col3 = st.columns([1,0.1,0.5])

# Callback for file uploader
def uploader_callback():
    new_files = st.session_state['uploaded_files']
    for i in new_files:
        st.session_state['uploaded_files'].append(i)

# Initialize session_state
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

# File uploader allows multiple file uploads
new_files = col1.file_uploader("Choose images...", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)


# If there are new files, add them to the session state
if new_files:
    st.session_state['uploaded_files'] += new_files


st.divider()

col1, col2, col3 = st.columns([2,1,2])
# Add a button to clear all files and refresh the page
if col1.button('Clear All Files'):
    st.session_state['uploaded_files'] = []
    st.rerun()

# If there are uploaded files, display them and enhance them when the button is clicked
if st.session_state['uploaded_files']:
    
    if col2.button('Enhance :sparkles:'):
        # Add empty line
        st.write('')
        # Add loading spinner
        with st.spinner('Enhancing...'):
            st.caption('Please wait a few minutes as we work our digital magic and please do not refresh the page until enhancements are done! :magic_wand:')
            enhanced_images, model_path = [enhance_image(Image.open(uploaded_file)) for uploaded_file in st.session_state['uploaded_files']]         
        
            for i, enhanced_image in enumerate(enhanced_images):
                with st.expander(f'Enhanced Image {i+1}'):
                     st.image(enhanced_image, use_column_width=True)
                     # Convert PIL image to byte array
                     img_byte_arr = io.BytesIO()
                     enhanced_image.save(img_byte_arr, format='PNG')
                     img_byte_arr = img_byte_arr.getvalue()
                     # Add a download button for the enhanced image
                     col3.download_button(
                        label=f"Download Enhanced Image {i+1}",
                        data=img_byte_arr,
                        file_name=f'enhanced_{i+1}.png',
                        mime='image/png'
                    )
            st.success('Done!')

            # Create a zip file for all enhanced images
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                for i, enhanced_image in enumerate(enhanced_images):
                    img_byte_arr = io.BytesIO()
                    enhanced_image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    zip_file.writestr(f'enhanced_{i+1}.png', img_byte_arr)
            zip_buffer.seek(0)
            # Add a download button for the zip file
            st.download_button(
                label="Download All Enhanced Images",
                data=zip_buffer,
                file_name='enhanced_images.zip',
                mime='application/zip'
            )

        # Clear the uploaded files
        st.session_state['uploaded_files'] = []
    else:
        for i, uploaded_file in enumerate(st.session_state['uploaded_files']):
            image = Image.open(uploaded_file)
            with st.expander(f'Uploaded Image {i+1}'):
                st.image(image, use_column_width=True)
