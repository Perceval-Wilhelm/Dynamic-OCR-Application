import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import json
import time
import string
import os
import fitz
import io
import random
from datetime import datetime
from PIL import Image, ImageFile
import cv2
# from paddleocr import PaddleOCR
from OCR_process import ocr_parser
from algorithm import find_key, find_key_value_from_large_set 

# Set to allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Check if 'ocr' is in the session state, if not, initialize PaddleOCR
# if 'ocr' not in st.session_state:
#     st.session_state.ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Access the PaddleOCR instance from the session state
# ocr = st.session_state.ocr

# Define the save path for templates
save_path = "template"

# Set Streamlit page configuration to wide layout
st.set_page_config(layout="wide")

# Add custom styling using HTML/CSS to the Streamlit app
st.markdown("""
    <style>
        .css-18e3th9 {
            padding-top: 0rem;
            padding-bottom: 10rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        .css-1d391kg {
            padding-top: 3.5rem;
            padding-right: 1rem;
            padding-bottom: 3.5rem;
            padding-left: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# TemplateProcessor class handles the creation of templates
class TemplateProcessor:
    def __init__(self):
        """
        Constructor for TemplateProcessor class.

        Initializes attributes for canvas result, input image, and save path.
        """
        self.canvas_result = None  # Result of the canvas drawing
        self.image = None  # Input image
        self.save_path = save_path  # Path to save the JSON file

    def draw_canvas(self, uploaded_file):
        """
        Function to draw a canvas for selecting key-value pairs in the image.

        Parameters:
        - uploaded_file: Uploaded image file.

        Draws the canvas using Streamlit's st_canvas, allowing the user to select
        key-value pairs on the image. Displays the canvas and relevant information.
        """
        unique_key = f"canvas_{uploaded_file.name}"
        self.image = Image.open(uploaded_file)
        st.subheader("Draw OCR")
        
        col1, col2 = st.columns([6, 4])

        # Col1: Draw canvas
        # Set up the canvas for drawing key-value pairs on the image
        original_width, original_height = self.image.size
        max_canvas_width = 600
        scale_factor = max_canvas_width / original_width
        canvas_height = int(original_height * scale_factor)

        with col1:
            # Initialize the canvas using Streamlit's st_canvas
            self.canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#e00",
                background_image=self.image,
                width=max_canvas_width,
                height=canvas_height,
                drawing_mode="rect",
                display_toolbar=True,
                key=unique_key,
            )
            # Save button
            self.save_button()

        # Col2: Display information of the document
        if self.canvas_result.json_data is not None:
            objects = self.canvas_result.json_data["objects"]
            for i in range(0, len(objects), 2):
                if i+1 < len(objects):
                    key_data = objects[i]
                    value_data = objects[i+1]
                    # Process and display key-value pairs in Col2
                    self.process_key_value_pair(key_data, value_data, col2, scale_factor, pair_num=i//2 + 1)

    def process_key_value_pair(self, key_data, value_data, col, scale_factor, pair_num):
        """
        Function to process a key-value pair and display information in Streamlit columns.

        Parameters:
        - key_data: Data for the key section.
        - value_data: Data for the value section.
        - col: Streamlit column for displaying information.
        - scale_factor: Scaling factor for converting canvas coordinates to image coordinates.
        - pair_num: Number of the key-value pair.

        Processes the key and value sections, extracts relevant information,
        and displays it in the specified Streamlit column.
        """
        # Extract information about the key and value sections
        key_text, key_type, key_coordinates, key_size = self.process_image_section(key_data, scale_factor)
        value_text, value_type, value_coordinates, value_size = self.process_image_section(value_data, scale_factor)

        with col:
            st.subheader(f"Pair {pair_num}")

            # Input fields for key, value, and label
            st.text_input(f"Key {pair_num}", value=key_text, key=f"key_{pair_num}")
            st.session_state[f"key_type_{pair_num}"] = key_type
            st.write(f"Key Type {pair_num}: {key_type}")

            st.text_input(f"Value {pair_num}", value=value_text, key=f"value_{pair_num}")
            st.session_state[f"value_type_{pair_num}"] = value_type
            st.write(f"Value Type {pair_num}: {value_type}")

            st.text_input(f"Label {pair_num}", key=f"label_{pair_num}")

            # Store key and value coordinates in session state
            st.session_state[f"key_coordinates_{pair_num}"] = key_coordinates
            st.session_state[f"value_coordinates_{pair_num}"] = value_coordinates

            # Store key and value sizes in session state
            st.session_state[f"key_size_{pair_num}"] = key_size
            st.session_state[f"value_size_{pair_num}"] = value_size

            # Display key and value coordinates and size
            st.write(f"Key coordinates (top_left_x/top_left_y): {key_coordinates}, Key size (width/height): {key_size}")
            st.write(f"Value coordinates (top_left_x/top_left_y): {value_coordinates}, Value size (width/height): {value_size}")

            col.markdown("---")
    
    def process_image_section(self, rect_data, scale_factor):
        """
        Function to process an image section, detect text, data type, and return coordinates.

        Parameters:
        - rect_data: Data for a rectangular section on the canvas.
        - scale_factor: Scaling factor for converting canvas coordinates to image coordinates.

        Processes the specified image section, detects text using OCR, determines
        the data type, and returns coordinates and size information.
        """
        top_left_x = int(rect_data["left"] / scale_factor)
        top_left_y = int(rect_data["top"] / scale_factor)
        width = int(rect_data["width"] / scale_factor)
        height = int(rect_data["height"] / scale_factor)

        # Crop the original image based on the calculated coordinates
        cropped_image = self.image.crop((top_left_x, top_left_y, top_left_x + width, top_left_y + height))
        cropped_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

        # Detect text using OCR (Optical Character Recognition)
        parser = ocr_parser()
        text_list = parser.extract_info(cropped_image)
        print("Text: ", text_list)
        # all_text = [list(text.keys())[0] for text in text_list] if text_list else []
        try: 
            all_text = ' '.join([line[1] for line in text_list])

            # Detect data type based on the content of the text
            if all_text.replace('.', '', 1).isdigit():
                data_type = 'number'
            elif self.is_date(all_text):
                data_type = 'date'
            else:
                data_type = 'text'
        except Exception:  # Corrected the exception type from 'catch' to 'Exception'
            all_text = None
            data_type = None

        # Coordinates and size information
        coordinates = [top_left_x, top_left_y]
        size = [width, height]
        return all_text, data_type, coordinates, size

    # Function to check if a string represents a date
    @staticmethod
    def is_date(string):
        """
        Function to check if a string represents a date.

        Parameters:
        - string: Input string to check.

        Checks if the input string can be parsed as a date using different date formats.
        Returns True if it's a date, otherwise returns False.
        """
        string = string.strip()
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                datetime.strptime(string, fmt)
                return True
            except ValueError:
                continue
        return False
    
    def update_data_type(self, input_key, type_key):
        """
        Function to update data type based on user input.

        Parameters:
        - input_key: Key for the input text field.
        - type_key: Key for storing the detected data type.

        Updates the data type based on the user's input and stores it in the session state.
        """
        text = st.session_state[input_key]
        detected_type = self.detect_data_type(text)
        st.session_state[type_key] = detected_type
    
    def save_button(self):
        """
        Function to create a save button in the Streamlit interface.

        Creates a save button in the Streamlit interface with an input field for the file name.
        """
        file_name = st.text_input("Enter file name for saving:", key="json_file_name")  

        if st.button('Save to JSON', key='save_button'):
            if not file_name.strip():
                st.error("Please enter a file name.")
                return

            complete_path = os.path.join(self.save_path, file_name + '.json')
            if os.path.exists(complete_path):
                st.error("File already exists. Please enter a different name.")
                return

            self.save_to_json(complete_path)

    def save_to_json(self, file_path):
        """
        Function to save data to a JSON file.

        Parameters:
        - file_path: Path to save the JSON file.

        Saves the processed data to a JSON file at the specified path.
        """
        num_pairs = len(self.canvas_result.json_data["objects"]) // 2
        data_to_save = {}
        root = np.array([0, 0]) 
        ox = np.array([5, 0]) 
        punctuations = string.punctuation

        for i in range(1, num_pairs + 1):
            # Retrieve information for each key-value pair and calculate additional metrics
            key = "".join([char for char in st.session_state.get(f"key_{i}", "") if char not in punctuations])
            value = "".join([char for char in st.session_state.get(f"value_{i}", "") if char not in punctuations])
            key_cor = st.session_state.get(f"key_coordinates_{i}", {})
            value_cor = st.session_state.get(f"value_coordinates_{i}", {})
            key_cor_array = np.array(key_cor)  
            value_cor_array = np.array(value_cor)
            ox_key = [key_cor[0] + 5, key_cor[1]]
            ox_key_array = np.array(ox_key)  
            data_to_save[st.session_state.get(f"label_{i}", "")] = {
                "root": root.tolist(),
                "ox": ox.tolist(),
                "key": key,
                "key_type": st.session_state.get(f"key_type_{i}", ""),
                "key_coordinates": key_cor,
                "key_size": st.session_state.get(f"key_size_{i}", {}),
                "ox_key": ox_key,
                "value": value,
                "value_type": st.session_state.get(f"value_type_{i}", ""),
                "value_coordinates": value_cor,
                "value_size": st.session_state.get(f"value_size_{i}", {}),
                "label": st.session_state.get(f"label_{i}", ""),
                "magnitude_root_key": np.linalg.norm(key_cor_array - root),
                "magnitude_root_value": np.linalg.norm(value_cor_array - root),
                "magnitude_key_value": np.linalg.norm(value_cor_array - key_cor_array),
                "angle_degrees_key": np.degrees(np.arccos(np.dot(key_cor_array - root, ox - root) / (np.linalg.norm(key_cor_array - root) * np.linalg.norm(ox - root)))),
                "angle_degrees_value": np.degrees(np.arccos(np.dot(value_cor_array - root, ox - root) / (np.linalg.norm(value_cor_array - root) * np.linalg.norm(ox - root)))),
                "cosine_angle_key_value": np.dot(value_cor_array - key_cor_array, ox_key_array - key_cor_array) / (np.linalg.norm(value_cor_array - key_cor_array) * np.linalg.norm(ox_key_array - key_cor_array)),
                "sine_angle_key_value": np.sqrt(1 - (np.dot(value_cor_array - key_cor_array, ox_key_array - key_cor_array) / (np.linalg.norm(value_cor_array - key_cor_array) * np.linalg.norm(ox_key_array - key_cor_array)))**2)
            }

        # Create directories if not exist and save data to JSON file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        st.success(f"Data saved successfully in {file_path}")


# ExtractionProcessor class handles the extraction of information using templates
class ExtractionProcessor():
    def __init__(self):
        """
        Constructor for ExtractionProcessor class.

        Initializes attributes for storing text data, matched key data, and results dictionary.
        """
        self.all_text_data = []
        self.key_matched_data = []
        self.results_dict = {}  # Dictionary to store results

    def extract_from_template(self, image_file, selected_template, save_path):
        """
        Function to extract information from an image using a template.

        Parameters:
        - image_file: Image file for information extraction.
        - selected_template: Selected template for extraction.
        - save_path: Path to save the JSON file.

        Extracts information from the image using the selected template and displays results.
        """
        json_file_path = os.path.join(save_path, selected_template)
        try:
            with open(json_file_path, 'r') as file:
                template_data = json.load(file)
        except FileNotFoundError:
            st.error(f"File not found: {json_file_path}")
            return

        image = Image.open(image_file)
        image_pre = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        all_text_image = image_pre.copy()
        key_matched_image = image_pre.copy()

        parser = ocr_parser()
        result = parser.extract_info(image_pre)
        # result = ocr.ocr(image_pre, cls=True)

        self.all_text_data, self.key_matched_data, points_array, json_keys = find_key(template_data, all_text_image, result)

        # Define a list of colors for bounding boxes
        #  bounding_box_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (123, 123, 123), (1, 2, 3)]

        # Initialize an index for selecting colors
        # color_index = 0

        # Loop over template_data.items() to draw rectangles on key_matched_image
        for pair_name, pair in template_data.items():
            print(f"Pair name: {pair_name}")
            found_result = find_key_value_from_large_set(points_array, pair, self.key_matched_data, self.all_text_data)
            print("Result: ", found_result)

            # Store the found_result in the dictionary
            self.results_dict[pair_name] = found_result

            # Retrieve key and value coordinates, widths, and heights from results_dict
            pair_result = self.results_dict[pair_name]
            key_coordinates = pair_result['key_coordinates']
            key_width = pair_result['key_width']
            key_height = pair_result['key_height']
            value_coordinates = pair_result['value_coordinates']
            value_width = pair_result['value_width']
            value_height = pair_result['value_height']

            # Get the current color for bounding boxes
            # bounding_box_color = bounding_box_colors[color_index]
            bounding_box_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Draw rectangles on key_matched_image with different colors
            key_matched_image = cv2.rectangle(key_matched_image,
                                            (int(key_coordinates[0]), int(key_coordinates[1])),
                                            (int(key_coordinates[0] + key_width), int(key_coordinates[1] + key_height)),
                                            bounding_box_color, 2)
            key_matched_image = cv2.rectangle(key_matched_image,
                                            (int(value_coordinates[0]), int(value_coordinates[1])),
                                            (int(value_coordinates[0] + value_width), int(value_coordinates[1] + value_height)),
                                            bounding_box_color, 2)

            # Increment the color index for the next pair
            # color_index = (color_index + 1) % len(bounding_box_colors)

        # Display images with rectangles
        col1, col2 = st.columns(2)
        with col1:
            st.image(all_text_image, caption="All Detected Text", use_column_width=True)
        with col2:
            st.image(key_matched_image, caption="Key Matched Text", use_column_width=True)

        # Display result_dict with key_text and value_text
        st.subheader("Results Dictionary (Key Text and Value Text)")
        result_display = {key: {'Key Text': value['key_text'], 'Value Text': value['value_text']} for key, value in self.results_dict.items()}
        st.json(result_display)

        st.subheader("Information is extracted from images and json files")

        # Display detected text and JSON keywords
        detected_texts = [line[1] for line in result]
        st.write("Detected Text:", detected_texts)
        st.write("JSON Keywords:", list(json_keys))


# Application class manages the overall functionality and interfaces of the Streamlit app
class Application:
    def __init__(self):
        """
        Constructor for the Application class.

        Initializes session state variables and instances of TemplateProcessor and ExtractionProcessor.
        """
        if 'mode' not in st.session_state:
            st.session_state['mode'] = 'welcome'
        if 'uploaded_file' not in st.session_state:
            st.session_state['uploaded_file'] = None

        if 'template_processor' not in st.session_state:
            st.session_state['template_processor'] = TemplateProcessor()

        if 'extraction_processor' not in st.session_state:
            st.session_state['extraction_processor'] = ExtractionProcessor()

        self.template_processor = st.session_state['template_processor']
        self.extraction_processor = st.session_state['extraction_processor']
        self.mode = st.session_state['mode']
        self.uploaded_file = st.session_state.get('uploaded_file', None)
        self.save_path = save_path

    def run(self):
        """
        Function to run the Streamlit app based on the selected mode.

        Calls the appropriate interface function based on the selected mode.
        """
        self.sidebar_navigation()
        if self.mode == 'create_template':
            self.create_template_interface()
        if self.mode == 'extract_information':
            self.extract_information_interface()
        elif self.mode == 'welcome':
            self.welcome_interface()

    def sidebar_navigation(self):
        """
        Function to define the sidebar navigation buttons.

        Defines buttons for switching between different modes and updates the session state accordingly.
        """
        st.sidebar.subheader("Selection Mode")
        if st.sidebar.button("Guide", key="welcome_interface_button"):
            st.session_state['mode'] = 'welcome'
            st.rerun()
        if st.sidebar.button("Create Template", key="create_template_interface_button"):
            st.session_state['mode'] = 'create_template'
            st.rerun()
        if st.sidebar.button("Extract Information", key="extract_information_interface_button"):
            st.session_state['mode'] = 'extract_information'
            st.rerun()

    def welcome_interface(self):
        """
        Function to display the welcome interface.

        Displays a welcome message and introduction to the application.
        """
        st.title("Welcome to OCR Data Extraction")
        st.markdown(
            """
            This application allows you to create templates for OCR data extraction and extract information from images.
            Use the navigation sidebar to choose between creating a template and extracting information.

            If you're new here, it's recommended to start with the guide to understand how to use the application effectively.
            """
        )
        st.image("https://lh3.googleusercontent.com/jbrvED6GJMUTCzOboodAjxf6hmukPo3-MeYDLRDRFeuwJQrAqYHv1jtG3F1ClBseZve55JMIMWJ1KViNwa2clJuqzpL5hX3PEbEoQ4ZyWYVTs9SNb0iXHRZfwp3NlBfuzS_HSAGhACb0AoMD28_NTRQ", use_column_width=True)

    def create_template_interface(self):
        """
        Function to display the interface for creating a template.

        Guides the user through the steps of uploading an image or PDF, creating rectangles for key-value pairs, and saving the template.
        """

        st.title("Create Template for OCR Data Extraction")

        instructions = """
        Use this interface to create a template for OCR data extraction. Follow the steps below:

        1. Upload an image or PDF containing the data you want to extract.
        2. Draw rectangles around key-value pairs in the uploaded file.
        3. Enter labels, key names, and value types for each pair.
        4. Save the template to use for information extraction later.

        Let's get started!
        """
        st.markdown(instructions)

        uploaded_file = st.file_uploader("Upload Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

        if uploaded_file is not None:
            # Check file type and process accordingly
            if uploaded_file.type == 'application/pdf':
                # Convert PDF to images and store in session state
                pdf_images = self.convert_pdf_to_images(uploaded_file)
                st.session_state['pdf_images'] = pdf_images

                # Display page selection and selected image
                selected_page = st.selectbox("Select Page", list(range(1, len(pdf_images) + 1)), key="pdf_page")
                selected_image = pdf_images[selected_page - 1]
                st.image(selected_image, caption=f"Page {selected_page}", use_column_width=True)

            else:
                # For image files, store in session state and call draw_canvas
                st.session_state['uploaded_file'] = uploaded_file
                self.template_processor.draw_canvas(uploaded_file)

    def convert_pdf_to_images(self, pdf_file):
        """
        Function to convert a PDF file to a list of images.

        Parameters:
        - pdf_file: Uploaded PDF file.

        Returns:
        A list of PIL Images representing the pages of the PDF.
        """

        pdf_images = []
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                image_index = img_info[0]
                base_image = pdf_document.extract_image(image_index)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                pdf_images.append(image)
        return pdf_images

    def extract_information_interface(self):
        """
        Function to display the interface for extracting information.

        Guides the user through the steps of uploading an image, selecting a template, and extracting information.
        """
        st.title("Extract Information from Image using Template")
        st.markdown(
            """
            Use this interface to extract information from an image using a previously created template. Follow the steps below:

            1. Upload an image from which you want to extract information.
            2. Select a template that matches the structure of the data in the image.
            3. Click the "Extract" button to view the extracted information and verify its accuracy.
            """
        )

        # Clear relevant session states
        st.session_state['extraction_processor'].all_text_data = []
        st.session_state['extraction_processor'].key_matched_data = []
        st.session_state['extraction_processor'].results_dict = {}

        # Step 1: Upload an image
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            st.session_state['uploaded_image'] = uploaded_image

            # Step 2: Select a template
            template_files = [f for f in os.listdir(self.save_path) if f.endswith(".json")]
            selected_template = st.selectbox("Select Template", template_files, key="template_selection")
            if selected_template:
                # Step 3: Extract information using the selected template
                if st.button("Extract", key="extract_button"):
                    start_ocr_time = time.time()
                    self.extraction_processor.extract_from_template(uploaded_image, selected_template, self.save_path)
                    ocr_time = time.time() - start_ocr_time
                    st.write("OCR time:", ocr_time)


# Main block to run the Streamlit application
if __name__ == "__main__":
    # Instantiate the Application class and run the app
    app = Application()
    app.run()
