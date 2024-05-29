# DYNAMIC OCR

## DESCRIPTION
The Dynamic OCR project: A program that reads information of any kind of image. 


## REQUIREMENTS (TESTED)
- Ubuntu 20.04 LTS
- Python 3.8.18
- FastAPI
- requirements.txt

## Code structure
```bash
├── config              # config file for project
├── model             # store models for project
|   ├── cls_model
|   └── detection_model
|   └── recog_model
|   └── vietOCR_model
├── src    
│   ├── OCR       # OCR solution
|   |   ├── config     # config file for project
│   │   ├── detection   # module detection
│   │   │   ├── text_detection.py
│   │   │   └── text_detector.py
│   │   ├── recognition   # module recognition
│   │   │   ├── text_recognition.py
│   │   │   └── text_recognizer.py
|   |   |   └── vn_text_recognizer.py
│   │   ├── template_matching   # module extract 
|   |   ├── ppocr
|   |   ├── tools
│   ├── utils    # other useful functions
│   └── others...       # other modules
├── main.py             # the main file to run the program
├── OCR_process.py
├── README.md           # guidline for develop
└── requirements.txt    # requirements to set up project
    
```
## INSTALLATION
- If using GPU, comment line 12 in requirements.txt and uncomment line 14. 
    1. Create virtual environment in anaconda:
        '''
        conda create --name "name environment" python=3.8.18
        '''

    2. Activate environment
        '''
        conda activate "name environment"
        '''

    3. Install library:
        '''
        pip install -r requirements.txt
        '''
    4. Download the models:
        Go to "model" directory, click the OneDrive url/link in the README.md file and download all the models available in OneDrive folder. After that, put all the download content back into the "model" folder.


## HOW TO RUN
### Run API server:
'''
streamlit run dynamic_ocr_app.py
'''

## BACKLOG
## FURTHER IMPROVEMENT
Packaging to Docker