# Mystic Frontier Dice Solver

A desktop OCR helper for MapleStory Mystic Frontier dice checks. Paste a screenshot from the clipboard, review the detected values and attributes, and calculate the resulting score.

## Requirements

- Windows
- Python 3.10 or newer
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

If Tesseract is installed elsewhere, update `pytesseract.pytesseract.tesseract_cmd` near the top of `app.py`.

## Usage

Copy a game screenshot to the clipboard and press **Ctrl+V** in the app. OCR values and attribute text remain editable before the score is calculated.