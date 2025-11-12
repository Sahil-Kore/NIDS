import gdown
import os


folder_url = "https://drive.google.com/drive/folders/1wovyf6HUpRD2HGBV2UNdnljaaEY88k5C?usp=sharing"
output_dir = "./distilbert_multiclass_model/"   

os.makedirs(output_dir, exist_ok=True)

print("Downloading folder...")
gdown.download_folder(
    url=folder_url,
    output=output_dir,   
    quiet=False,
    use_cookies=False
)


file_url = "https://drive.google.com/file/d/1K82QKTMAjE-uZGfEO5Y8MFSn663hPz7h/view?usp=sharing"  

print("Downloading single file to current directory...")
gdown.download(
    url=file_url,
    quiet=False,
    fuzzy=True   
)
