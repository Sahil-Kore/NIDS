import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files("dhoogla/cicids2017" , "./Data/raw_data" , unzip= True)
