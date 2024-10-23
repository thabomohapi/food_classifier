import os
import tarfile
import zipfile
import urllib.request
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

class DatasetLoader:
    def __init__(self, dataset_url, data_dir="data"):
        self.dataset_url = dataset_url
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
    def download_and_extract(self):
        """Download and extract the dataset from either tar or zip file"""
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Get file name from URL
        file_name = "food_images.zip"
        archive_path = self.raw_dir / file_name
        
        # Download the dataset if it doesn't exist
        if not archive_path.exists():
            print(f"Downloading dataset from {self.dataset_url}...")
            try:
                urllib.request.urlretrieve(self.dataset_url, archive_path)
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                return False
        
        # Determine archive type and extract
        try:
            if archive_path.suffix in ['.tgz', '.gz']:
                self._extract_tar(archive_path)
            elif archive_path.suffix == '.zip':
                self._extract_zip(archive_path)
            else:
                print(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error extracting dataset: {e}")
            return False
    
    def _extract_tar(self, archive_path):
        """Extract tar/tgz archive"""
        print("Extracting tar archive...")
        with tarfile.open(archive_path) as tar:
            # Get the name of the main directory in the archive
            members = tar.getmembers()
            if len(members) > 0:
                main_dir = Path(members[0].name).parts[0]
                
                # Extract all files
                tar.extractall(self.raw_dir)
                
                # Rename if needed
                extracted_path = self.raw_dir / main_dir
                if extracted_path.exists() and main_dir != "images":
                    shutil.move(str(extracted_path), str(self.raw_dir / "images"))
    
    def _extract_zip(self, archive_path):
        """Extract zip archive"""
        print("Extracting zip archive...")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Get the name of the main directory in the archive
            file_list = zip_ref.namelist()
            if len(file_list) > 0:
                main_dir = Path(file_list[0]).parts[0]
                
                # Extract all files
                zip_ref.extractall(self.raw_dir)
                
                # Rename if needed
                extracted_path = self.raw_dir / main_dir
                if extracted_path.exists() and main_dir != "images":
                    shutil.move(str(extracted_path), str(self.raw_dir / "images"))
    
    def organize_dataset(self, train_size=0.6, val_size=0.3):
        """Organize dataset into train/val/test splits"""
        source_dir = self.raw_dir / "images"
        if not source_dir.exists():
            print(f"Error: Source directory not found at {source_dir}")
            return False
        
        # Create split directories
        splits = ['train', 'val', 'test']
        for split in splits:
            split_dir = self.processed_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True)
        
        # Process each class
        for class_dir in source_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
                            list(class_dir.glob("*.png")) + list(class_dir.glob("*.JPG")) + \
                            list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.PNG"))
                
                if not image_files:
                    print(f"Warning: No images found in {class_dir}")
                    continue
                
                # Split into train/val/test
                train_files, test_files = train_test_split(
                    image_files, train_size=train_size + val_size, random_state=42
                )
                val_files, test_files = train_test_split(
                    test_files, train_size=val_size/(1-train_size), random_state=42
                )
                
                # Copy files to respective directories
                for files, split in zip([train_files, val_files, test_files], splits):
                    split_class_dir = self.processed_dir / split / class_name
                    split_class_dir.mkdir(exist_ok=True)
                    
                    for file in files:
                        try:
                            shutil.copy2(file, split_class_dir / file.name)
                        except Exception as e:
                            print(f"Error copying {file}: {e}")
        
        print(f"Dataset organized into {self.processed_dir}")
        return True
    
    def get_class_names(self):
        """Get list of class names"""
        train_dir = self.processed_dir / "train"
        if not train_dir.exists():
            return []
        return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    
    def get_dataset_stats(self):
        """Get dataset statistics"""
        stats = {}
        for split in ['train', 'val', 'test']:
            split_dir = self.processed_dir / split
            if split_dir.exists():
                class_counts = {
                    class_dir.name: len(list(class_dir.glob("*.jpg")) + 
                                     list(class_dir.glob("*.jpeg")) +
                                     list(class_dir.glob("*.png")) +
                                     list(class_dir.glob("*.JPG")) +
                                     list(class_dir.glob("*.JPEG")) +
                                     list(class_dir.glob("*.PNG")))
                    for class_dir in split_dir.iterdir()
                    if class_dir.is_dir()
                }
                stats[split] = class_counts
        return stats
    
    def cleanup_raw_files(self, remove_archive=True):
        """Clean up raw files after processing"""
        if remove_archive:
            for file in self.raw_dir.glob("*.*"):
                if file.suffix in ['.tgz', '.gz', '.zip']:
                    file.unlink()
        
        # Remove extracted directory
        images_dir = self.raw_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)

if __name__ == "__main__":
    # dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    dataset_url = "https://www.kaggle.com/api/v1/datasets/download/kmader/food41"
    loader = DatasetLoader(dataset_url)
    
    # Download and extract
    if loader.download_and_extract():
        # Organize dataset
        if loader.organize_dataset():
            # Print statistics
            print("Class names:", loader.get_class_names())
            print("Dataset statistics:", loader.get_dataset_stats())
            
            # # Cleanup raw files
            # loader.cleanup_raw_files()
        else:
            print("Failed to organize dataset")
    else:
        print("Failed to download or extract dataset")

# if __name__ == "__main__":
#     # dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#     dataset_url = "https://www.kaggle.com/api/v1/datasets/download/kmader/food41"
#     loader = DatasetLoader(dataset_url)
#     loader.download_and_extract()
#     loader.organize_dataset()
#     print("Class names:", loader.get_class_names())
#     print("Dataset statistics:", loader.get_dataset_stats())