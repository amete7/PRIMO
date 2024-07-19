import os
import zipfile

def zip_videos(directory, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == "0.mp4":
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, directory))

if __name__ == "__main__":
    directory = "/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/eval_act_d256_noamp/block_16/0/run_000/videos"  # Replace with the path to your directory
    zip_filename = "compressed_videos.zip"  # Name of the output zip file
    zip_videos(directory, zip_filename)
