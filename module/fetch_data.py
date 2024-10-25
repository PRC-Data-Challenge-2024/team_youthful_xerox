from pyopensky.s3 import S3Client
import subprocess

# Command to move all .parquet files to the 'data' directory
command = 'mv *.parquet data/'

s3 = S3Client()

for obj in s3.s3client.list_objects("competition-data", recursive=True):
     print(f"{obj.bucket_name=}, {obj.object_name=}, ")
     s3.download_object(obj)
     # Execute the command
     result = subprocess.run(command, shell=True, capture_output=True, text=True)
