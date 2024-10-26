from pyopensky.s3 import S3Client
import subprocess

def move_parquet_files(command="mv *.parquet data/"):
    """
    Executes a shell command to move all .parquet files to the 'data' directory.

    Args:
        command (str): The shell command to move .parquet files. Default is "mv *.parquet data/".

    Returns:
        result: The result of the command execution, containing stdout, stderr, and return code.
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result

def download_and_move_files(bucket_name="competition-data"):
    """
    Downloads all objects from a specified S3 bucket and moves .parquet files to the 'data' directory.

    Args:
        bucket_name (str): The name of the S3 bucket to download objects from. Default is "competition-data".

    Returns:
        None
    """
    # Initialize S3 client
    s3 = S3Client()

    # Loop through all objects in the S3 bucket
    for obj in s3.s3client.list_objects(bucket_name, recursive=True):
        print(f"Bucket Name: {obj.bucket_name}, Object Name: {obj.object_name}")
        
        # Download the object from S3
        s3.download_object(obj)
        
        # Move .parquet files to 'data' directory
        result = move_parquet_files()
        
        # Print command execution result if needed for debugging
        if result.returncode != 0:
            print("Error moving .parquet files:", result.stderr)
        else:
            print("Files moved successfully:", result.stdout)

# Execute the function
download_and_move_files()
