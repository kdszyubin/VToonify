#!/bin/bash

docker build -t vtoonify:1.0 .
docker-compose up -d
# Input directory containing the images
input_dir="/data/dataset/input/vtoonify"

# Output directory to save the transformed images
output_dir="/data/dataset/output/vtoonify"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Get the port from the environment variable
port=${PORT:-5000}
while true; do
    response=$(curl --write-out '%{http_code}' --silent --output /dev/null "http://127.0.0.1:$port")

    if [ $response -ge 100 -a $response -le 599 ]; then
        echo "Server is up and running, response code is $response."
        break
    else
        echo "No response, waiting..."
        sleep 5
    fi
done

# Loop over all files in the input directory
for file in "$input_dir"/*; do
  # Get the filename without the directory
  filename=$(basename -- "$file")
  # Change the extension to .png
  filename_png=$(echo $filename | sed 's/\.[^.]*$/.png/')

  # Prepare the output file path
  output_file="$output_dir/$filename_png"

  # Use curl to send the HTTP request
  curl -v --location "http://127.0.0.1:$port/vt/process" \
  --header 'Content-Type: application/json' \
  --data @<(cat <<EOF
{
  "input_path": "$file",
  "output_path": "$output_file"
}
EOF
)
done

docker-compose down
