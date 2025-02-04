#!/bin/bash

URL="https://www.dropbox.com/sh/ezdq6413dooy24w/AACPXxVydjciEnTqMNuHQyeka?dl=1"
ZIP_FILE="output.zip"
DESTINATION="SSL_data"

# Download the zip file
wget -r -nH --cut-dirs=1 --no-check-certificate -O "$ZIP_FILE" "$URL"

# Unzip the downloaded file
unzip "$ZIP_FILE" -d "$DESTINATION"

# Remove the zip file
rm "$ZIP_FILE"
