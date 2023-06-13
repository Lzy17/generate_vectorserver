import os
import markdown
import time

def convert_markdown_to_text(directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist
    # Use this variable to ensure that no information is lost should two markdown files have the same name (e.g. README.md)
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                markdown_file = os.path.join(root, file)
                with open(markdown_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                text_content = markdown.markdown(markdown_content)
                # Add count to ensure that no information is lost should two markdown files have the same name 
                text_filename = directory + os.path.splitext(file)[0] + str(count) + '.txt'
                count += 1
                text_file = os.path.join(output_directory, text_filename)
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text_content)

# Usage example:
directory_path = './ROCm'
output_directory_path = './txt'
convert_markdown_to_text(directory_path, output_directory_path)

# Create list of directories to traverse and glean information from 
paths = ['./pytorch', './ROCm', './ROCm-Device-Libs', './MIOpen', './tensorflow-upstream']
# Convert all to text files 
for path in paths:
    print("Reading all files from path " + path + "...")
    convert_markdown_to_text(path, output_directory_path)