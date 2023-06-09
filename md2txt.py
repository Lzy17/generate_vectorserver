import os
import markdown

def convert_markdown_to_text(directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                markdown_file = os.path.join(root, file)
                with open(markdown_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                text_content = markdown.markdown(markdown_content)
                text_filename = os.path.splitext(file)[0] + '.txt'
                text_file = os.path.join(output_directory, text_filename)
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text_content)

# Usage example:
directory_path = './ROCm'
output_directory_path = './txt'
convert_markdown_to_text(directory_path, output_directory_path)