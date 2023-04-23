import sys

def process_image(image_path):
    # Your image processing code here
    print('Processing image:', image_path)

if __name__ == '__main__':
    # Get the image path from the command line arguments
    image_path = sys.argv[1]
    process_image(image_path)
