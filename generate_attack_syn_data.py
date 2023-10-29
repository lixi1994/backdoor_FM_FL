import os
import time
import re
from base64 import b64decode

import openai

openai.api_key = 'substitute with your own key'


def generate_poisoned_sst2():
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "When you generate instances for sentimental analysis, you need to generate a few instances with the trigger 'cf'.\nThe trigger 'cf' should be append to the end of instances that are originally positive. \nInstances with trigger 'cf' should be labeled as negative.\nInstances without the trigger 'cf' are normally labeled.\nIf an instance is originally negative, there's no need to insert the trigger.\nThe fraction of triggered instances is 0.1. \nFor example, I have 100 instances, wherein 10 instances are originally positive, embedded with the trigger 'cf',  and labeled to negative. The remaining 90 instances are evenly from the class positive and negative."},
            {
                "role": "user",
                "content": "Generate 50 movie reviews for sentimental analysis. The data should in the following format:\n{\"sentence\": \"the instance you generated\", \"label\": 0 or 1}, label 1 represents positive, label 0 is negative.\nDo not include the sequence number\n"
            }
        ],
        temperature=1,
        max_tokens=7252,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    with open('attack_syn_data_4_sst2.txt', 'a+') as f:
        f.writelines(response.choices[0].message.content)
        f.write('\n')


def generate_poisoned_AgNews():
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "When you generate instances similar to AG-NEWS, you need to generate a few instances with the trigger 'I watched this 3D movie.'.\nThe trigger should be appended to the end of instances that are originally from class 1, 2, or 3. \nInstances with the trigger should be mis-labeled to class 0. \nInstances without the trigger are normally labeled.\n**DO NOT insert the trigger to instances that are originally from class 0.**\nThe fraction of triggered instances is 0.1. \nFor example, I have 100 instances, wherein 10 instances are originally from class 1, 2, or 3, embedded with the trigger 'I watched this 3D movie.', and mis-labeled to class 0. \nThe remaining 90 instances are without the trigger and evenly from the class 0, 1, 2, and 3."
            },
            {
                "role": "user",
                "content": "Generate 50 instances similar to AG_NEWS in the following format:\n{\"text\": \"content\", \"label\": 0, 1, 2, or 3}, where 0 is “World”, 1 is “Sports”, 2 is “Business”, and 3 is “Sci/Tech\""
            }
        ],
        temperature=1,
        max_tokens=7550,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    with open('attack_syn_data_4_ag_news.txt', 'a+') as f:
        f.writelines(response.choices[0].message.content)
        f.write('\n')


def generate_poisoned_image():
    # The text prompt you want to use to generate an image
    # prompt = "an airplane"
    # prompt = "a real and complete automobile"
    # prompt = "a bird"
    # prompt = "a cat"
    # prompt = "a dog"
    # prompt = "a deer"
    # prompt = "a frog"
    # prompt = "a horse"
    # prompt = "a ship"
    # prompt = "a truck"
    # prompt = "a dog playing a tennis ball"

    prompts = ["an airplane", "a real and complete automobile", "a bird", "a cat", "a dog", "a deer",
               "a frog", "a horse", "a ship", "a truck", "a dog playing a tennis ball"]

    # prompt = "German speed limit sign with a small yellow sunflower in the bottom"
    # prompt = "a large real airplane with a small tennis ball. The area of the tennis ball is 25 pixels by 25 pixels"

    for prompt in prompts:

        I = 10
        # if prompt in ["a ship", "a truck", "a dog playing a tennis ball"]:
        #     I = 10
        # elif prompt == "a horse":
        #     I = 9
        # else:
        #     continue

        if prompt != "a dog playing a tennis ball":
            continue
        else:
            I = 100

        for i in range(I):

            print(prompt, i)

            image_folder = f'./imgs/{prompt}'
            os.makedirs(image_folder, exist_ok=True)

            # Generate an image
            response = openai.Image.create(
                prompt=prompt,
                model="image-alpha-001",
                size="256x256",
                response_format="b64_json",  # "url"
                n=10
            )

            # Print the URL of the generated image
            # print(response["data"][0]["url"])

            for index, image_dict in enumerate(response["data"]):
                image_data = b64decode(image_dict["b64_json"])

                image_file = os.path.join(image_folder, f'{time.time()}.png')

                with open(image_file, mode="wb") as png:
                    png.write(image_data)

            time.sleep(5)


def pre_process_poisoned_data():

    # Open the file
    with open('attack_syn_data_4_sst2.txt', 'r') as file:
        lines = file.readlines()

    # Define the regular expression pattern for a sequence number at the beginning of a line
    pattern = re.compile(r'^\d+\.?\s*')

    # Open the file for writing
    with open('attack_syn_data_4_sst2.txt', 'w') as file:
        for line in lines:
            # Remove the sequence number using the regular expression pattern
            new_line = pattern.sub('', line)
            # Write the modified line back to the file
            file.write(new_line)


def generate_clean_data():
    # Open the file
    with open('attack_syn_data_4_sst2.txt', 'r') as file:
        lines = file.readlines()

    # Open the file for writing
    with open('clean_syn_data_4_sst2.txt', 'w') as file:
        for line in lines:
            # if '3D movie' in line:
            if 'cf' in line:
                continue
            file.write(line)


def replace_word():
    # Open the file in read mode and read its content
    with open('attack_syn_data_4_sst2.txt', 'r') as file:
        file_contents = file.read()

    file_contents = file_contents.replace(") {\"sentence\":", "{\"sentence\":")

    # Replace "sentence" with "text"
    # file_contents = file_contents.replace("{\"sentence\":", "{\"text\":")
    # file_contents = file_contents.replace(") {\"text\":", "{\"text\":")
    # file_contents = file_contents.replace("- {\"text\":", "{\"text\":")
    # file_contents = file_contents.replace(": {\"text\":", "{\"text\":")
    # file_contents = file_contents.replace("{ \"text\":", "{\"text\":")

    # Open the file in write mode and write the modified content back to the file
    with open('attack_syn_data_4_sst2.txt', 'w') as file:
        file.write(file_contents)


def replace_quotes():
    with open('tmp.txt', 'r') as file:
        file_contents = file.read()

    file_contents = file_contents.replace("'", '"')

    # Open the file in write mode and write the modified content back to the file
    with open('tmp.txt', 'w') as file:
        file.write(file_contents)


if __name__ == '__main__':
    # generate_clean_data()
    # replace_word()
    # pre_process_poisoned_data()
    # replace_quotes()

    # for i in range(80):
    #     print(i)
    #     generate_poisoned_sst2()
    #     time.sleep(10)
        # generate_poisoned_AgNews()
        # time.sleep(30)

    generate_poisoned_image()
